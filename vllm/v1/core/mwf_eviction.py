import math
import time
from dataclasses import dataclass
from collections.abc import Callable
from typing import TYPE_CHECKING

MWF_DELTA_SLOPE = 0.00043533405552524906
MWF_DELTA_INTERCEPT = 16.975203511045493

if TYPE_CHECKING:
    from vllm._mwf_candidate_index import CandidateKind, MWFCandidateIndex
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

@dataclass(frozen=True)
class MWFScore:
    weight: float
    frequency: float
    prefix_tokens: int
    age_s: float


def compute_mwf_score(
    *,
    prefix_tokens: int,
    last_accessed: float,
    now: float,
    minexp_a: float,
    minexp_b: float,
    minexp_alpha: float,
    minexp_beta: float,
) -> MWFScore:
    """Compute the original MinExp MWF score for one cached block."""
    if prefix_tokens <= 0:
        raise ValueError("MWF prefix_tokens must be positive")

    values = (
        last_accessed,
        now,
        minexp_a,
        minexp_b,
        minexp_alpha,
        minexp_beta,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("MWF parameters and timestamps must be finite")
    if minexp_alpha <= 0 or minexp_beta <= 0:
        raise ValueError("MWF MinExp alpha and beta must be positive")

    age_s = max(0.0, now - last_accessed)
    log_frequency = min(
        -(minexp_a + age_s) / minexp_alpha,
        -(minexp_b + age_s) / minexp_beta,
    )
    frequency = math.exp(log_frequency)
    recompute_cost = (
        MWF_DELTA_SLOPE * prefix_tokens + MWF_DELTA_INTERCEPT
    )
    return MWFScore(
        weight=frequency * recompute_cost,
        frequency=frequency,
        prefix_tokens=prefix_tokens,
        age_s=age_s,
    )

def compute_mwf_score_keys(
    *,
    prefix_tokens: int,
    last_accessed: float,
    time_origin: float,
    minexp_a: float,
    minexp_b: float,
    minexp_alpha: float,
    minexp_beta: float,
) -> tuple[float, float]:
    """Build the two log-score keys for the native MWF index."""
    if prefix_tokens <= 0:
        raise ValueError("MWF prefix_tokens must be positive")

    values = (
        last_accessed,
        time_origin,
        minexp_a,
        minexp_b,
        minexp_alpha,
        minexp_beta,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("MWF parameters and timestamps must be finite")
    if minexp_alpha <= 0 or minexp_beta <= 0:
        raise ValueError("MWF MinExp alpha and beta must be positive")

    cost = MWF_DELTA_SLOPE * prefix_tokens + MWF_DELTA_INTERCEPT
    log_cost = math.log(cost)
    relative_time = last_accessed - time_origin
    keys = (
        log_cost + (relative_time - minexp_a) / minexp_alpha,
        log_cost + (relative_time - minexp_b) / minexp_beta,
    )
    if not all(math.isfinite(key) for key in keys):
        raise ValueError("MWF score keys must be finite")
    return keys

@dataclass
class _BlockMetadata:
    last_accessed: float
    prefix_tokens: int


class MWFEvictionManager:
    def __init__(
        self,
        *,
        minexp_a: float,
        minexp_b: float,
        minexp_alpha: float,
        minexp_beta: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        params = dict(
            minexp_a=minexp_a,
            minexp_b=minexp_b,
            minexp_alpha=minexp_alpha,
            minexp_beta=minexp_beta,
        )
        if not all(math.isfinite(value) for value in params.values()):
            raise ValueError("MWF MinExp parameters must be finite")
        if minexp_alpha <= 0 or minexp_beta <= 0:
            raise ValueError("MWF MinExp alpha and beta must be positive")
        if not callable(clock):
            raise TypeError("MWF clock must be callable")

        self._params = params
        self._clock = clock
        self._time_origin = self._clock()
        if not math.isfinite(self._time_origin):
            raise ValueError("MWF time origin must be finite")
        self._metadata: dict[int, _BlockMetadata] = {}
        self._kind: "type[CandidateKind] | None" = None
        self._index: "MWFCandidateIndex | None" = None

    @staticmethod
    def _check_block_id(block_id: int) -> None:
        if type(block_id) is not int or block_id < 0:
            raise ValueError("MWF block_id must be a nonnegative integer")

    def on_block_cached(self, block_id: int, prefix_tokens: int) -> None:
        self._check_block_id(block_id)
        if type(prefix_tokens) is not int or prefix_tokens <= 0:
            raise ValueError("MWF prefix_tokens must be a positive integer")
        now = self._clock()
        if not math.isfinite(now):
            raise ValueError("MWF timestamp must be finite")
        if now < self._time_origin:
            raise ValueError("MWF cache time must not precede time origin")
        self._metadata[block_id] = _BlockMetadata(now, prefix_tokens)

    def on_block_accessed(self, block_id: int) -> None:
        self._check_block_id(block_id)
        metadata = self._metadata.get(block_id)
        if metadata is None:
            return

        now = self._clock()
        if not math.isfinite(now):
            raise ValueError("MWF timestamp must be finite")
        if now < metadata.last_accessed:
            raise ValueError("MWF access time must not precede last access")
        metadata.last_accessed = now

    def get_score(self, block_id: int, *, now: float | None = None) -> MWFScore:
        self._check_block_id(block_id)
        metadata = self._metadata[block_id]
        now = self._clock() if now is None else now
        if now < metadata.last_accessed:
            raise ValueError("MWF now must not precede last access")
        return compute_mwf_score(
            prefix_tokens=metadata.prefix_tokens,
            last_accessed=metadata.last_accessed,
            now=now,
            **self._params,
        )

    def get_score_keys(self, block_id: int) -> tuple[float, float]:
        self._check_block_id(block_id)
        metadata = self._metadata[block_id]
        return compute_mwf_score_keys(
            prefix_tokens=metadata.prefix_tokens,
            last_accessed=metadata.last_accessed,
            time_origin=self._time_origin,
            **self._params,
        )

    def on_block_reused(self, block_id: int) -> None:
        self._check_block_id(block_id)
        self._metadata.pop(block_id, None)

    def _ensure_native(self) -> None:
        if self._index is not None:
            return

        from vllm._mwf_candidate_index import CandidateKind, MWFCandidateIndex

        index = MWFCandidateIndex(
            alpha=self._params["minexp_alpha"],
            beta=self._params["minexp_beta"],
        )
        self._kind = CandidateKind
        self._index = index

    def on_block_freed(self, block: "KVCacheBlock") -> None:
        self._check_block_id(block.block_id)
        if block.is_null or block.ref_cnt != 0:
            raise ValueError(f"Block {block.block_id} is not a free candidate")

        has_hash = block.block_hash is not None
        tracked = has_hash and block.block_id in self._metadata
        key_a, key_b = self.get_score_keys(block.block_id) if tracked else (0.0, 0.0)

        self._ensure_native()
        assert self._kind is not None and self._index is not None
        if not has_hash:
            kind = self._kind.UNHASHED
        elif tracked:
            kind = self._kind.TRACKED
        else:
            kind = self._kind.MISSING

        self._index.add(block.block_id, kind, key_a, key_b)