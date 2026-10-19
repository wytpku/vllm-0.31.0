# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING

MWF_DELTA_SLOPE = 0.00043533405552524906
MWF_DELTA_INTERCEPT = 16.975203511045493

if TYPE_CHECKING:
    from vllm._mwf_candidate_index import CandidateKind, MWFCandidateIndex
    from vllm.v1.core.block_pool import BlockPool
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
    recompute_cost = MWF_DELTA_SLOPE * prefix_tokens + MWF_DELTA_INTERCEPT
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
        updated = _BlockMetadata(now, prefix_tokens)
        self._refresh_tracked_candidate(block_id, updated)
        self._metadata[block_id] = updated

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
        updated = _BlockMetadata(now, metadata.prefix_tokens)
        self._refresh_tracked_candidate(block_id, updated)
        self._metadata[block_id] = updated

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
        if self._index is not None:
            assert self._kind is not None
            self._index.update(block_id, self._kind.UNHASHED)
        self._metadata.pop(block_id, None)

    def on_block_allocated(self, block_id: int) -> None:
        self._check_block_id(block_id)
        if self._index is not None:
            self._index.erase(block_id)

    def on_block_moved(self, src_block_id: int, dst_block_id: int) -> None:
        """Transfer the score when CoW moves live cache entries between pages.

        Call after moving hashes. The pool must resync on a native update
        failure, since the metadata already describes the new hash ownership.
        """
        self._check_block_id(src_block_id)
        self._check_block_id(dst_block_id)
        if src_block_id == dst_block_id:
            raise ValueError("MWF content migration requires distinct blocks")
        metadata = self._metadata.pop(src_block_id, None)
        self._metadata.pop(dst_block_id, None)
        if metadata is not None:
            self._metadata[dst_block_id] = metadata
        if self._index is not None:
            assert self._kind is not None
            self._index.update(src_block_id, self._kind.UNHASHED)
            if metadata is None:
                self._index.update(dst_block_id, self._kind.MISSING)
            else:
                self._refresh_tracked_candidate(dst_block_id, metadata)

    def reset(self) -> None:
        if self._index is not None:
            self._index.clear()
        self._metadata.clear()

    def resync(self, free_blocks: Iterable["KVCacheBlock"]) -> None:
        from vllm._mwf_candidate_index import CandidateKind, MWFCandidateIndex

        rebuilt = MWFCandidateIndex(
            alpha=self._params["minexp_alpha"],
            beta=self._params["minexp_beta"],
        )
        for block in free_blocks:
            self._check_block_id(block.block_id)
            if block.is_null or block.ref_cnt != 0:
                raise ValueError(f"Block {block.block_id} is not a free candidate")

            key_a = key_b = 0.0
            if block.block_hash is None:
                kind = CandidateKind.UNHASHED
            elif block.block_id in self._metadata:
                kind = CandidateKind.TRACKED
                key_a, key_b = self.get_score_keys(block.block_id)
            else:
                kind = CandidateKind.MISSING

            rebuilt.add(block.block_id, kind, key_a, key_b)

        self._kind = CandidateKind
        self._index = rebuilt

    def select_blocks(
        self, num_blocks: int, *, pool: "BlockPool"
    ) -> list["KVCacheBlock"]:
        if type(num_blocks) is not int or num_blocks < 0:
            raise ValueError("num_blocks must be a nonnegative integer")
        if num_blocks == 0:
            return []

        queue = pool.free_block_queue
        num_free = pool.get_num_free_blocks()
        if num_blocks > num_free:
            raise ValueError("Not enough free blocks")

        now = self._clock()
        elapsed = now - self._time_origin
        if not math.isfinite(now) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("MWF elapsed time must be finite and nonnegative")

        try:
            if self._index is None or self._index.size() != num_free:
                raise RuntimeError("MWF index/free queue size mismatch")

            block_ids = self._index.pop_n(num_blocks, elapsed)
            if len(block_ids) != num_blocks or len(set(block_ids)) != num_blocks:
                raise RuntimeError("MWF returned invalid candidate IDs")

            selected = []
            for block_id in block_ids:
                if not 0 <= block_id < len(pool.blocks):
                    raise RuntimeError("MWF candidate ID is outside this pool")
                block = pool.blocks[block_id]
                prev, nxt = block.prev_free_block, block.next_free_block
                if (
                    block.block_id != block_id
                    or block.pool is not pool
                    or block.is_null
                    or block.ref_cnt != 0
                    or prev is None
                    or nxt is None
                    or prev.next_free_block is not block
                    or nxt.prev_free_block is not block
                    or not (prev is queue.fake_free_list_head or prev.pool is pool)
                    or not (nxt is queue.fake_free_list_tail or nxt.pool is pool)
                ):
                    raise RuntimeError("MWF selected an invalid free block")
                selected.append(block)
        except Exception:
            snapshot = list(islice(queue.iter_blocks_after(None), num_free + 1))
            if (
                len(snapshot) != num_free
                or snapshot[-1].next_free_block is not queue.fake_free_list_tail
            ):
                raise RuntimeError("Cannot rebuild MWF from an invalid free queue")
            self.resync(snapshot)
            raise

        return selected

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

    def _refresh_tracked_candidate(
        self, block_id: int, metadata: _BlockMetadata
    ) -> None:
        if self._index is None:
            return

        key_a, key_b = compute_mwf_score_keys(
            prefix_tokens=metadata.prefix_tokens,
            last_accessed=metadata.last_accessed,
            time_origin=self._time_origin,
            **self._params,
        )
        assert self._kind is not None
        self._index.update(block_id, self._kind.TRACKED, key_a, key_b)
