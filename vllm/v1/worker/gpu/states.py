# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import torch

from vllm.v1.request import (
    decode_multi_segment_cache_locs,
    encode_multi_segment_cache_locs,
)
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor


def _metadata_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("metadata values must be integers")
    return int(value)


def normalize_multi_segment_cache_locs(
    locs: Sequence[int] | None,
) -> list[int]:
    if locs is None:
        return [0]
    values = [_metadata_int(value) for value in locs]
    try:
        first_padding = values.index(-1)
    except ValueError:
        first_padding = len(values)
    if any(value != -1 for value in values[first_padding:]):
        raise ValueError("multi-segment metadata only allows trailing -1 padding")
    values = values[:first_padding]
    if not values:
        raise ValueError("multi-segment metadata cannot be empty or all padding")
    segments, first_compute_start = decode_multi_segment_cache_locs(values)
    return encode_multi_segment_cache_locs(segments, first_compute_start)


def pad_multi_segment_cache_locs(
    rows: Sequence[Sequence[int] | None],
) -> np.ndarray:
    normalized = [normalize_multi_segment_cache_locs(row) for row in rows]
    if any(value > np.iinfo(np.int32).max for row in normalized for value in row):
        raise ValueError("cache location exceeds int32 metadata capacity")
    width = max((len(row) for row in normalized), default=1)
    padded = np.full((len(normalized), width), -1, dtype=np.int32)
    for index, row in enumerate(normalized):
        padded[index, : len(row)] = row
    return padded


def get_uncached_query_ranges(
    locs: Sequence[int] | None,
    logical_start: int,
    logical_count: int,
) -> tuple[tuple[tuple[int, int], ...], int]:
    """Intersect cached segments with a window and return its complement."""
    logical_start = _metadata_int(logical_start)
    logical_count = _metadata_int(logical_count)
    if logical_start < 0 or logical_count < 0:
        raise ValueError("logical window must be non-negative")
    logical_end = logical_start + logical_count
    canonical = normalize_multi_segment_cache_locs(locs)
    ranges: list[tuple[int, int]] = []
    cursor = logical_start
    for index in range(0, len(canonical) - 1, 2):
        begin, end = canonical[index : index + 2]
        if end <= logical_start:
            continue
        if begin >= logical_end:
            break
        if cursor < begin:
            ranges.append((cursor, min(begin, logical_end)))
        cursor = max(cursor, min(end, logical_end))
    if cursor < logical_end:
        ranges.append((cursor, logical_end))
    return tuple(ranges), sum(end - begin for begin, end in ranges)


@dataclass(frozen=True)
class QueryExecutionPlan:
    logical_start: int
    logical_end: int
    logical_advance: int
    uncached_ranges: tuple[tuple[int, int], ...]
    query_ranges: tuple[tuple[int, int], ...]
    packed_q_count: int
    logits_row: int | None
    msa_enabled: bool
    fallback_reason: str | None

    @property
    def logical_positions(self) -> tuple[int, ...]:
        return tuple(
            position
            for begin, end in self.query_ranges
            for position in range(begin, end)
        )


def build_query_execution_plan(
    locs: Sequence[int] | None,
    logical_start: int,
    logical_count: int,
    *,
    allow_msa: bool = False,
) -> QueryExecutionPlan:
    logical_start = _metadata_int(logical_start)
    logical_count = _metadata_int(logical_count)
    if logical_start < 0 or logical_count < 0:
        raise ValueError("logical window must be non-negative")
    if not isinstance(allow_msa, bool):
        raise ValueError("allow_msa must be a bool")
    canonical = normalize_multi_segment_cache_locs(locs)
    if logical_start < canonical[-1]:
        raise ValueError("logical_start precedes first_compute_start")
    logical_end = logical_start + logical_count
    uncached_ranges, uncached_count = get_uncached_query_ranges(
        canonical, logical_start, logical_count
    )
    has_reuse = uncached_count < logical_count
    msa_enabled = allow_msa and has_reuse
    query_ranges = (
        uncached_ranges
        if msa_enabled
        else (((logical_start, logical_end),) if logical_count else ())
    )
    if msa_enabled and (
        not query_ranges or query_ranges[-1][1] != logical_end
    ):
        raise ValueError("MSA query must include the final logical token")
    packed_q_count = sum(end - begin for begin, end in query_ranges)
    return QueryExecutionPlan(
        logical_start=logical_start,
        logical_end=logical_end,
        logical_advance=logical_count,
        uncached_ranges=uncached_ranges,
        query_ranges=query_ranges,
        packed_q_count=packed_q_count,
        logits_row=packed_q_count - 1 if packed_q_count else None,
        msa_enabled=msa_enabled,
        fallback_reason=(
            "msa_execution_disabled" if has_reuse and not allow_msa else None
        ),
    )


def build_dense_query_execution_plan(
    logical_start: int,
    logical_count: int,
) -> QueryExecutionPlan:
    """Build a dense plan without consulting cached-segment metadata."""
    logical_start = _metadata_int(logical_start)
    logical_count = _metadata_int(logical_count)
    if logical_start < 0 or logical_count < 0:
        raise ValueError("logical window must be non-negative")
    logical_end = logical_start + logical_count
    ranges = ((logical_start, logical_end),) if logical_count else ()
    return QueryExecutionPlan(
        logical_start=logical_start,
        logical_end=logical_end,
        logical_advance=logical_count,
        uncached_ranges=ranges,
        query_ranges=ranges,
        packed_q_count=logical_count,
        logits_row=logical_count - 1 if logical_count else None,
        msa_enabled=False,
        fallback_reason=None,
    )


@dataclass(frozen=True)
class BatchQueryLayout:
    query_start_loc: tuple[int, ...]
    logits_indices: tuple[int | None, ...]
    logical_ends: tuple[int, ...]
    num_actual_tokens: int
    plans: tuple[QueryExecutionPlan, ...]

    @property
    def logical_positions(self) -> tuple[int, ...]:
        return tuple(
            position for plan in self.plans for position in plan.logical_positions
        )


def build_batch_query_layout(
    plans: Sequence[QueryExecutionPlan],
) -> BatchQueryLayout:
    query_start_loc = [0]
    logits_indices: list[int | None] = []
    logical_ends: list[int] = []
    snapshot = tuple(plans)
    for plan in snapshot:
        if (
            plan.logical_start < 0
            or plan.logical_advance < 0
            or plan.packed_q_count < 0
            or plan.logical_end - plan.logical_start != plan.logical_advance
            or plan.packed_q_count != sum(
                end - begin for begin, end in plan.query_ranges
            )
            or plan.logits_row != (
                plan.packed_q_count - 1 if plan.packed_q_count else None
            )
        ):
            raise ValueError("Inconsistent query execution plan")
        cursor = plan.logical_start
        for begin, end in plan.query_ranges:
            if begin < cursor or end <= begin or end > plan.logical_end:
                raise ValueError("Invalid query execution ranges")
            cursor = end
        if plan.packed_q_count and cursor != plan.logical_end:
            raise ValueError("Query execution plan omits the final logical token")
        offset = query_start_loc[-1]
        logits_indices.append(
            offset + plan.logits_row if plan.logits_row is not None else None
        )
        logical_ends.append(plan.logical_end)
        query_start_loc.append(offset + plan.packed_q_count)
    return BatchQueryLayout(
        query_start_loc=tuple(query_start_loc),
        logits_indices=tuple(logits_indices),
        logical_ends=tuple(logical_ends),
        num_actual_tokens=query_start_loc[-1],
        plans=snapshot,
    )



class RequestState:
    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        num_speculative_steps: int,
        vocab_size: int,
        device: torch.device,
        num_prefill_lookahead: int = 1,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.num_speculative_steps = num_speculative_steps
        self.vocab_size = vocab_size
        self.device = device

        self.req_id_to_index: dict[str, int] = {}
        self.index_to_req_id: dict[int, str] = {}
        self.free_indices = list(range(max_num_reqs))

        # NOTE(woosuk): This tensor can be extremely large (e.g., several GBs)
        # depending on the configured max_num_reqs and max_model_len.
        # To save GPU memory, we use UVA instead of GPU for this tensor.
        self.all_token_ids = StagedWriteTensor(
            (self.max_num_reqs, self.max_model_len),
            dtype=torch.int32,
            device=device,
            uva_instead_of_gpu=True,
        )
        # NOTE(woosuk): Distinguish clearly between prompt_len and prefill_len:
        # - prompt_len: Number of tokens in the user-provided prompt.
        # - prefill_len: Number of tokens passed into the model runner.
        #   This can include the prompt and additional partial output tokens,
        #   so prefill_len >= prompt_len.
        # Usually, prefill_len equals prompt_len, but in cases such as resumption after
        # preemption, prefill_len may be greater. Differentiating between these values
        # is crucial, as certain features such as prompt logprobs or frequency penalties
        # must treat prompt and output tokens separately.
        self.prompt_len = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)
        self.prefill_len = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)
        # total_len = prompt_len + output_len. It grows as the request progresses.
        self.total_len = StagedWriteTensor(
            self.max_num_reqs, dtype=torch.int32, device=device
        )

        # Number of computed tokens.
        self.num_computed_prefill_tokens = np.zeros(self.max_num_reqs, dtype=np.int32)
        self.num_computed_tokens = StagedWriteTensor(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        # Optimistic CPU mirror of num_computed_tokens (upper bound on GPU value).
        self.num_computed_tokens_np = np.zeros(self.max_num_reqs, dtype=np.int32)
        self.continuum_suffix_computed_block_locs: list[list[int]] = [
            [0] for _ in range(self.max_num_reqs)
        ]

        # Last sampled tokens.
        self.last_sampled_tokens = torch.zeros(
            self.max_num_reqs, 1, dtype=torch.int64, device=device
        )

        # Max total seq length (prompt_len + max_tokens).
        self.max_seq_len = np.zeros(self.max_num_reqs, dtype=np.int32)

        # Draft tokens.
        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )

        self.next_prefill_tokens = torch.zeros(
            num_prefill_lookahead, self.max_num_reqs, dtype=torch.int32, device=device
        )

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    def add_request(
        self,
        req_id: str,
        prompt_len: int,
        all_token_ids: list[int],
        num_computed_tokens: int,
        max_tokens: int,
        continuum_suffix_computed_block_locs: Sequence[int] | None = None,
    ) -> None:
        suffix_locs = normalize_multi_segment_cache_locs(
            continuum_suffix_computed_block_locs
        )
        assert len(self.free_indices) > 0, "No free indices"
        req_idx = self.free_indices.pop()
        self.req_id_to_index[req_id] = req_idx
        self.index_to_req_id[req_idx] = req_id

        self.max_seq_len[req_idx] = prompt_len + max_tokens
        self.prompt_len.np[req_idx] = prompt_len
        prefill_len = len(all_token_ids)
        assert prefill_len >= prompt_len, (
            f"prefill_len {prefill_len} < prompt_len {prompt_len}"
        )
        self.prefill_len.np[req_idx] = prefill_len
        self.total_len.stage_write_elem(req_idx, prefill_len)
        self.all_token_ids.stage_write(req_idx, 0, all_token_ids)
        self.num_computed_prefill_tokens[req_idx] = num_computed_tokens
        self.num_computed_tokens_np[req_idx] = num_computed_tokens
        self.num_computed_tokens.stage_write_elem(req_idx, num_computed_tokens)
        self.continuum_suffix_computed_block_locs[req_idx] = suffix_locs

        self.draft_tokens[req_idx].zero_()

    def apply_staged_writes(self) -> None:
        self.prompt_len.copy_to_uva()
        self.prefill_len.copy_to_uva()
        self.total_len.apply_write()
        self.all_token_ids.apply_write()
        self.num_computed_tokens.apply_write()

    def remove_request(self, req_id: str) -> int | None:
        """Return the freed slot index, or None if the request was not found."""
        req_idx = self.req_id_to_index.pop(req_id, None)
        if req_idx is None:
            return None
        self.index_to_req_id.pop(req_idx, None)
        self.continuum_suffix_computed_block_locs[req_idx] = [0]
        self.free_indices.append(req_idx)
        return req_idx
