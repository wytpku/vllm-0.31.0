# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from vllm.v1.request import (
    decode_multi_segment_cache_locs,
    encode_multi_segment_cache_locs,
    reset_multi_segment_cache_state,
)
from vllm.v1.worker.gpu.states import (
    RequestState,
    build_batch_query_layout,
    build_dense_query_execution_plan,
    build_query_execution_plan,
    get_uncached_query_ranges,
    normalize_multi_segment_cache_locs,
    pad_multi_segment_cache_locs,
)

pytestmark = pytest.mark.cpu_test


class TestMultiSegmentMetadata(unittest.TestCase):
    def test_codec_preserves_arbitrary_half_open_ranges_and_legacy_empty_pairs(self):
        segments = [(48, 80), (96, 112), (144, 160)]
        encoded = encode_multi_segment_cache_locs(segments, 32)
        self.assertEqual(encoded, [48, 80, 96, 112, 144, 160, 32])
        self.assertEqual(decode_multi_segment_cache_locs(encoded), (segments, 32))
        self.assertEqual(decode_multi_segment_cache_locs([48, 80, 0, 0, 32]),
                         ([(48, 80)], 32))
        self.assertEqual(decode_multi_segment_cache_locs([32]), ([], 32))

    def test_invalid_structure_and_non_integer_locations_are_rejected(self):
        for locs in (
            [], [0, 1], [48, 32, 0], [48, 80, 64, 96, 32],
            [16, 32, 20], [0, 0, -1], [1.0], [True], ["0"],
        ):
            with self.subTest(locs=locs), self.assertRaises(ValueError):
                decode_multi_segment_cache_locs(locs)
        with self.assertRaises(ValueError):
            encode_multi_segment_cache_locs([(1, 2.0)], 0)

    def test_normalization_only_accepts_trailing_padding(self):
        self.assertEqual(normalize_multi_segment_cache_locs(None), [0])
        self.assertEqual(
            normalize_multi_segment_cache_locs([48, 80, 0, 0, 32, -1, -1]),
            [48, 80, 32],
        )
        for row in ([], [-1], [32, -1, 0], [48, 80, 32.0], [False]):
            with self.subTest(row=row), self.assertRaises(ValueError):
                normalize_multi_segment_cache_locs(row)

    def test_window_complement_matches_per_token_reference(self):
        for mask in range(64):
            cached = {position for position in range(6) if mask & (1 << position)}
            segments = []
            for position in sorted(cached):
                if segments and segments[-1][1] == position:
                    segments[-1] = (segments[-1][0], position + 1)
                else:
                    segments.append((position, position + 1))
            locs = encode_multi_segment_cache_locs(segments, 0)
            for start in range(10):
                for count in range(10):
                    ranges, actual = get_uncached_query_ranges(locs, start, count)
                    positions = tuple(p for a, b in ranges for p in range(a, b))
                    expected = tuple(p for p in range(start, start + count)
                                     if p not in cached)
                    self.assertEqual(positions, expected)
                    self.assertEqual(actual, len(expected))

    def test_packed_q_preserves_logical_positions_and_dense_default(self):
        locs = [48, 80, 32]
        plan = build_query_execution_plan(locs, 32, 65, allow_msa=True)
        self.assertEqual(plan.query_ranges, ((32, 48), (80, 97)))
        self.assertEqual(plan.logical_advance, 65)
        self.assertEqual(plan.packed_q_count, 33)
        self.assertEqual(plan.logits_row, 32)
        self.assertEqual(plan.logical_positions, (*range(32, 48), *range(80, 97)))
        dense = build_query_execution_plan(locs, 32, 65)
        self.assertEqual(dense.query_ranges, ((32, 97),))
        self.assertEqual(dense.packed_q_count, 65)
        self.assertEqual(dense.fallback_reason, "msa_execution_disabled")
        self.assertEqual(build_dense_query_execution_plan(32, 65),
                         build_query_execution_plan(None, 32, 65))

    def test_last_logical_token_must_be_computed(self):
        with self.assertRaisesRegex(ValueError, "final logical token"):
            build_query_execution_plan([48, 96, 32], 32, 64, allow_msa=True)
        empty = build_query_execution_plan([32], 32, 0, allow_msa=True)
        self.assertEqual(empty.query_ranges, ())
        self.assertIsNone(empty.logits_row)
        self.assertFalse(empty.msa_enabled)

    def test_invalid_windows_and_switch_types_are_rejected(self):
        for start, count in ((-1, 1), (0, -1), (True, 1), (0, 1.0)):
            with self.subTest(start=start, count=count), self.assertRaises(ValueError):
                build_query_execution_plan(None, start, count)
        with self.assertRaises(ValueError):
            build_query_execution_plan([32], 16, 65)
        with self.assertRaises(ValueError):
            build_query_execution_plan(None, 0, 1, allow_msa=1)

    def test_batch_layout_snapshots_packed_rows_and_logical_ends(self):
        first = build_query_execution_plan([48, 80, 32], 32, 65, allow_msa=True)
        plans = [first, build_dense_query_execution_plan(0, 2)]
        layout = build_batch_query_layout(plans)
        plans.clear()
        self.assertEqual(layout.query_start_loc, (0, 33, 35))
        self.assertEqual(layout.logits_indices, (32, 34))
        self.assertEqual(layout.logical_ends, (97, 2))
        self.assertEqual(layout.num_actual_tokens, 35)
        self.assertEqual(len(layout.plans), 2)
        self.assertEqual(layout.logical_positions[-2:], (0, 1))
        with self.assertRaises(FrozenInstanceError):
            setattr(first, "logical_end", 99)
        self.assertEqual(build_batch_query_layout(()).query_start_loc, (0,))

    def test_batch_layout_rejects_inconsistent_plans(self):
        plan = build_dense_query_execution_plan(32, 16)
        for bad in (
            replace(plan, packed_q_count=15),
            replace(plan, logits_row=0),
            replace(plan, query_ranges=((31, 47),)),
            replace(plan, logical_advance=-16, logical_end=16),
        ):
            with self.subTest(plan=bad), self.assertRaises(ValueError):
                build_batch_query_layout((bad,))

    def test_reset_clears_candidates_and_stores_canonical_locations(self):
        request = SimpleNamespace(continuum_multiseg_cached_blocks=object())
        reset_multi_segment_cache_state(request, 32)
        self.assertEqual(request.continuum_suffix_computed_block_locs, [32])
        self.assertIsNone(request.continuum_multiseg_cached_blocks)


class TestMultiSegmentRequestState(unittest.TestCase):
    def state(self):
        state = RequestState.__new__(RequestState)
        state.free_indices = [0, 1]
        state.req_id_to_index = {}
        state.index_to_req_id = {}
        state.max_seq_len = [0, 0]
        state.prompt_len = SimpleNamespace(np=[0, 0])
        state.prefill_len = SimpleNamespace(np=[0, 0])
        state.total_len = Mock()
        state.all_token_ids = Mock()
        state.num_computed_prefill_tokens = [0, 0]
        state.num_computed_tokens_np = [0, 0]
        state.num_computed_tokens = Mock()
        state.continuum_suffix_computed_block_locs = [[0], [0]]
        state.draft_tokens = [Mock(), Mock()]
        return state

    def test_request_metadata_is_copied_and_reset_on_slot_reuse(self):
        state = self.state()
        locs = [48, 80, 32]
        state.add_request("first", 97, list(range(97)), 32, 8, locs)
        locs[0] = 0
        self.assertEqual(state.continuum_suffix_computed_block_locs[1],
                         [48, 80, 32])
        self.assertEqual(state.remove_request("first"), 1)
        self.assertEqual(state.continuum_suffix_computed_block_locs[1], [0])
        state.add_request("second", 1, [1], 0, 8)
        self.assertEqual(state.continuum_suffix_computed_block_locs[1], [0])
        self.assertIsNone(state.remove_request("missing"))

    def test_bad_metadata_is_rejected_before_request_state_is_mutated(self):
        state = self.state()
        with self.assertRaises(ValueError):
            state.add_request("bad", 1, [1], 0, 8, [0, -1, 0])
        self.assertEqual(state.free_indices, [0, 1])
        self.assertEqual(state.req_id_to_index, {})
        state.all_token_ids.stage_write.assert_not_called()


class TestMultiSegmentPadding(unittest.TestCase):
    def test_padding_preserves_integer_types_and_independent_rows(self):
        rows = [[32], [np.int32(48), np.int64(80), 32], None]
        padded = pad_multi_segment_cache_locs(rows)
        rows[0][0] = 99
        self.assertEqual(padded.tolist(), [[32, -1, -1], [48, 80, 32], [0, -1, -1]])
        self.assertEqual(padded.dtype, np.int32)
        self.assertEqual(pad_multi_segment_cache_locs(()).shape, (0, 1))

    def test_padding_refuses_bool_and_int32_overflow(self):
        for row in ([np.bool_(True)], [2**31]):
            with self.subTest(row=row), self.assertRaises(ValueError):
                pad_multi_segment_cache_locs((row,))
