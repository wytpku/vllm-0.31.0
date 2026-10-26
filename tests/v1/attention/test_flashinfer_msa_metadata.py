# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tensor contracts for MSA planning; FlashInfer wrappers are spies."""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("flashinfer")

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends import flashinfer as backend
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheLayout
from vllm.v1.worker.gpu.states import build_query_execution_plan

pytestmark = pytest.mark.cpu_test

BLOCK_TABLE = (9, 3, 7, 1, 8, 4, 6, 2, 5)


class PlanSpy:
    instances = []

    def __init__(self, workspace, kv_layout, backend="auto"):
        self.workspace = workspace
        self.kv_layout = kv_layout
        self.backend = backend
        self.plan_args = None
        self.instances.append(self)

    def plan(self, **kwargs):
        if self.plan_args is not None:
            raise AssertionError("An MSA wrapper cannot overwrite a previous plan")
        self.plan_args = kwargs

    def run(self, *args, **kwargs):
        raise AssertionError("Layer-2 tests cannot execute attention")


class TestFlashInferMSAMetadata(unittest.TestCase):
    def setUp(self):
        PlanSpy.instances.clear()
        self.enterContext(
            patch.object(backend, "BatchPrefillWithPagedKVCacheWrapper", PlanSpy)
        )

    def builder(self, page_size=16, dtype=torch.float16, layout=KVCacheLayout.LBNHC):
        builder = object.__new__(backend.FlashInferMetadataBuilder)
        builder.device = torch.device("cpu")
        builder.kv_cache_spec = FullAttentionSpec(
            block_size=page_size, num_kv_heads=2, head_size=128, dtype=dtype
        )
        builder.cache_config = SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: layout
        )
        builder.page_size = page_size
        builder.num_qo_heads = 4
        builder.num_kv_heads = 2
        builder.head_dim = 128
        builder.use_dcp = builder.has_sinks = builder.enable_cuda_graph = False
        builder.window_left = -1
        builder.kv_cache_dtype = dtype
        builder.model_config = SimpleNamespace(dtype=dtype)
        builder.global_hyperparameters = SimpleNamespace(has_same_all_params=True)
        builder.sm_scale = 128**-0.5
        builder.logits_soft_cap = None
        builder.q_data_type_prefill = builder.q_data_type_decode = dtype
        builder.prefill_fixed_split_size = -1
        builder.disable_split_kv = False
        builder.use_xqa = builder.use_trtllm_gen_varlen_decode = False
        builder.reorder_batch_threshold = 1
        workspace = torch.empty(64, dtype=torch.uint8)
        builder._get_workspace_buffer = lambda: workspace
        return builder

    def common(self, page_size=16, three_ranges=False, partial_tail=True):
        p = page_size
        start = 2 * p
        if three_ranges:
            locs = [3 * p, 5 * p, 6 * p, 7 * p, start]
            end = 8 * p + int(partial_tail)
        else:
            locs = [3 * p, 5 * p, start]
            end = 6 * p + int(partial_tail)
        plan = build_query_execution_plan(locs, start, end - start, allow_msa=True)
        positions = plan.logical_positions
        slots = [BLOCK_TABLE[pos // p] * p + pos % p for pos in positions]
        return CommonAttentionMetadata(
            query_start_loc=torch.tensor([0, plan.packed_q_count], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor(
                [0, plan.packed_q_count], dtype=torch.int32
            ),
            seq_lens=torch.tensor([end], dtype=torch.int32),
            num_reqs=1,
            num_actual_tokens=plan.packed_q_count,
            max_query_len=plan.packed_q_count,
            max_seq_len=end,
            block_table_tensor=torch.tensor([BLOCK_TABLE], dtype=torch.int32),
            slot_mapping=torch.tensor(slots, dtype=torch.int64),
            positions=torch.tensor(positions, dtype=torch.int64),
            causal=True,
            query_execution_plans=(plan,),
        )

    def test_runtime_page_dtype_and_layout_produce_independent_segment_plans(self):
        for page_size in (16, 32, 64):
            for dtype in (torch.float16, torch.bfloat16):
                for layout in (KVCacheLayout.LBNHC, KVCacheLayout.LBHNC):
                    with self.subTest(
                        page_size=page_size, dtype=dtype, layout=layout
                    ):
                        builder = self.builder(page_size, dtype, layout)
                        common = self.common(page_size)
                        metadata = builder.build(0, common)
                        first, second = metadata.msa_calls
                        self.assertEqual(first.q_slice, slice(0, page_size))
                        self.assertEqual(
                            second.q_slice, slice(page_size, 2 * page_size + 1)
                        )
                        self.assertEqual((first.kv_len, second.kv_len),
                                         (3 * page_size, 6 * page_size + 1))
                        for call, pages, last in (
                            (first, 3, page_size), (second, 7, 1)
                        ):
                            args = call.wrapper.plan_args
                            self.assertEqual(args["paged_kv_indices"].tolist(),
                                             list(BLOCK_TABLE[:pages]))
                            self.assertEqual(
                                args["paged_kv_last_page_len"].tolist(), [last]
                            )
                            self.assertEqual(args["seq_lens"].tolist(), [call.kv_len])
                            self.assertEqual(args["page_size"], page_size)
                            self.assertEqual(args["kv_data_type"], dtype)
                            self.assertFalse(args["non_blocking"])
                        self.assertIsNot(first.wrapper, second.wrapper)
                        self.assertNotEqual(first.workspace.data_ptr(),
                                            second.workspace.data_ptr())
                        self.assertEqual(metadata.num_actual_tokens, 2 * page_size + 1)
                        self.assertEqual(metadata.num_prefills, 1)
                        self.assertEqual(metadata.num_decodes, 0)
                        expected_layout = (
                            "NHD" if layout == KVCacheLayout.LBNHC else "HND"
                        )
                        self.assertEqual(first.wrapper.kv_layout, expected_layout)

    def test_multiple_holes_and_full_tail_preserve_packed_coverage(self):
        for partial in (False, True):
            with self.subTest(partial=partial):
                metadata = self.builder().build(
                    0, self.common(three_ranges=True, partial_tail=partial)
                )
                calls = metadata.msa_calls
                self.assertEqual([call.kv_len for call in calls],
                                 [48, 96, 128 + int(partial)])
                rows = [row for call in calls
                        for row in range(call.q_slice.start, call.q_slice.stop)]
                self.assertEqual(rows, list(range(metadata.num_actual_tokens)))

    def test_later_table_slot_updates_and_builds_do_not_mutate_snapshots(self):
        builder = self.builder()
        common = self.common()
        first = builder.build(0, common)
        slots = first.slot_mapping.tolist()
        common.block_table_tensor.zero_()
        common.slot_mapping.zero_()
        second = builder.build(0, self.common())
        self.assertEqual(first.slot_mapping.tolist(), slots)
        self.assertEqual(first.msa_calls[0].page_ids.tolist(), list(BLOCK_TABLE[:3]))
        all_calls = first.msa_calls + second.msa_calls
        self.assertEqual(len({id(call.wrapper) for call in all_calls}), 4)
        self.assertEqual(len({call.workspace.data_ptr() for call in all_calls}), 4)

    def test_unsupported_attention_configuration_is_rejected_before_planning(self):
        for name, value in (
            ("use_dcp", True), ("has_sinks", True), ("window_left", 16),
            ("head_dim", 64), ("kv_cache_dtype", torch.float32),
            ("q_data_type_prefill", torch.float32), ("page_size", 128),
            ("attention_chunk_size", 16), ("non_causal", True),
        ):
            with self.subTest(name=name):
                builder = self.builder()
                if name in ("attention_chunk_size", "non_causal"):
                    builder.kv_cache_spec = replace(
                        builder.kv_cache_spec, **{name: value}
                    )
                else:
                    setattr(builder, name, value)
                before = len(PlanSpy.instances)
                with self.assertRaises(NotImplementedError):
                    builder.build(0, self.common())
                self.assertEqual(len(PlanSpy.instances), before)

    def test_inconsistent_positions_slots_lengths_or_pages_are_rejected(self):
        for part in (
            "positions", "slots", "seq_len", "boundaries", "gpu_boundaries",
            "cpu_seq_len", "slot_device", "slot_dtype", "pages", "count",
        ):
            with self.subTest(part=part):
                common = self.common()
                if part == "positions":
                    common.positions[0] = 0
                elif part == "slots":
                    common.slot_mapping[0] = -1
                elif part == "seq_len":
                    common.seq_lens[0] -= 1
                elif part == "boundaries":
                    common.query_start_loc_cpu[1] -= 1
                elif part == "gpu_boundaries":
                    common.query_start_loc[1] -= 1
                elif part == "cpu_seq_len":
                    common.seq_lens_cpu_upper_bound = torch.tensor(
                        [common.max_seq_len - 1], dtype=torch.int32
                    )
                elif part == "slot_device":
                    common.slot_mapping = common.slot_mapping.to(device="meta")
                elif part == "slot_dtype":
                    common.slot_mapping = common.slot_mapping.to(dtype=torch.float32)
                elif part == "pages":
                    common.block_table_tensor[0, 0] = 0
                else:
                    common.num_actual_tokens -= 1
                with self.assertRaises(ValueError):
                    self.builder().build(0, common)

    def test_cascade_graph_and_multi_request_planning_are_rejected(self):
        builder = self.builder()
        common = self.common()
        with self.assertRaises(NotImplementedError):
            builder.build(16, common)
        with self.assertRaises(NotImplementedError):
            builder.build(0, common, fast_build=True)
        builder.enable_cuda_graph = True
        with self.assertRaises(NotImplementedError):
            builder.build(0, common)
        with self.assertRaises(NotImplementedError):
            builder._plan_msa_calls(common)
        builder.enable_cuda_graph = False
        common.num_reqs = 2
        with self.assertRaises(NotImplementedError):
            builder.build(0, common)

    def test_forward_guard_runs_before_scales_or_attention_are_touched(self):
        metadata = self.builder().build(0, self.common())
        impl = object.__new__(backend.FlashInferImpl)
        with self.assertRaisesRegex(NotImplementedError, "layer 3"):
            impl.forward(None, None, None, None, None, metadata, None)
        self.assertEqual(vars(impl), {})

    def test_default_metadata_reaches_regular_dense_dispatch(self):
        common = self.common()
        common.query_execution_plans = None
        with patch.object(backend, "split_decodes_and_prefills",
                          side_effect=RuntimeError("regular dispatch")):
            with self.assertRaisesRegex(RuntimeError, "regular dispatch"):
                self.builder().build(0, common)
