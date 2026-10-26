# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the V2 model runner's InputBatch (vllm.v1.worker.gpu.input_batch)."""

import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.worker.gpu import cp_utils
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    InputBuffers,
    apply_msa_input_layout,
    prepare_msa_prefill_inputs,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.states import (
    build_batch_query_layout,
    build_query_execution_plan,
)

DEVICE = current_platform.device_type


class TestMSAQueryPreparation(unittest.TestCase):
    def fixture(self):
        runner = GPUModelRunner.__new__(GPUModelRunner)
        runner.req_states = SimpleNamespace(
            req_id_to_index={"request": 0},
            num_computed_tokens_np=[32],
            continuum_suffix_computed_block_locs=[[48, 80, 32]],
        )
        cached = SimpleNamespace(
            req_ids=["request"], num_computed_tokens=[32],
            continuum_suffix_computed_block_locs=[],
        )
        output = SimpleNamespace(
            scheduled_new_reqs=[], scheduled_cached_reqs=cached,
            num_scheduled_tokens={"request": 65},
        )
        batch = SimpleNamespace(
            req_ids=["request"], num_scheduled_tokens=[65], idx_mapping_np=[0],
        )
        return runner, output, batch

    def test_controlled_query_plan_preserves_logical_advance(self):
        runner, output, batch = self.fixture()
        prepared = runner.prepare_query_inputs(output, batch, allow_msa=True)
        plan = prepared.query_execution_plans[0]
        self.assertEqual(plan.logical_advance, 65)
        self.assertEqual(plan.packed_q_count, 33)
        self.assertEqual(prepared.query_layout.query_start_loc, (0, 33))
        self.assertEqual(prepared.query_layout.logical_ends, (97,))
        self.assertEqual(prepared.query_layout.logits_indices, (32,))
        self.assertEqual(output.num_scheduled_tokens, {"request": 65})
        self.assertEqual(runner.req_states.num_computed_tokens_np, [32])

    def test_production_gate_uses_incoming_logical_progress(self):
        runner, output, batch = self.fixture()
        with self.assertRaises(NotImplementedError):
            runner.prepare_query_inputs(output, batch)
        with self.assertRaises(NotImplementedError):
            runner._check_msa_execution_gate(output)
        output.scheduled_cached_reqs.num_computed_tokens[0] = 97
        output.num_scheduled_tokens["request"] = 1
        runner._check_msa_execution_gate(output)
        self.assertEqual(runner.req_states.num_computed_tokens_np, [32])

    def test_mismatched_cpu_batch_is_rejected(self):
        runner, output, batch = self.fixture()
        batch.num_scheduled_tokens[0] = 64
        with self.assertRaises(ValueError):
            runner.prepare_query_inputs(output, batch, allow_msa=True)

    def test_dense_decode_after_cached_window_retains_its_layout(self):
        runner, output, batch = self.fixture()
        runner.req_states.num_computed_tokens_np[0] = 97
        output.num_scheduled_tokens["request"] = 1
        batch.num_scheduled_tokens[0] = 1
        prepared = runner.prepare_query_inputs(output, batch)
        self.assertFalse(prepared.query_execution_plans[0].msa_enabled)
        self.assertEqual(prepared.query_layout.logical_ends, (98,))
        self.assertEqual(prepared.query_layout.logical_positions, (97,))


def test_msa_prefill_cpu_packs_true_logical_positions():
    plan = build_query_execution_plan([48, 80, 32], 32, 65, allow_msa=True)
    buffers = InputBuffers(1, 96, torch.device("cpu"))
    request_tokens = torch.arange(129, dtype=torch.int32) + 1000
    logits = prepare_msa_prefill_inputs(buffers, request_tokens, plan, 97)
    expected_positions = torch.tensor([*range(32, 48), *range(80, 97)])
    assert torch.equal(buffers.positions[:33], expected_positions)
    assert torch.equal(buffers.input_ids[:33], request_tokens[expected_positions])
    assert buffers.seq_lens.tolist() == [97]
    assert logits.tolist() == [32]
    assert buffers.input_ids[logits[0]].item() == 1096


def test_msa_input_batch_preserves_logical_scheduling_and_dense_capacity():
    plan = build_query_execution_plan([48, 80, 32], 32, 65, allow_msa=True)
    layout = build_batch_query_layout((plan,))
    buffers = InputBuffers(1, 65, torch.device("cpu"))
    batch = InputBatch.make_dummy(1, 65, buffers)
    batch.prefill_len_np = np.array([97], dtype=np.int32)
    batch.num_computed_tokens_np = np.array([32], dtype=np.int32)
    apply_msa_input_layout(batch, buffers, torch.arange(97), (plan,), layout)
    assert batch.num_scheduled_tokens.tolist() == [65]
    assert batch.num_tokens == 33
    assert batch.num_tokens_after_padding == 65
    assert batch.query_start_loc_np.tolist() == [0, 33]
    assert batch.query_start_loc.tolist() == [0, 33]
    assert batch.seq_lens.tolist() == [97]
    assert batch.seq_lens_cpu_upper_bound.tolist() == [97]
    assert batch.logits_indices.tolist() == [32]
    assert not batch.is_padding[:33].any()
    assert batch.is_padding[33:].all()


@pytest.mark.parametrize(
    "failure",
    [
        "request_padding", "capacity", "buffer_capacity", "logical_end",
        "logits", "count",
    ],
)
def test_msa_layout_rejection_precedes_buffer_writes(failure):
    plan = build_query_execution_plan([48, 80, 32], 32, 65, allow_msa=True)
    layout = build_batch_query_layout((plan,))
    buffers = InputBuffers(1, 65, torch.device("cpu"))
    batch = InputBatch.make_dummy(1, 65, buffers)
    batch.prefill_len_np = np.array([97], dtype=np.int32)
    if failure == "request_padding":
        batch.num_reqs_after_padding = 2
    elif failure == "capacity":
        batch.num_tokens_after_padding = 32
    elif failure == "buffer_capacity":
        batch.num_tokens_after_padding = 66
    elif failure == "logical_end":
        layout = replace(layout, logical_ends=(96,))
    elif failure == "logits":
        layout = replace(layout, logits_indices=(31,))
    else:
        layout = replace(layout, num_actual_tokens=32)
    before = (
        buffers.input_ids.clone(), buffers.positions.clone(), buffers.seq_lens.clone(),
        buffers.query_start_loc.clone(),
    )
    with pytest.raises((ValueError, NotImplementedError)):
        apply_msa_input_layout(batch, buffers, torch.arange(97), (plan,), layout)
    for actual, expected in zip(
        (
            buffers.input_ids, buffers.positions, buffers.seq_lens,
            buffers.query_start_loc,
        ),
        before,
    ):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("failure", ["unfinished", "short_tokens", "rank", "capacity"])
def test_msa_prefill_rejection_precedes_buffer_writes(failure):
    plan = build_query_execution_plan([48, 80, 32], 32, 65, allow_msa=True)
    buffers = InputBuffers(1, 16 if failure == "capacity" else 96, torch.device("cpu"))
    buffers.input_ids.fill_(77)
    buffers.positions.fill_(-1)
    before = (
        buffers.input_ids.clone(), buffers.positions.clone(), buffers.seq_lens.clone()
    )
    tokens = torch.arange(96 if failure == "short_tokens" else 97)
    if failure == "rank":
        tokens = tokens.unsqueeze(0)
    with pytest.raises((ValueError, NotImplementedError)):
        prepare_msa_prefill_inputs(
            buffers, tokens, plan, 96 if failure == "unfinished" else 97
        )
    for actual, expected in zip(
        (buffers.input_ids, buffers.positions, buffers.seq_lens), before
    ):
        assert torch.equal(actual, expected)


def test_msa_common_metadata_uses_logical_prefix_and_keeps_plans_when_unpadded():
    plan = build_query_execution_plan([48, 80, 32], 32, 65, allow_msa=True)
    metadata = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 33]),
        query_start_loc_cpu=torch.tensor([0, 33]),
        seq_lens=torch.tensor([97]), num_reqs=1, num_actual_tokens=33,
        max_query_len=33, max_seq_len=97,
        block_table_tensor=torch.tensor([[9, 3, 7, 2, 5, 8, 1]]),
        slot_mapping=torch.arange(33),
        positions=torch.tensor(plan.logical_positions),
        query_execution_plans=(plan,),
        continuum_suffix_computed_block_locs=np.array([[48, 80, 32]]),
    )
    assert metadata.compute_num_computed_tokens().tolist() == [32]
    cropped = metadata.unpadded(33, 1)
    assert cropped.query_execution_plans == (plan,)
    assert cropped.positions.tolist() == list(plan.logical_positions)
    assert cropped.continuum_suffix_computed_block_locs.tolist() == [[48, 80, 32]]


@pytest.mark.parametrize(
    "num_reqs,num_tokens",
    [
        (256, 496),  # remainder 240: previously gave the last request 241 tokens
        (128, 512),  # no remainder
        (3, 8),
        (1, 7),
    ],
)
def test_make_dummy_distributes_remainder(num_reqs: int, num_tokens: int):
    """No dummy request may exceed ceil(num_tokens / num_reqs) tokens.

    Dumping the remainder on a single request can produce a dummy request with
    seq_len > max_model_len, which the block tables cannot back; attention
    kernels running on the dummy batch during cudagraph capture then read
    block-table entries out of bounds (https://github.com/vllm-project/vllm/pull/49364
    CI failure).
    """
    buffers = InputBuffers(
        max_num_reqs=num_reqs, max_num_tokens=num_tokens, device=torch.device(DEVICE)
    )
    batch = InputBatch.make_dummy(num_reqs, num_tokens, buffers)

    max_per_req = -(-num_tokens // num_reqs)
    assert batch.num_scheduled_tokens.sum() == num_tokens
    assert batch.num_scheduled_tokens.max() == max_per_req
    assert batch.num_scheduled_tokens.min() >= num_tokens // num_reqs
    # Requests with an extra token are placed at the end of the batch.
    assert (batch.num_scheduled_tokens[:-1] <= batch.num_scheduled_tokens[1:]).all()

    # seq_len == query_len for the dummy prefill-shaped batch, on GPU and CPU.
    query_lens = batch.query_start_loc_np[1:] - batch.query_start_loc_np[:-1]
    assert (query_lens == batch.num_scheduled_tokens).all()
    assert torch.equal(
        batch.seq_lens, torch.from_numpy(batch.num_scheduled_tokens).to(DEVICE)
    )
    assert batch.query_start_loc_np[-1] == num_tokens
    assert torch.equal(
        batch.query_start_loc.cpu(), torch.from_numpy(batch.query_start_loc_np)
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA top-k.")
@pytest.mark.parametrize("is_padding", [True, False])
def test_make_dummy_padding_controls_moe_routing(monkeypatch, is_padding: bool):
    """Dummy tokens marked as padding are dropped by the MoE router (top-k id
    -1), which is wanted for idle DP ranks. Profile runs must not mark them, or
    no token reaches the experts and MoE memory is never profiled."""
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "1")
    num_tokens = 16
    buffers = InputBuffers(
        max_num_reqs=4, max_num_tokens=num_tokens, device=torch.device(DEVICE)
    )
    batch = InputBatch.make_dummy(4, num_tokens, buffers, is_padding=is_padding)
    hidden_states = torch.randn(num_tokens, 4, device=DEVICE)
    router_logits = torch.randn(num_tokens, 8, device=DEVICE)

    with set_forward_context(None, VllmConfig(), is_padding=batch.is_padding):
        _, topk_ids, _ = fused_topk(hidden_states, router_logits, 2, False)

    assert bool((topk_ids == -1).all()) is is_padding
    assert bool((topk_ids >= 0).all()) is not is_padding


def test_prepare_dcp_local_seq_lens_uses_shared_buffer(monkeypatch):
    """The batch must view the caller-owned buffer, sliced to padded length.

    Runtime (and capture) paths all funnel through this helper; attention
    metadata indexes padded rows, so the view must reach
    num_reqs_after_padding, and it must alias the persistent buffer so CUDA
    graph replay sees the recomputed values.
    """
    buffers = InputBuffers(max_num_reqs=4, max_num_tokens=4, device=torch.device("cpu"))
    batch = InputBatch.make_dummy(2, 4, buffers)
    batch.num_reqs_after_padding = 4

    def fake_kernel(
        output,
        seq_lens,
        dcp_size,
        dcp_rank,
        cp_interleave,
        num_reqs,
        max_num_reqs,
        block_size,
    ):
        assert output is buffers.dcp_local_seq_lens
        assert seq_lens is batch.seq_lens
        assert (num_reqs, dcp_size, dcp_rank, cp_interleave) == (2, 4, 1, 16)
        assert (max_num_reqs, block_size) == (4, 128)
        output[:] = torch.tensor([1, 2, 0, 0], dtype=output.dtype)

    class FakeKernel:
        def __getitem__(self, grid):
            assert grid == (1,)
            return fake_kernel

    monkeypatch.setattr(cp_utils, "_dcp_local_seq_lens_kernel", FakeKernel())
    batch.dcp_local_seq_lens = cp_utils.prepare_dcp_local_seq_lens(
        buffers.dcp_local_seq_lens,
        batch.seq_lens,
        batch.num_reqs,
        4,
        1,
        16,
        num_reqs_padded=batch.num_reqs_after_padding,
    )

    assert batch.dcp_local_seq_lens is not None
    assert batch.dcp_local_seq_lens.data_ptr() == buffers.dcp_local_seq_lens.data_ptr()
    assert batch.dcp_local_seq_lens.tolist() == [1, 2, 0, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="triton kernel needs CUDA")
@pytest.mark.parametrize("dcp_size", [2, 4])
@pytest.mark.parametrize("cp_interleave", [1, 16])
def test_prepare_dcp_local_seq_lens_matches_reference(
    dcp_size: int, cp_interleave: int
):
    """Every rank's local lengths must equal the reference torch formula."""
    device = torch.device("cuda:0")
    seq_lens_np = np.array([7, 16, 33, 64, 512, 1023], dtype=np.int32)
    buffers = InputBuffers(max_num_reqs=8, max_num_tokens=32, device=device)
    batch = InputBatch.make_dummy(6, 12, buffers)
    buffers.seq_lens[: len(seq_lens_np)] = torch.from_numpy(seq_lens_np).to(device)

    for dcp_rank in range(dcp_size):
        buffers.dcp_local_seq_lens.fill_(-1)
        batch.dcp_local_seq_lens = cp_utils.prepare_dcp_local_seq_lens(
            buffers.dcp_local_seq_lens,
            batch.seq_lens,
            batch.num_reqs,
            dcp_size,
            dcp_rank,
            cp_interleave,
            num_reqs_padded=batch.num_reqs_after_padding,
        )
        expected = get_dcp_local_seq_lens(
            torch.from_numpy(seq_lens_np), dcp_size, dcp_rank, cp_interleave
        )
        assert batch.dcp_local_seq_lens is not None
        assert torch.equal(batch.dcp_local_seq_lens.cpu(), expected.to(torch.int32))
