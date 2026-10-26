# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Normal-package import, GPU input kernels and real FlashInfer plan gates."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
import sysconfig
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SOURCES = (
    "vllm/v1/request.py",
    "vllm/v1/core/block_pool.py",
    "vllm/v1/core/kv_cache_manager.py",
    "vllm/v1/core/kv_cache_coordinator.py",
    "vllm/v1/core/single_type_kv_cache_manager.py",
    "vllm/v1/core/sched/output.py",
    "vllm/v1/core/sched/scheduler.py",
    "vllm/v1/worker/gpu/states.py",
    "vllm/v1/worker/gpu/input_batch.py",
    "vllm/v1/worker/gpu/block_table.py",
    "vllm/v1/worker/gpu/model_runner.py",
    "vllm/v1/worker/gpu_model_runner.py",
    "vllm/v1/worker/gpu/model_states/default.py",
    "vllm/v1/worker/gpu/attn_utils.py",
    "vllm/v1/attention/backend.py",
    "vllm/v1/attention/backends/flashinfer.py",
    "tests/msa_layer2/gpu_metadata_probe.py",
)
MODULES = (
    "vllm", "flashinfer", "vllm._C_stable_libtorch",
    "vllm.v1.attention.backends.flashinfer", "vllm.v1.worker.gpu.input_batch",
    "vllm.v1.worker.gpu.block_table", "vllm.v1.worker.gpu.states",
)
SUPPORTED_PAGE_SIZES = (16, 32, 64)
PAGE_IDS = (9, 3, 7, 1, 8, 4, 6, 2, 5)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_manifest(repo):
    return {name: sha256(repo / name) for name in SOURCES}


def module_record(module):
    name = module.__name__
    try:
        version = importlib.metadata.version(name.split(".")[0])
    except importlib.metadata.PackageNotFoundError:
        version = getattr(module, "__version__", None)
    path = Path(module.__file__).resolve()
    return {"path": str(path), "version": version, "sha256": sha256(path)}


def case_spec(page_size, scenario):
    """Independent expected logical ranges for the synthetic gate matrix."""
    p = page_size
    if scenario in ("A", "B", "C"):
        start = 2 * p
        end = (8 * p + 1) if scenario == "C" else 6 * p + (scenario == "A")
        locs = [3 * p, 5 * p, start]
        ranges = [(2 * p, 3 * p), (5 * p, end)]
        if scenario == "C":
            locs = [3 * p, 5 * p, 6 * p, 7 * p, start]
            ranges = [(2 * p, 3 * p), (5 * p, 6 * p), (7 * p, end)]
        kind = "msa"
    elif scenario.startswith("resume_"):
        start = {"resume_full": 0, "resume_prefix": 272, "resume_decode": 304}[
            scenario
        ]
        end, locs, ranges, kind = 305, [start], [(start, 305)], "resume"
    elif scenario == "slot_reuse":
        start, end, locs, ranges, kind = 0, p + 1, [0], [(0, p + 1)], "slot_reuse"
    else:
        raise ValueError("Unknown metadata scenario: " + scenario)
    positions = [value for begin, stop in ranges for value in range(begin, stop)]
    return {
        "scenario": scenario, "kind": kind, "page_size": p,
        "logical_start": start, "logical_end": end, "logical_advance": end - start,
        "cache_locs": locs, "query_ranges": ranges, "positions": positions,
        "packed_q_count": len(positions), "logits_row": len(positions) - 1,
        "logits_logical_position": end - 1,
    }


def expected_matrix(stage, page_sizes):
    ids = []
    if stage == "kernels":
        for page in page_sizes:
            ids.extend(f"kernels/page{page}/{case}" for case in (
                "A", "B", "C", "resume_full", "resume_prefix", "resume_decode",
                "slot_reuse",
            ))
    elif stage == "plan":
        ids = [
            f"plan/page{page}/{dtype}/{layout}/{case}"
            for page in page_sizes for dtype in ("float16", "bfloat16")
            for layout in ("NHD", "HND") for case in ("A", "B", "C")
        ]
    return {
        "case_ids": ids, "case_count": len(ids),
        "plan_count": 28 * len(page_sizes) if stage == "plan" else 0,
        "dense_progress_count": 3 if stage in ("kernels", "plan") else 0,
        "dense_case_ids": [
            "dense/start32/q65/reject0", "dense/start97/q1/reject0",
            "dense/start97/q4/reject2",
        ] if stage in ("kernels", "plan") else [],
        "page_sizes": list(page_sizes),
    }


def check_gpu(index):
    if not index.isdecimal():
        raise ValueError("--gpu must name one physical NVIDIA GPU index")
    line = subprocess.check_output([
        "nvidia-smi", "--id=" + index,
        "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=15).strip()
    physical, uuid, memory, utilization = (part.strip() for part in line.split(","))
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader",
    ], text=True, timeout=15)
    if int(memory) > 1024 or int(utilization) > 5 or any(
        row.split(",")[0].strip() == uuid for row in processes.splitlines()
    ):
        raise RuntimeError("Selected GPU is busy; this probe does not reserve a GPU")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    return {"index": physical, "uuid": uuid, "memory_mib": int(memory)}


def configure_caches(root):
    names = {
        "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp",
        "XDG_CACHE_HOME": "xdg", "CUDA_CACHE_PATH": "cuda",
        "TRITON_CACHE_DIR": "triton", "TORCHINDUCTOR_CACHE_DIR": "inductor",
        "TORCH_EXTENSIONS_DIR": "torch-extensions",
        "FLASHINFER_WORKSPACE_BASE": "flashinfer", "TVM_FFI_CACHE_DIR": "tvm-ffi",
    }
    for name, folder in names.items():
        path = root / "cache" / folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1")
    sys.dont_write_bytecode = True
    return {name: os.environ[name] for name in names}


def load_package_modules(repo):
    torch = importlib.import_module("torch")
    modules = {name: importlib.import_module(name) for name in MODULES}
    modules["torch"] = torch
    for name, module in modules.items():
        if name.startswith("vllm") and not (
            Path(module.__file__).resolve().is_relative_to(repo)
        ):
            raise RuntimeError("Imported source is outside --repo: " + name)
    return torch, modules


def physical_pages(count):
    if count <= len(PAGE_IDS):
        return list(PAGE_IDS[:count])
    pages = list(range(1, count + 1))
    return list(reversed(pages[::2] + pages[1::2]))


def exercise_inputs(torch, spec, device, tables=None, block_ids=None):
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.worker.gpu.block_table import BlockTables
    from vllm.v1.worker.gpu.input_batch import (
        InputBuffers, post_update, prepare_msa_prefill_inputs,
        prepare_pos_seq_lens, prepare_prefill_inputs,
    )
    from vllm.v1.worker.gpu.states import (
        build_dense_query_execution_plan, build_query_execution_plan,
    )

    p, start, end = spec["page_size"], spec["logical_start"], spec["logical_end"]
    if spec["kind"] == "msa":
        plan = build_query_execution_plan(
            spec["cache_locs"], start, end - start, allow_msa=True
        )
    else:
        plan = build_dense_query_execution_plan(start, end - start)
    count, capacity = spec["packed_q_count"], max(96, spec["packed_q_count"] + 7)
    if plan.logical_positions != tuple(spec["positions"]):
        raise AssertionError("Actual query planner differs from independent fixture")
    buffers = InputBuffers(3, capacity, device)
    tokens = torch.arange(end, dtype=torch.int32, device=device) + 1000
    indices = torch.tensor([2], dtype=torch.int32, device=device)
    boundaries = torch.tensor([0, count], dtype=torch.int32, device=device)
    computed = torch.tensor([11, 22, start], dtype=torch.int32, device=device)
    total = torch.tensor([11, 22, end], dtype=torch.int32, device=device)
    all_tokens = torch.full((3, end + 2), -1, dtype=torch.int32, device=device)
    all_tokens[2, :end].copy_(tokens)
    if plan.msa_enabled:
        logits = prepare_msa_prefill_inputs(buffers, tokens, plan, end)
    else:
        prepare_prefill_inputs(
            buffers.input_ids, torch.zeros((1, 3), dtype=torch.int32, device=device),
            indices, boundaries, all_tokens,
            torch.tensor([0, 0, end], dtype=torch.int32, device=device), computed,
        )
        prepare_pos_seq_lens(indices, boundaries, computed,
                             buffers.positions, buffers.seq_lens)
        logits = boundaries[1:] - 1
    pages = (end + p - 1) // p
    block_ids = list(block_ids) if block_ids is not None else physical_pages(pages)
    if tables is None:
        tables = BlockTables([p], 3, capacity, [max(pages, 32)], device,
                             kernel_block_sizes=[p])
    tables.append_block_ids(2, (block_ids,), overwrite=True)
    tables.apply_staged_writes()
    slots = tables.compute_slot_mappings(
        indices, boundaries, buffers.positions, num_tokens_padded=capacity
    )[0]
    block_table = tables.gather_block_tables(indices, num_reqs_padded=1)[0]
    last = torch.full((3,), -1, dtype=torch.int32, device=device)
    post_update(
        indices, computed, last, None,
        torch.tensor([[77]], dtype=torch.int32, device=device),
        torch.ones(1, dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device), boundaries,
        all_tokens, total,
        logical_seq_lens=buffers.seq_lens[:1] if plan.msa_enabled else None,
    )
    torch.cuda.synchronize(device)
    positions = spec["positions"]
    expected_slots = [block_ids[pos // p] * p + pos % p for pos in positions]
    checks = {
        "logical_positions": buffers.positions[:count].cpu().tolist() == positions,
        "input_token_ids": buffers.input_ids[:count].cpu().tolist()
        == [1000 + pos for pos in positions],
        "slot_mapping": slots[:count].cpu().tolist() == expected_slots,
        "slot_padding": slots[count:].cpu().tolist() == [-1] * (capacity - count),
        "logical_seq_len": buffers.seq_lens[:1].cpu().tolist() == [end],
        "query_start_loc": boundaries.cpu().tolist() == [0, count],
        "logits_row": logits.cpu().tolist() == [count - 1]
        and positions[-1] == end - 1,
        "logical_progress": computed.cpu().tolist() == [11, 22, end],
        "sample_placement": total.cpu().tolist() == [11, 22, end + 1]
        and last.cpu().tolist() == [-1, -1, 77]
        and all_tokens[2, end].item() == 77,
        "other_request_rows": bool((all_tokens[:2] == -1).all().item()),
        "worker_page_table": block_table[0, :pages].cpu().tolist() == block_ids,
    }
    if not all(checks.values()):
        raise AssertionError("GPU metadata input checks failed: " + str(checks))
    history = tokens.cpu().tolist()
    common = CommonAttentionMetadata(
        query_start_loc=boundaries,
        query_start_loc_cpu=torch.tensor([0, count], dtype=torch.int32, device="cpu"),
        seq_lens=buffers.seq_lens[:1], num_reqs=1, num_actual_tokens=count,
        seq_lens_cpu_upper_bound=torch.tensor([end], dtype=torch.int32, device="cpu"),
        max_query_len=count, max_seq_len=end,
        block_table_tensor=block_table, slot_mapping=slots,
        positions=buffers.positions[:count], causal=True,
        query_execution_plans=(plan,),
    )
    record = {
        **spec, "request_state_index": 2, "physical_pages": block_ids,
        "slots": expected_slots, "input_token_ids": [1000 + pos for pos in positions],
        "history_sha256": hashlib.sha256(json.dumps(history).encode()).hexdigest(),
        "computed_after": end, "checks": checks,
        "expected": {"query_start_loc": [0, count], "seq_lens": [end],
                     "computed_after": end, "logits_logical_position": end - 1},
    }
    return common, record, tables


def kernel_cases(torch, args):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    records = []
    for p in args.page_sizes:
        for scenario in ("A", "B", "C"):
            _, record, _ = exercise_inputs(torch, case_spec(p, scenario), device)
            records.append({**record, "case_id": f"kernels/page{p}/{scenario}"})
        resumes = []
        for scenario in ("resume_full", "resume_prefix", "resume_decode"):
            _, record, tables = exercise_inputs(torch, case_spec(p, scenario), device)
            resumes.append(record)
            records.append({**record, "case_id": f"kernels/page{p}/{scenario}"})
        if len({record["history_sha256"] for record in resumes}) != 1:
            raise AssertionError("Resume controls do not share one complete history")
        _, record, _ = exercise_inputs(
            torch, case_spec(p, "slot_reuse"), device, tables=tables,
            block_ids=[47, 19],
        )
        record["checks"]["overwritten_request_slot"] = (
            record["physical_pages"] == [47, 19]
        )
        records.append({**record, "case_id": f"kernels/page{p}/slot_reuse"})
    return records


@contextmanager
def guard_flashinfer_calls(wrapper_type, counter):
    """Count real plan invocations while rejecting attention execution."""
    original_plan = wrapper_type.plan

    @wraps(original_plan)
    def counted_plan(wrapper, *args, **kwargs):
        counter["real_plan_calls"] += 1
        return original_plan(wrapper, *args, **kwargs)

    def forbid_run(*args, **kwargs):
        counter["attention_run_calls"] += 1
        raise AssertionError("Layer-2 gate attempted attention.run")

    with (
        patch.object(wrapper_type, "plan", counted_plan),
        patch.object(wrapper_type, "run", forbid_run),
    ):
        yield


def inspect_plan_buffers(call, query_len, kv_len, page_size, page_ids):
    """Verify retained buffers and return their observed values for the report."""
    expected = {
        "qo_indptr": [0, query_len],
        "kv_indptr": [0, (kv_len + page_size - 1) // page_size],
        "last_page_len": [(kv_len - 1) % page_size + 1],
        "seq_lens": [kv_len],
        "page_ids_cpu": page_ids,
    }
    if len(call.plan_buffers) != len(expected):
        raise AssertionError("FlashInfer plan does not retain all input buffers")
    actual = {}
    for name, tensor in zip(expected, call.plan_buffers, strict=True):
        if str(tensor.device) != "cpu":
            raise AssertionError("FlashInfer retained plan buffer is not on CPU")
        actual[name] = tensor.tolist()
    if actual != expected:
        raise AssertionError("FlashInfer retained plan buffer values mismatch")
    return actual


def plan_cases(torch, backend, args, run_counter):
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheLayout

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    records = []

    with guard_flashinfer_calls(
        backend.BatchPrefillWithPagedKVCacheWrapper, run_counter
    ):
        for p in args.page_sizes:
            for dtype in (torch.float16, torch.bfloat16):
                for layout, layout_name in (
                    (KVCacheLayout.LBNHC, "NHD"), (KVCacheLayout.LBHNC, "HND")
                ):
                    for scenario in ("A", "B", "C"):
                        common, record, _ = exercise_inputs(
                            torch, case_spec(p, scenario), device
                        )
                        builder = object.__new__(backend.FlashInferMetadataBuilder)
                        builder.device = device
                        builder.kv_cache_spec = FullAttentionSpec(
                            block_size=p, num_kv_heads=8, head_size=128, dtype=dtype
                        )
                        builder.cache_config = SimpleNamespace(
                            get_resolved_kv_cache_layout=lambda: layout
                        )
                        builder.model_config = SimpleNamespace(dtype=dtype)
                        builder.page_size = p
                        builder.num_qo_heads, builder.num_kv_heads = 32, 8
                        builder.head_dim = 128
                        builder.use_dcp = builder.has_sinks = False
                        builder.enable_cuda_graph = False
                        builder.window_left = -1
                        builder.global_hyperparameters = SimpleNamespace(
                            has_same_all_params=True
                        )
                        builder.sm_scale, builder.logits_soft_cap = 128**-0.5, None
                        builder.kv_cache_dtype = dtype
                        builder.q_data_type_prefill = builder.q_data_type_decode = dtype
                        builder.prefill_fixed_split_size = -1
                        builder.disable_split_kv = False
                        workspace = torch.empty(args.workspace_mib * 1024**2,
                                                dtype=torch.uint8, device=device)
                        builder._get_workspace_buffer = lambda: workspace
                        metadata = builder.build(0, common)
                        torch.cuda.synchronize(device)
                        snapshots = [call.page_ids.cpu().tolist()
                                     for call in metadata.msa_calls]
                        slot_snapshot = metadata.slot_mapping.cpu().tolist()
                        common.block_table_tensor.zero_()
                        common.slot_mapping.zero_()
                        torch.cuda.synchronize(device)
                        checks = {
                            "page_snapshot": [
                                call.page_ids.cpu().tolist()
                                for call in metadata.msa_calls
                            ] == snapshots,
                            "slot_snapshot": metadata.slot_mapping.cpu().tolist()
                            == slot_snapshot,
                            "independent_wrappers": len({id(call.wrapper)
                                for call in metadata.msa_calls}) == len(snapshots),
                            "independent_workspaces": len({call.workspace.data_ptr()
                                for call in metadata.msa_calls}) == len(snapshots),
                        }
                        calls, offset = [], 0
                        for call, (begin, end), pages in zip(
                            metadata.msa_calls, record["query_ranges"], snapshots,
                            strict=True,
                        ):
                            expected = (offset, offset + end - begin, end)
                            actual = (
                                call.q_slice.start, call.q_slice.stop, call.kv_len
                            )
                            expected_pages = record["physical_pages"][
                                :(end + p - 1) // p
                            ]
                            if actual != expected or pages != expected_pages:
                                raise AssertionError(
                                    "FlashInfer Q/KV/page boundary mismatch"
                                )
                            plan_buffers = inspect_plan_buffers(
                                call, end - begin, end, p, expected_pages
                            )
                            calls.append({"q_slice": list(actual[:2]), "kv_len": end,
                                          "page_ids": pages,
                                          "last_page_len":
                                          plan_buffers["last_page_len"][0],
                                          "plan_buffers": plan_buffers,
                                          "wrapper": type(call.wrapper).__module__ + "."
                                          + type(call.wrapper).__name__})
                            offset += end - begin
                        checks["q_kv_page_contract"] = (
                            offset == record["packed_q_count"]
                        )
                        checks["plan_buffers_contract"] = True
                        try:
                            object.__new__(backend.FlashInferImpl).forward(
                                None, None, None, None, None, metadata, None
                            )
                        except NotImplementedError:
                            checks["forward_guard"] = True
                        else:
                            raise AssertionError("MSA forward guard is absent")
                        if not all(checks.values()):
                            raise AssertionError(
                                "FlashInfer plan checks failed: " + str(checks)
                            )
                        dtype_name = str(dtype).split(".")[-1]
                        record["checks"].update(checks)
                        records.append({
                            **record,
                            "case_id": (
                                f"plan/page{p}/{dtype_name}/{layout_name}/{scenario}"
                            ),
                            "dtype": str(dtype), "layout": layout_name, "calls": calls,
                        })
    if run_counter["attention_run_calls"]:
        raise AssertionError("Attention execution occurred in the metadata gate")
    return records


def dense_progress_controls(torch):
    from vllm.v1.worker.gpu.input_batch import post_update

    device = torch.device("cuda:0")
    records = []
    for start, count, rejected in ((32, 65, 0), (97, 1, 0), (97, 4, 2)):
        indices = torch.tensor([2], dtype=torch.int32, device=device)
        computed = torch.tensor([11, 22, start], dtype=torch.int32, device=device)
        post_update(
            indices, computed, torch.zeros(3, dtype=torch.int32, device=device),
            None, torch.tensor([[77]], dtype=torch.int32, device=device),
            torch.ones(1, dtype=torch.int32, device=device),
            torch.tensor([rejected], dtype=torch.int32, device=device),
            torch.tensor([0, count], dtype=torch.int32, device=device),
            torch.zeros((3, 200), dtype=torch.int32, device=device),
            torch.tensor([11, 22, start + count], dtype=torch.int32, device=device),
        )
        torch.cuda.synchronize(device)
        expected = start + count - rejected
        if computed.cpu().tolist() != [11, 22, expected]:
            raise AssertionError("Dense logical progress changed")
        records.append({
            "case_id": f"dense/start{start}/q{count}/reject{rejected}",
            "start": start, "query_count": count, "rejected": rejected,
            "computed_after": expected, "checks": {"logical_progress": True},
        })
    return records


def validate_matrix(result, expected):
    cases = result.get("cases", [])
    ids = [case["case_id"] for case in cases]
    if sorted(ids) != sorted(expected["case_ids"]) or len(ids) != len(set(ids)):
        raise AssertionError("Incomplete or duplicated metadata case matrix")
    if any(
        not case.get("checks") or not all(case["checks"].values()) for case in cases
    ):
        raise AssertionError("A metadata case lacks passing assertions")
    plans = sum(len(case.get("calls", [])) for case in cases)
    if plans != expected["plan_count"]:
        raise AssertionError("Incorrect number of independent FlashInfer plans")
    real_plans = result.get("real_plan_calls", 0)
    if type(real_plans) is not int or real_plans != plans:
        raise AssertionError("Real FlashInfer plan calls differ from metadata calls")
    if len(result.get("dense_progress_controls", [])) != (
        expected["dense_progress_count"]
    ):
        raise AssertionError("Dense progress controls are missing")
    dense = result.get("dense_progress_controls", [])
    if (
        [item["case_id"] for item in dense] != expected["dense_case_ids"]
        or any(not item.get("checks") or not all(item["checks"].values())
               for item in dense)
    ):
        raise AssertionError("Dense progress controls do not match their contract")
    result.update(case_count=len(cases), plan_count=plans, plan_calls=plans)
    result["coverage"] = {
        "case_ids": ids,
        "dense_case_ids": [item["case_id"]
                           for item in result["dense_progress_controls"]],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("import", "kernels", "plan"), default="plan"
    )
    parser.add_argument("--gpu", help="An idle physical NVIDIA GPU index")
    parser.add_argument(
        "--page-sizes", type=lambda value: tuple(map(int, value.split(","))),
        default=(16,),
    )
    parser.add_argument("--workspace-mib", type=int, default=128)
    args = parser.parse_args(argv)
    args.repo, args.output = args.repo.resolve(), args.output.resolve()
    if args.output.exists() or any(
        root == args.output or root in args.output.parents
        for root in (Path("/tmp"), Path("/private/tmp"))
    ):
        parser.error("Use a new output file in a persistent result directory")
    if (
        args.workspace_mib <= 0 or not args.page_sizes
        or len(set(args.page_sizes)) != len(args.page_sizes)
        or any(p not in SUPPORTED_PAGE_SIZES for p in args.page_sizes)
    ):
        parser.error("Use positive workspace and unique page sizes from 16,32,64")
    if args.stage != "import" and (args.gpu is None or not args.gpu.isdecimal()):
        parser.error("GPU stages require an explicit physical --gpu index")
    return args


def main(argv=None):
    args = parse_args(argv)
    repo, output = args.repo, args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    expected = expected_matrix(args.stage, args.page_sizes)
    result = {
        "success": False, "stage": args.stage,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "normal-package synthetic metadata and GPU component probe",
        "builder_constructor": "synthetic __new__; no model loaded",
        "full_runner_integration": "not_run", "cache_installation": "not_run",
        "normal_package_import": "not_run", "gpu_kernels": "not_run",
        "gpu_metadata_plan": "not_run", "attention_run_calls": 0,
        "model_forward": "not_run", "end_to_end": "not_run", "performance": "not_run",
        "expected": expected, "checks": {}, "page_sizes": list(args.page_sizes),
        "case_count": 0, "plan_count": 0, "plan_calls": 0, "real_plan_calls": 0,
        "coverage": {"case_ids": [], "dense_case_ids": []},
    }
    run_counter = {"attention_run_calls": 0, "real_plan_calls": 0}
    before = None
    try:
        before = source_manifest(repo)
        result["source_sha256"] = before
        result["script_sha256"] = before["tests/msa_layer2/gpu_metadata_probe.py"]
        result["cache_paths"] = configure_caches(output.parent)
        if args.stage == "import":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        else:
            result["gpu_before"] = check_gpu(args.gpu)
        result["head"] = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        result["git_status"] = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--short"], text=True
        )
        result["python"] = {"executable": sys.executable, "version": sys.version,
                            "soabi": sysconfig.get_config_var("SOABI")}
        torch, modules = load_package_modules(repo)
        result["modules"] = {
            name: module_record(module) for name, module in modules.items()
        }
        result["torch_cuda"] = torch.version.cuda
        result["normal_package_import"] = "passed"
        result["checks"]["module_origins"] = True
        if args.stage == "import":
            if torch.cuda.is_initialized():
                raise AssertionError("Import stage unexpectedly initialized CUDA")
            result["checks"]["cuda_not_initialized"] = True
        else:
            result["gpu_runtime"] = {"name": torch.cuda.get_device_name(0),
                                     "capability": torch.cuda.get_device_capability(0)}
            result["page_sizes"] = list(args.page_sizes)
            result["workspace_mib_per_call"] = args.workspace_mib
            if args.stage == "kernels":
                result["cases"] = kernel_cases(torch, args)
            else:
                result["cases"] = plan_cases(
                    torch, modules["vllm.v1.attention.backends.flashinfer"], args,
                    run_counter,
                )
            result["dense_progress_controls"] = dense_progress_controls(torch)
            result["real_plan_calls"] = run_counter["real_plan_calls"]
            validate_matrix(result, expected)
            result["checks"]["complete_case_matrix"] = True
            result["gpu_kernels"] = "passed"
            if args.stage == "plan":
                result["gpu_metadata_plan"] = "passed"
        result["cuda_initialized"] = torch.cuda.is_initialized()
        after = source_manifest(repo)
        result["source_sha256_after"] = after
        result["sources_stable"] = after == before
        if not result["sources_stable"]:
            raise RuntimeError("Source changed while the probe was running")
        result["checks"]["sources_stable"] = True
        result["modules_after"] = {
            name: module_record(module) for name, module in modules.items()
        }
        result["modules_stable"] = result["modules_after"] == result["modules"]
        if not result["modules_stable"]:
            raise RuntimeError("Imported module files changed during the probe")
        result["checks"]["modules_stable"] = True
        if run_counter["attention_run_calls"]:
            raise AssertionError("Attention execution occurred in the metadata gate")
        result["checks"]["attention_run_disabled"] = True
        result["success"] = True
    except Exception:
        result["error"] = traceback.format_exc()
        if before is not None:
            try:
                after = source_manifest(repo)
                result["source_sha256_after"] = after
                result["sources_stable"] = after == before
            except Exception:
                result["sources_stable"] = False
    result["attention_run_calls"] = run_counter["attention_run_calls"]
    result["real_plan_calls"] = run_counter["real_plan_calls"]
    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    with output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"success": result["success"], "output": str(output)}))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
