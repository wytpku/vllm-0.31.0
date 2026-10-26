# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run v0.31 MSA contracts from actual source without importing vLLM.

This is source-isolated CPU evidence. Imports, package construction, real
NumPy/Torch operations, native extensions and FlashInfer execution are not
validated. Selected method bodies are unchanged; runner/backend holder classes
omit unrelated bases and methods. Protocol fixtures are identified in JSON.
"""

import argparse
import ast
import copy as copy_module
import enum
import hashlib
import itertools
import json
import math
import platform
import subprocess
import sys
import time
import traceback
import types
import unittest
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, dataclass, field, replace
from datetime import datetime, timezone
from functools import cached_property
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple, NewType, overload
from unittest.mock import Mock, patch


SOURCE_FILES = (
    "vllm/v1/request.py",
    "vllm/v1/core/kv_cache_utils.py",
    "vllm/v1/core/block_pool.py",
    "vllm/v1/core/mwf_eviction.py",
    "vllm/v1/core/single_type_kv_cache_manager.py",
    "vllm/v1/core/kv_cache_coordinator.py",
    "vllm/v1/core/kv_cache_manager.py",
    "vllm/v1/core/sched/output.py",
    "vllm/v1/core/sched/scheduler.py",
    "vllm/v1/worker/gpu/states.py",
    "vllm/v1/worker/gpu/input_batch.py",
    "vllm/v1/worker/gpu/model_runner.py",
    "vllm/v1/worker/gpu_model_runner.py",
    "vllm/v1/worker/gpu/attn_utils.py",
    "vllm/v1/worker/gpu/model_states/default.py",
    "vllm/v1/attention/backend.py",
    "vllm/v1/attention/backends/flashinfer.py",
    "vllm/v1/attention/backends/utils.py",
    "vllm/v1/kv_cache_interface.py",
    "vllm/v1/kv_cache_layout.py",
    "vllm/utils/math_utils.py",
    "tests/v1/core/test_single_type_kv_cache_manager.py",
    "tests/v1/test_multi_segment_metadata.py",
    "tests/v1/worker/test_gpu_input_batch_v2.py",
    "tests/v1/attention/test_flashinfer_msa_metadata.py",
    "tests/msa_layer2/local_cpu_audit.py",
)


def declared_names(node):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, ast.Assign):
        return {
            child.id for target in node.targets for child in ast.walk(target)
            if isinstance(child, ast.Name)
        }
    return set()


class ProtocolTensor:
    """Shape/view/copy fixture; implements no numerical Torch backend."""

    def __init__(self, values, dtype="int64", device="cpu"):
        def shape(value):
            if isinstance(value, (list, tuple)):
                return (len(value),) + shape(value[0]) if value else (0,)
            return ()

        def flatten(value):
            if isinstance(value, (list, tuple)):
                return [item for row in value for item in flatten(row)]
            return [value]

        self.shape, self.storage = shape(values), flatten(values)
        stride, strides = 1, []
        for size in reversed(self.shape):
            strides.append(stride)
            stride *= size
        self.strides = tuple(reversed(strides))
        self.offset = 0
        self.dtype, self.device = dtype, device

    @property
    def ndim(self):
        return len(self.shape)

    def selection(self, key):
        key = key if isinstance(key, tuple) else (key,)
        key += (slice(None),) * (self.ndim - len(key))
        offset, shape, strides = self.offset, [], []
        for item, size, stride in zip(key, self.shape, self.strides):
            if isinstance(item, slice):
                start, stop, step = item.indices(size)
                offset += start * stride
                shape.append(len(range(start, stop, step)))
                strides.append(stride * step)
            else:
                offset += (item if item >= 0 else item + size) * stride
        return offset, tuple(shape), tuple(strides)

    def __getitem__(self, key):
        offset, shape, strides = self.selection(key)
        if not shape:
            return self.storage[offset]
        view = object.__new__(ProtocolTensor)
        view.storage, view.shape, view.strides = self.storage, shape, strides
        view.offset, view.dtype, view.device = offset, self.dtype, self.device
        return view

    def __setitem__(self, key, value):
        offset, shape, _ = self.selection(key)
        if shape:
            for location, item in zip(
                self[key].locations(), ProtocolTensor(value).storage, strict=True
            ):
                self.storage[location] = item
        else:
            self.storage[offset] = value

    def locations(self):
        for indices in itertools.product(*(range(size) for size in self.shape)):
            yield self.offset + sum(
                index * stride for index, stride in zip(indices, self.strides)
            )

    def tolist(self):
        def visit(axis, offset):
            if axis == self.ndim:
                return self.storage[offset]
            return [
                visit(axis + 1, offset + index * self.strides[axis])
                for index in range(self.shape[axis])
            ]
        return visit(0, self.offset)

    def clone(self):
        return ProtocolTensor(self.tolist(), self.dtype, self.device)

    def to(self, *, device=None, dtype=None, copy=False):
        device, dtype = device or self.device, dtype or self.dtype
        if copy or device != self.device or dtype != self.dtype:
            return ProtocolTensor(self.tolist(), dtype, device)
        return self

    def zero_(self):
        for location in self.locations():
            self.storage[location] = 0
        return self

    def data_ptr(self):
        return id(self.storage) + self.offset


def protocol_empty(size, *, dtype="float32", device="cpu"):
    shape = (size,) if isinstance(size, int) else tuple(size)

    def zeros(dimensions):
        if not dimensions:
            return 0
        return [zeros(dimensions[1:]) for _ in range(dimensions[0])]

    return ProtocolTensor(zeros(shape), dtype, device)


def protocol_torch():
    return SimpleNamespace(
        Tensor=ProtocolTensor,
        tensor=lambda values, dtype="int64", device="cpu": ProtocolTensor(
            values, dtype, device
        ),
        empty=protocol_empty,
        empty_like=lambda other: protocol_empty(
            other.shape, dtype=other.dtype, device=other.device
        ),
        device=lambda device: device,
        float16="float16", bfloat16="bfloat16", float32="float32",
        int32="int32", int64="int64", uint8="uint8",
    )


class SourceLoader:
    def __init__(self, repo):
        self.repo = repo
        self.module = types.ModuleType("_msa_v0310_cpu_source_contracts")
        sys.modules[self.module.__name__] = self.module
        self.ns = self.module.__dict__
        self.ns.update(
            ABC=ABC, abstractmethod=abstractmethod, cached_property=cached_property,
            contextmanager=contextmanager, copy=copy_module.copy,
            dataclass=dataclass, field=field, replace=replace,
            FrozenInstanceError=FrozenInstanceError, defaultdict=defaultdict,
            Sequence=Sequence, Integral=Integral, NamedTuple=NamedTuple,
            NewType=NewType, overload=overload, enum=enum, Enum=enum.Enum,
            IntEnum=enum.IntEnum, SimpleNamespace=SimpleNamespace,
            unittest=unittest, Mock=Mock, patch=patch, math=math,
            time=time, itertools=itertools, islice=itertools.islice,
            MEDIUM_GPU="GPU", MEDIUM_CPU="CPU",
            torch=SimpleNamespace(float32="opaque_dtype_fixture"),
        )
        self.loaded = {}

    def load(self, relative, names=(), methods=None):
        """Execute selected source nodes; retain selected method bodies intact."""
        methods = methods or {}
        path = self.repo / relative
        tree = ast.parse(path.read_text(), filename=str(path))
        selected, found = [], set()
        requested = set(names) | set(methods)
        for original in tree.body:
            declared = declared_names(original)
            if isinstance(original, ast.ClassDef) and original.name in methods:
                node = copy_module.deepcopy(original)
                wanted = set(methods[node.name])
                node.body = [
                    item for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name in wanted
                ]
                if {item.name for item in node.body} != wanted:
                    raise ValueError(f"Missing selected methods in {relative}")
                node.bases, node.keywords, node.decorator_list = [], [], []
                selected.append(node)
                found.add(node.name)
            elif declared & set(names):
                selected.append(copy_module.deepcopy(original))
                found.update(declared & set(names))
        if found != requested:
            raise ValueError(
                f"Missing source definitions: {relative}: {requested - found}"
            )
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        unit = ast.fix_missing_locations(
            ast.Module(body=[future, *selected], type_ignores=[])
        )
        exec(compile(unit, str(path), "exec"), self.ns)
        record = self.loaded.setdefault(
            relative, {"definitions": [], "method_holders": {}}
        )
        record["definitions"].extend(sorted(names))
        record["method_holders"].update(
            {key: sorted(value) for key, value in methods.items()}
        )

    def test_classes(self, relative, prefix, excluded=()):
        names = [
            node.name for node in ast.parse((self.repo / relative).read_text()).body
            if isinstance(node, ast.ClassDef) and node.name.startswith(prefix)
            and node.name not in excluded
        ]
        if not names:
            raise ValueError(f"No selected test classes in {relative}")
        self.load(relative, names)
        return [(relative, name) for name in names]


def load_contracts(loader):
    loader.load("vllm/utils/math_utils.py", ("cdiv", "round_down"))
    loader.load("vllm/v1/core/kv_cache_utils.py", (
        "BlockHash", "BlockHashWithGroupId", "make_block_hash_with_group_id",
        "get_block_hash", "get_group_id", "KVCacheBlock", "FreeKVCacheBlockQueue",
        "BlockHashListWithBlockSize", "BlockHashList", "resolve_block_hashes",
    ))
    loader.load("vllm/v1/kv_cache_interface.py", (
        "KVQuantMode", "KVCacheSpec", "AttentionSpec", "FullAttentionSpec",
        "ChunkedLocalAttentionSpec", "CrossAttentionSpec",
    ))
    loader.load("vllm/v1/core/mwf_eviction.py", (
        "MWF_DELTA_SLOPE", "MWF_DELTA_INTERCEPT", "MWFScore",
        "compute_mwf_score", "compute_mwf_score_keys", "_BlockMetadata",
        "MWFEvictionManager",
    ))
    loader.load("vllm/v1/core/block_pool.py", ("BlockHashToBlockMap", "BlockPool"))
    loader.load("vllm/v1/request.py", (
        "MultiSegmentCacheSegment", "_cache_location_int", "RequestStatus",
        "encode_multi_segment_cache_locs", "decode_multi_segment_cache_locs",
        "reset_multi_segment_cache_state",
    ))
    loader.load("vllm/v1/core/single_type_kv_cache_manager.py", (
        "MultiSegHitSnapshot", "MultiSegAllocationPlan", "plan_multiseg_allocation",
        "SingleTypeKVCacheManager", "FullAttentionManager", "CrossAttentionManager",
    ))
    loader.load("vllm/v1/core/kv_cache_coordinator.py", (
        "KVCacheCoordinator", "UnitaryKVCacheCoordinator",
    ))
    loader.load("vllm/v1/core/kv_cache_manager.py", ("KVCacheBlocks", "KVCacheManager"))
    loader.load("vllm/v1/worker/gpu/states.py", (
        "_metadata_int", "normalize_multi_segment_cache_locs",
        "get_uncached_query_ranges", "QueryExecutionPlan",
        "build_query_execution_plan", "build_dense_query_execution_plan",
        "BatchQueryLayout", "build_batch_query_layout", "RequestState",
    ))
    loader.load("vllm/v1/worker/gpu/model_runner.py", ("PreparedQueryInputs",), {
        "GPUModelRunner": ("_check_msa_execution_gate", "prepare_query_inputs"),
    })
    targets = (
        loader.test_classes(
            "tests/v1/core/test_single_type_kv_cache_manager.py", "TestMultiSeg"
        )
        + loader.test_classes(
            "tests/v1/test_multi_segment_metadata.py", "TestMultiSegment",
            excluded=("TestMultiSegmentPadding",),
        )
        + loader.test_classes("tests/v1/worker/test_gpu_input_batch_v2.py", "TestMSA")
    )
    return targets + load_flashinfer_contracts(loader)


def load_flashinfer_contracts(loader):
    loader.ns["torch"] = protocol_torch()
    loader.load("vllm/v1/kv_cache_layout.py", (
        "_DIM_L", "_DIM_B", "_DIM_H", "_DIM_N", "_DIM_C", "KVCacheLayout",
    ))
    loader.load("vllm/v1/attention/backend.py", ("CommonAttentionMetadata",))
    loader.load("vllm/v1/attention/backends/utils.py", (
        "_FLASHINFER_LAYOUT_NAMES", "get_flashinfer_layout_string",
        "split_decodes_and_prefills",
    ))
    loader.ns.update(
        backend=loader.module, BatchPrefillWithPagedKVCacheWrapper=None
    )
    loader.load("vllm/v1/attention/backends/flashinfer.py", (
        "FlashInferMSACall", "FlashInferMetadata",
    ), {
        "FlashInferMetadataBuilder": (
            "kv_cache_layout", "_new_msa_wrapper", "_plan_msa_calls", "build",
        ),
        "FlashInferImpl": ("forward",),
    })
    relative = "tests/v1/attention/test_flashinfer_msa_metadata.py"
    loader.load(relative, ("BLOCK_TABLE", "PlanSpy"))
    return loader.test_classes(relative, "TestFlashInferMSA")


class AuditResult(unittest.TextTestResult):
    def __init__(self, *args):
        super().__init__(*args)
        self.successes = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.successes.append(test.id())


def fingerprint(repo):
    return {
        relative: hashlib.sha256((repo / relative).read_bytes()).hexdigest()
        for relative in SOURCE_FILES
    }


def git_value(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.rstrip("\n")


def run_audit(repo, verbose=False):
    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repo": str(repo),
        "python": {"executable": sys.executable, "version": platform.python_version()},
        "evidence_scope": "source_isolated_cpu_contracts",
        "fixtures": {
            "torch": "Python shape/view/copy protocol; no real Torch operations",
            "flashinfer": "PlanSpy from the current test source; run() always rejects",
            "runner_holder": "unchanged methods; unrelated inheritance omitted",
        },
        "not_run": {
            "normal_package": "vLLM and native extensions are not imported",
            "numpy_padding": "NumPy array operations need the normal CPU environment",
            "real_cpu_tensors": "Torch input packing needs the normal CPU environment",
            "native_mwf_index": "native index is not loaded",
            "gpu_kernels": "no CUDA execution",
            "flashinfer_plan": "no real FlashInfer wrapper execution",
            "attention_run": "layer 3 remains gated",
            "model_end_to_end": "no model loaded",
            "performance": "no benchmark",
        },
        "classes": [],
        "tests_run": 0, "failures": 0, "errors": 0, "skips": 0,
    }
    try:
        status = git_value(repo, "status", "--short")
        report["git"] = {
            "head": git_value(repo, "rev-parse", "HEAD"),
            "branch": git_value(repo, "branch", "--show-current"),
            "dirty": bool(status), "status": status.splitlines(),
        }
        before = fingerprint(repo)
        report["source_sha256"] = before
        report["source_digest"] = hashlib.sha256(
            json.dumps(before, sort_keys=True).encode()
        ).hexdigest()
        for relative in SOURCE_FILES:
            ast.parse((repo / relative).read_text(), filename=str(repo / relative))
        loader = SourceLoader(repo)
        targets = load_contracts(loader)
        for relative, name in targets:
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(loader.ns[name])
            result = unittest.TextTestRunner(
                verbosity=2 if verbose else 1, resultclass=AuditResult
            ).run(suite)
            report["classes"].append({
                "file": relative, "class": name, "tests_run": result.testsRun,
                "evidence_scope": (
                    "source_isolated_builder_plan_spy"
                    if name.startswith("TestFlashInfer")
                    else "source_isolated_cpu_contracts"
                ),
                "successful_methods": len(result.successes),
                "failures": [{"test": test.id(), "traceback": detail}
                             for test, detail in result.failures],
                "errors": [{"test": test.id(), "traceback": detail}
                           for test, detail in result.errors],
                "skips": [{"test": test.id(), "reason": reason}
                          for test, reason in result.skipped],
            })
        report["source_definitions"] = loader.loaded
        report["sources_stable"] = fingerprint(repo) == before
        report["tests_run"] = sum(item["tests_run"] for item in report["classes"])
        report["failures"] = sum(len(item["failures"]) for item in report["classes"])
        report["errors"] = sum(len(item["errors"]) for item in report["classes"])
        report["skips"] = sum(len(item["skips"]) for item in report["classes"])
        report["success"] = (
            report["tests_run"] > 0 and report["failures"] == report["errors"] == 0
            and report["sources_stable"]
        )
    except Exception:
        report["success"] = False
        report["load_error"] = traceback.format_exc()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "--output", type=Path, help="New JSON path; existing files are preserved"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("--output already exists; use a new report path")
    report = run_audit(args.repo.resolve(), args.verbose)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(payload)
        print(f"REPORT_JSON={args.output.resolve()}")
        print(json.dumps({key: report.get(key) for key in
                          ("success", "tests_run", "failures", "errors", "skips")}))
    else:
        print(payload, end="")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
