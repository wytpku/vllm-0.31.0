# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Normal-package CPU/native metadata gate; never builds or runs CUDA kernels."""

import argparse
import ast
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import json
import math
import os
import re
import signal
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit


MODULES = {
    "vllm": "vllm/__init__.py",
    "vllm.version": "vllm/version.py",
    "vllm._version": "vllm/_version.py",
    "vllm._C_stable_libtorch": None,
    "vllm._mwf_candidate_index": None,
    "vllm.v1.request": "vllm/v1/request.py",
    "vllm.v1.core.block_pool": "vllm/v1/core/block_pool.py",
    "vllm.v1.core.kv_cache_manager": "vllm/v1/core/kv_cache_manager.py",
    "vllm.v1.core.single_type_kv_cache_manager":
        "vllm/v1/core/single_type_kv_cache_manager.py",
    "vllm.v1.core.sched.scheduler": "vllm/v1/core/sched/scheduler.py",
    "vllm.v1.core.sched.output": "vllm/v1/core/sched/output.py",
    "vllm.v1.worker.gpu.states": "vllm/v1/worker/gpu/states.py",
    "vllm.v1.worker.gpu.input_batch": "vllm/v1/worker/gpu/input_batch.py",
    "vllm.v1.worker.gpu.model_runner": "vllm/v1/worker/gpu/model_runner.py",
    "vllm.v1.worker.gpu.model_states.default":
        "vllm/v1/worker/gpu/model_states/default.py",
    "vllm.v1.worker.gpu.attn_utils": "vllm/v1/worker/gpu/attn_utils.py",
    "vllm.v1.attention.backend": "vllm/v1/attention/backend.py",
    "vllm.v1.attention.backends.flashinfer":
        "vllm/v1/attention/backends/flashinfer.py",
}
NATIVE_MODULES = ("vllm._C_stable_libtorch", "vllm._mwf_candidate_index")
NATIVE_TEST_FILES = (
    "tests/v1/core/test_mwf_candidate_index.py",
    "tests/v1/core/test_mwf_eviction.py",
)
WORKER_TESTS = (
    "TestMSAQueryPreparation",
    "test_msa_prefill_cpu_packs_true_logical_positions",
    "test_msa_input_batch_preserves_logical_scheduling_and_dense_capacity",
    "test_msa_layout_rejection_precedes_buffer_writes",
    "test_msa_prefill_rejection_precedes_buffer_writes",
    "test_msa_common_metadata_uses_logical_prefix_and_keeps_plans_when_unpadded",
)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def validate_output(path):
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")
    if any(root == path or root in path.parents
           for root in (Path("/tmp"), Path("/private/tmp"))):
        raise ValueError("Use a persistent result directory outside /tmp")
    return path


def require_origin(path, repo, relative=None, native=False):
    path, repo = Path(path).resolve(), Path(repo).resolve()
    if not path.is_file() or not path.is_relative_to(repo):
        raise ValueError(f"Imported module is outside --repo: {path}")
    if relative is not None and path != (repo / relative).resolve():
        raise ValueError(f"Imported source differs from --repo/{relative}: {path}")
    if native and not str(path).endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES)):
        raise ValueError(f"Required native module is not a compiled extension: {path}")
    return path


def source_files(repo):
    """Share the isolated gate's explicit source set without importing vLLM."""
    path = repo / "tests/msa_layer2/local_cpu_audit.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    sets = [ast.literal_eval(node.value) for node in tree.body
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "SOURCE_FILES"
                for target in node.targets
            )]
    if len(sets) != 1:
        raise ValueError("Cannot locate the CPU gate source manifest")
    names = set(sets[0]) | set(NATIVE_TEST_FILES) | {
        "tests/msa_layer2/package_cpu_gate.py",
        "csrc/mwf_candidate_index.h", "csrc/mwf_candidate_index.cpp",
        "csrc/mwf_candidate_index_bindings.cpp",
    }
    return sorted(names)


def source_manifest(repo):
    return {name: sha256(repo / name) for name in source_files(repo)}


def selected_tests(repo):
    core = "tests/v1/core/test_single_type_kv_cache_manager.py"
    names = [node.name for node in ast.parse((repo / core).read_text()).body
             if isinstance(node, ast.ClassDef) and node.name.startswith("TestMultiSeg")]
    if not names:
        raise ValueError("No MSA cache tests found")
    worker = "tests/v1/worker/test_gpu_input_batch_v2.py"
    available = {node.name for node in ast.parse((repo / worker).read_text()).body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    if not set(WORKER_TESTS).issubset(available):
        raise ValueError("The explicit CPU worker test selection is incomplete")
    return [
        *(f"{core}::{name}" for name in names),
        "tests/v1/test_multi_segment_metadata.py",
        *(f"{worker}::{name}" for name in WORKER_TESTS),
        "tests/v1/attention/test_flashinfer_msa_metadata.py::TestFlashInferMSAMetadata",
        *NATIVE_TEST_FILES,
    ]


def parse_junit(path):
    """Count actual testcase elements, retaining failure/error/skip details."""
    root = ET.parse(path).getroot()
    if root.tag not in ("testsuite", "testsuites"):
        raise ValueError("JUnit root must be testsuite or testsuites")
    cases = list(root.iter("testcase"))
    result = {"tests": len(cases), "failures": 0, "errors": 0, "skipped": 0,
              "failed_cases": [], "skipped_cases": []}
    for case in cases:
        label = case.get("classname", "") + "::" + case.get("name", "")
        for kind in ("failure", "error", "skipped"):
            records = list(case.findall(kind))
            field = {
                "failure": "failures", "error": "errors", "skipped": "skipped"
            }[kind]
            result[field] += len(records)
            for record in records:
                details = "skipped_cases" if kind == "skipped" else "failed_cases"
                result[details].append({
                    "case": label, "kind": kind, "message": record.get("message", ""),
                    "detail": record.text or "",
                })
    # Retain declared counters too: collection failures must never disappear
    # merely because pytest did not emit a normal testcase for them.
    suites = list(root.iter("testsuite"))
    for key in ("failures", "errors", "skipped"):
        counters = [int(suite.get(key, "0")) for suite in suites]
        if any(value < 0 for value in counters):
            raise ValueError("JUnit counters must be nonnegative")
        leaves = [suite for suite in suites if not list(suite.findall("testsuite"))]
        result[key] = max(result[key], sum(int(s.get(key, "0")) for s in leaves),
                          max(counters, default=0))
    return result


def pytest_passed(exit_code, result):
    return exit_code == 0 and result["tests"] > 0 and all(
        result[name] == 0 for name in ("failures", "errors", "skipped")
    )


def matches_pytest_node(nodeid, required):
    return (
        nodeid == required or nodeid.startswith(required + "::")
        or nodeid.startswith(required + "[")
    )


class RequiredPytestCoverage:
    """Reject deselection and verify every selected test reached its call phase."""

    def __init__(self, repo, required):
        self.repo = Path(repo).resolve()
        self.required = tuple(required)
        self.collected = []
        self.executed = []
        self.deselected = []
        self._canonical = {}

    def nodeid(self, item):
        path = Path(item.path).resolve().relative_to(self.repo).as_posix()
        suffix = item.nodeid.split("::", 1)
        return path + ("::" + suffix[1] if len(suffix) == 2 else "")

    def pytest_deselected(self, items):
        self.deselected.extend(self.nodeid(item) for item in items)

    def pytest_collection_finish(self, session):
        for item in session.items:
            canonical = self.nodeid(item)
            self.collected.append(canonical)
            self._canonical[item.nodeid] = canonical

    def pytest_runtest_logreport(self, report):
        # pytest-subtests emits extra call reports for unittest subTest scopes;
        # only the enclosing item's report marks execution of its node ID.
        if report.when == "call" and getattr(report, "context", None) is None:
            self.executed.append(self._canonical.get(report.nodeid, report.nodeid))

    def result(self):
        collected, executed = set(self.collected), set(self.executed)
        missing = [required for required in self.required if not any(
            matches_pytest_node(nodeid, required) for nodeid in collected
        )]
        unexpected = sorted(nodeid for nodeid in collected if not any(
            matches_pytest_node(nodeid, required) for required in self.required
        ))
        result = {
            "required_nodes": list(self.required),
            "collected_nodeids": self.collected,
            "executed_nodeids": self.executed,
            "deselected_nodeids": self.deselected,
            "missing_required_nodes": missing,
            "unexpected_collected_nodeids": unexpected,
            "unexecuted_nodeids": sorted(collected - executed),
            "unexpected_executed_nodeids": sorted(executed - collected),
            "duplicate_nodeids": (
                len(collected) != len(self.collected)
                or len(executed) != len(self.executed)
            ),
        }
        result["passed"] = bool(self.required and collected) and not any((
            missing, unexpected, self.deselected, result["unexecuted_nodeids"],
            result["unexpected_executed_nodeids"], result["duplicate_nodeids"],
        ))
        return result


def module_record(module, repo, relative=None, native=False):
    path = require_origin(module.__file__, repo, relative, native)
    return {"path": str(path), "sha256": sha256(path),
            "version": getattr(module, "__version__", None), "native": native}


def normalize_precompiled_version(version):
    if not isinstance(version, str):
        raise ValueError("Package version must be a string")
    normalized = version
    if normalized.endswith("+precompiled"):
        normalized = normalized[:-len("+precompiled")]
    elif "+" in normalized and normalized.endswith(".precompiled"):
        normalized = normalized[:-len(".precompiled")]
    pattern = (
        r"(?P<release>\d+(?:\.\d+)+)(?:(?:a|b|rc)\d+)?"
        r"(?:\.post\d+)?(?:\.dev\d+)?(?:\+[a-zA-Z0-9]+(?:[._-][a-zA-Z0-9]+)*)?"
    )
    match = re.fullmatch(pattern, normalized)
    if match is None or [int(n) for n in match["release"].split(".")[:2]] != [0, 31]:
        raise ValueError(f"Required a real vLLM 0.31-series version, got {version!r}")
    return normalized


def validate_binding(distribution_version, runtime_version, direct_url, repo):
    distribution = normalize_precompiled_version(distribution_version)
    runtime = normalize_precompiled_version(runtime_version)
    if distribution != runtime:
        raise ValueError("Installed distribution/runtime versions do not match")
    if not isinstance(direct_url, dict) or direct_url.get("dir_info", {}).get(
        "editable"
    ) is not True:
        raise ValueError("vLLM must have a verified editable direct_url.json")
    parsed = urlsplit(direct_url.get("url", ""))
    editable = Path(unquote(parsed.path)).resolve()
    if (
        parsed.scheme != "file" or parsed.netloc not in ("", "localhost")
        or editable != Path(repo).resolve()
    ):
        raise ValueError("Editable vLLM points outside --repo")
    return {"environment_binding_passed": True,
            "distribution_version": distribution_version,
            "runtime_version": runtime_version,
            "normalized_distribution_version": distribution,
            "normalized_runtime_version": runtime,
            "editable_source": str(editable), "editable": True,
            "direct_url_scheme": parsed.scheme}


def probe_imports(repo):
    torch = importlib.import_module("torch")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU worker started with CUDA already initialized")
    modules = {}
    for name, relative in MODULES.items():
        module = importlib.import_module(name)
        modules[name] = module_record(module, repo, relative, name in NATIVE_MODULES)
    for name in ("torch", "flashinfer", "numpy", "pytest"):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        modules[name] = {"path": str(path), "sha256": sha256(path),
                         "version": getattr(module, "__version__", None)}
    distribution = importlib.metadata.distribution("vllm")
    direct = distribution.read_text("direct_url.json")
    version = modules["vllm"]["version"]
    binding = validate_binding(
        distribution.version, version, json.loads(direct) if direct else None, repo
    )
    if torch.cuda.is_initialized():
        raise RuntimeError("Normal CPU imports initialized CUDA")
    return {
        "modules": modules, "cuda_initialized": False,
        "torch_cuda_build": torch.version.cuda,
        **binding,
        "version_note": "Versions are recorded as installed; no SCM/version rewriting.",
    }


def audit_pool(pool):
    """Check canonical identities and bounded free links, allowing hash aliases."""
    queue = pool.free_block_queue
    previous = queue.fake_free_list_head
    current = previous.next_free_block
    free_ids = set()
    for _ in range(len(pool.blocks) + 1):
        if current is queue.fake_free_list_tail:
            break
        if (
            current is None or current.is_null
            or not 0 <= current.block_id < len(pool.blocks)
            or pool.blocks[current.block_id] is not current or current.pool is not pool
            or current.block_id in free_ids or current.ref_cnt != 0
            or current.prev_free_block is not previous
        ):
            raise AssertionError("Invalid or cyclic free queue")
        free_ids.add(current.block_id)
        previous, current = current, current.next_free_block
    else:
        raise AssertionError("Free queue traversal exceeded pool size")
    if (
        current.prev_free_block is not previous
        or len(free_ids) != queue.num_free_blocks
    ):
        raise AssertionError("Free queue links/count disagree")
    for index, block in enumerate(pool.blocks):
        if block.block_id != index or block.pool is not pool:
            raise AssertionError("Pool contains a noncanonical block")
        if not block.is_null and ((block.ref_cnt == 0) != (index in free_ids)):
            raise AssertionError("Block references disagree with free membership")
        if not block.is_null and block.ref_cnt < 0:
            raise AssertionError("Negative block reference count")
        if block.block_hash is not None and not pool.cached_block_hash_to_block.contain(
            block.block_hash, index
        ):
            raise AssertionError("Primary hash is absent from the cache map")
    alias_count = 0
    for block_id, aliases in pool.cached_block_hashes_by_block.items():
        block = pool.blocks[block_id]
        for key in aliases:
            alias_count += 1
            if block.block_hash is None or not pool.cached_block_hash_to_block.contain(
                key, block_id
            ):
                raise AssertionError("Reverse alias is absent from the cache map")
    for key, value in pool.cached_block_hash_to_block._cache.items():
        entries = (
            value.items() if isinstance(value, dict) else ((value.block_id, value),)
        )
        for block_id, block in entries:
            if block_id != block.block_id:
                raise AssertionError("Cache bucket ID differs from its block")
            if pool.blocks[block.block_id] is not block or block.is_null:
                raise AssertionError("Cache map contains a noncanonical/null block")
            aliases = pool.cached_block_hashes_by_block.get(block.block_id, ())
            if key != block.block_hash and key not in aliases:
                raise AssertionError(
                    "Cache key is neither primary nor a registered alias"
                )
    policy = pool.mwf_eviction_manager
    if policy is not None and (
        policy._index is None or policy._index.size() != len(free_ids)
    ):
        raise AssertionError("Native MWF candidates disagree with the free queue")
    return {"free_blocks": len(free_ids),
            "cache_keys": len(pool.cached_block_hash_to_block),
            "aliases": alias_count}


def native_smoke():
    """Use the installed native binding and real pool; no mocks or build."""
    from vllm._mwf_candidate_index import CandidateKind, MWFCandidateIndex
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id

    index = MWFCandidateIndex(alpha=1.0, beta=2.0)
    index.add(7, CandidateKind.UNHASHED)
    if index.size() != 1 or index.update(7, CandidateKind.TRACKED) is not True:
        raise AssertionError("Native add/update contract failed")
    if index.pop_n(1, 0.0) != [7]:
        raise AssertionError("Native pop contract failed")
    index.add(8, CandidateKind.MISSING)
    if index.erase(8) is not True or index.size() != 0:
        raise AssertionError("Native erase contract failed")
    pool = BlockPool(4, enable_caching=True, hash_block_size=16)
    pool.enable_mwf_eviction(minexp_a=0.0, minexp_b=-19.15349,
                             minexp_alpha=1915.332, minexp_beta=47.63942)
    blocks = pool.get_new_blocks(3)
    for number, block in enumerate(blocks, 1):
        key = make_block_hash_with_group_id(BlockHash(bytes([number])), 0)
        pool._insert_block_hash(key, block, 16 * number)
    alias = make_block_hash_with_group_id(BlockHash(b"alias"), 0)
    pool._insert_block_hash(alias, blocks[0], 16)
    pool.free_blocks(reversed(blocks))
    policy = pool.mwf_eviction_manager

    def state():
        audit_pool(pool)
        return (
            tuple((b.ref_cnt, b.block_hash, b.block_hash_num_tokens)
                  for b in pool.blocks),
            tuple(b.block_id for b in pool.free_block_queue.get_all_free_blocks()),
            {key: (value.last_accessed, value.prefix_tokens)
             for key, value in policy._metadata.items()},
            {key: frozenset(value) for key, value
             in pool.cached_block_hashes_by_block.items()},
        )

    before = state()

    class InjectedPublicationFailure(Exception):
        pass

    try:
        with pool.multiseg_transaction():
            pool.touch([blocks[0]])
            pool.get_new_blocks(1)
            raise InjectedPublicationFailure
    except InjectedPublicationFailure:
        pass
    if state() != before or policy._index.size() != pool.get_num_free_blocks():
        raise AssertionError("Real MWF metadata/native index rollback failed")
    reused = pool.get_new_blocks(3)
    if {block.block_id for block in reused} != {block.block_id for block in blocks}:
        raise AssertionError("Restored native candidate membership is incorrect")
    pool.free_blocks(reused)
    if pool.get_num_free_blocks() != 3 or policy._index.size() != 3:
        raise AssertionError("Real pool/native release contract failed")
    return {"binding_api": "passed", "real_block_pool": "passed",
            "multiseg_rollback_native_resync": "passed", "native_mocked": False,
            "alias_aware_pool_audit": "passed", "final_pool": audit_pool(pool)}


def worker(repo, output, mode):
    result = {"success": False, "worker": mode, "started_at_utc": timestamp()}
    try:
        result.update(probe_imports(repo))
        result["normal_package_imports_passed"] = True
        if mode == "inspect":
            result["native_smoke"] = native_smoke()
            result["native_smoke_passed"] = True
        else:
            pytest = importlib.import_module("pytest")
            junit = output.parent / "pytest.xml"
            nodes = selected_tests(repo)
            result["pytest_nodes"] = nodes
            coverage = RequiredPytestCoverage(repo, nodes)
            status = int(pytest.main([
                "-q", "-p", "no:cacheprovider", "--junitxml=" + str(junit),
                *(str(repo / node.split("::", 1)[0])
                  + ("::" + node.split("::", 1)[1] if "::" in node else "")
                  for node in nodes),
            ], plugins=[coverage]))
            counts = parse_junit(junit)
            result["pytest"] = {**counts, "exit_code": status,
                                "junit_path": str(junit),
                                "selection_coverage": coverage.result()}
            if (
                not pytest_passed(status, counts)
                or not result["pytest"]["selection_coverage"]["passed"]
            ):
                raise RuntimeError(
                    "Normal package pytest failed, skipped or omitted required tests"
                )
        torch = importlib.import_module("torch")
        result["cuda_initialized"] = torch.cuda.is_initialized()
        if result["cuda_initialized"]:
            raise RuntimeError("CPU package/native stage initialized CUDA")
        result["modules_after"] = probe_imports(repo)["modules"]
        if result["modules_after"] != result["modules"]:
            raise RuntimeError("Imported module/binary identity changed during worker")
        result["success"] = True
    except BaseException:
        result["error"] = traceback.format_exc()
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                result["cuda_initialized"] = torch.cuda.is_initialized()
            except Exception:
                result["cuda_state_error"] = traceback.format_exc()
    result["finished_at_utc"] = timestamp()
    write_json_new(output, result)
    return 0 if result["success"] else 1


def child_environment(artifacts):
    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        env.pop(name, None)
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               VLLM_ENABLE_V1_MULTIPROCESSING="0", VLLM_USE_V2_MODEL_RUNNER="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1")
    for name, folder in {
        "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp", "XDG_CACHE_HOME": "xdg",
        "CUDA_CACHE_PATH": "cuda", "TRITON_CACHE_DIR": "triton",
        "TORCHINDUCTOR_CACHE_DIR": "inductor", "TORCH_EXTENSIONS_DIR": "torch",
        "FLASHINFER_WORKSPACE_BASE": "flashinfer", "TVM_FFI_CACHE_DIR": "tvm-ffi",
    }.items():
        path = artifacts / "cache" / folder
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    return env


def stop_child_group(process):
    """Clean our worker session on timeout, cancellation and normal completion."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    # A terminated group leader can leave descendants behind. Kill the owned
    # group even when wait() already returned for the leader.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_child(command, log_path, cwd, env, timeout):
    with log_path.open("x") as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        finally:
            stop_child_group(process)


def handle_termination(signum, _frame):
    # Let the active worker's finally block finish even if another TERM arrives.
    signal.signal(signum, signal.SIG_IGN)
    raise InterruptedError(f"Package gate interrupted by signal {signum}")


def run_gate(repo, output, timeout, executor=run_child):
    output = validate_output(output)
    artifacts = output.parent / (output.stem + "-artifacts")
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir()
    report = {
        "stage": "package", "success": False, "started_at_utc": timestamp(),
        "scope": "normal-package CPU tensors and actual installed native lifecycle",
        "normal_package_imports_passed": False, "native_smoke_passed": False,
        "environment_binding_passed": False,
        "sources_stable": False, "source_sha256": {}, "modules": {},
        "cuda_initialized": False, "gpu": "not_run", "attention_run_calls": 0,
        "model_forward": "not_run", "performance": "not_run", "workers": {},
        "artifacts": str(artifacts),
        "pytest": {"tests": 0, "failures": 0, "errors": 0, "skipped": 0,
                   "exit_code": None, "junit_path": str(artifacts / "pytest.xml"),
                   "log_path": str(artifacts / "pytest.log"), "status": "not_run"},
    }
    before = None
    try:
        before = source_manifest(repo)
        report["source_sha256"] = before
        report["head"] = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        report["git_describe"] = subprocess.check_output(
            ["git", "-C", str(repo), "describe", "--tags", "--long",
             "--always", "--dirty"],
            text=True,
        ).strip()
        report["git_status"] = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--short"], text=True
        )
        report["python"] = {"executable": sys.executable, "version": sys.version}
        env = child_environment(artifacts)
        for mode in ("inspect", "pytest"):
            worker_json, log = artifacts / (mode + ".json"), artifacts / (mode + ".log")
            command = [sys.executable, "-B", str(Path(__file__).resolve()),
                       "--repo", str(repo), "--output", str(worker_json),
                       "--worker", mode]
            report["workers"][mode] = {
                "status": "running", "command": command,
                "json_path": str(worker_json), "log_path": str(log),
            }
            print("PACKAGE_PHASE=" + mode, flush=True)
            code = executor(command, log, artifacts, env, timeout)
            child = json.loads(worker_json.read_text()) if worker_json.is_file() else {}
            report["workers"][mode].update(
                exit_code=code, result=child,
                status="passed" if code == 0 and child.get("success") else "failed",
            )
            if child.get("cuda_initialized") is True:
                report["cuda_initialized"] = True
            if mode == "inspect":
                report["normal_package_imports_passed"] = (
                    child.get("normal_package_imports_passed") is True
                )
                report["environment_binding_passed"] = (
                    child.get("environment_binding_passed") is True
                )
                if "modules" in child:
                    report["modules"] = child["modules"]
            if mode == "pytest" and "pytest" in child:
                report["pytest"] = {**child["pytest"], "log_path": str(log),
                                    "status": "failed"}
            if code != 0 or child.get("success") is not True:
                raise RuntimeError(f"Package {mode} failed; see {log}")
            if child.get("cuda_initialized") is not False:
                raise RuntimeError(
                    "CPU worker did not confirm CUDA remained uninitialized"
                )
            if mode == "inspect":
                report.update({key: child[key] for key in (
                    "modules", "runtime_version", "distribution_version",
                    "editable_source", "normalized_distribution_version",
                    "normalized_runtime_version", "environment_binding_passed",
                )})
                if not report["environment_binding_passed"]:
                    raise RuntimeError(
                        "Installed vLLM environment binding did not pass"
                    )
                if not report["normal_package_imports_passed"]:
                    raise RuntimeError("Normal package imports did not pass")
                report["native_smoke_passed"] = child.get("native_smoke_passed") is True
                if not report["native_smoke_passed"]:
                    raise RuntimeError("Actual native smoke did not pass")
            else:
                if child.get("modules") != report["modules"]:
                    raise RuntimeError(
                        "Package/native import identity differs across workers"
                    )
                if (
                    not pytest_passed(report["pytest"]["exit_code"], report["pytest"])
                    or child["pytest"].get("selection_coverage", {}).get("passed")
                    is not True
                ):
                    raise RuntimeError(
                        "Package pytest contains failure, skip or missing tests"
                    )
                report["pytest"]["status"] = "passed"
        report["sources_stable"] = source_manifest(repo) == before
        if not report["sources_stable"]:
            raise RuntimeError("Source changed during normal package validation")
        report["success"] = True
    except BaseException:
        report["error"] = traceback.format_exc()
    finally:
        if before is not None:
            try:
                report["sources_stable"] = source_manifest(repo) == before
                if not report["sources_stable"]:
                    report["success"] = False
                    report["source_drift"] = "Source identity changed during the stage"
            except Exception:
                report["sources_stable"] = report["success"] = False
                report["source_check_error"] = traceback.format_exc()
        report["finished_at_utc"] = timestamp()
        write_json_new(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument(
        "--worker", choices=("inspect", "pytest"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    repo = args.repo.resolve()
    try:
        output = validate_output(args.output)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    if args.worker:
        return worker(repo, output, args.worker)
    previous_term_handler = signal.signal(signal.SIGTERM, handle_termination)
    try:
        report = run_gate(repo, output, args.timeout)
    finally:
        signal.signal(signal.SIGTERM, previous_term_handler)
    print("REPORT_JSON=" + str(output), flush=True)
    print(json.dumps({key: report[key] for key in (
        "stage", "success", "normal_package_imports_passed", "native_smoke_passed"
    )}), flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
