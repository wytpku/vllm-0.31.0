# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the v0.31 MSA metadata gate with source-bound, separate evidence stages."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import tarfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


STAGES = ("inspect", "cpu", "package", "import", "kernels", "plan")
IMPLEMENTATION_FILES = (
    "vllm/v1/request.py", "vllm/v1/core/block_pool.py",
    "vllm/v1/core/kv_cache_utils.py", "vllm/v1/core/mwf_eviction.py",
    "vllm/v1/core/single_type_kv_cache_manager.py",
    "vllm/v1/core/kv_cache_manager.py", "vllm/v1/core/kv_cache_coordinator.py",
    "vllm/v1/core/sched/output.py", "vllm/v1/core/sched/scheduler.py",
    "vllm/v1/worker/gpu/states.py", "vllm/v1/worker/gpu/input_batch.py",
    "vllm/v1/worker/gpu/block_table.py", "vllm/v1/worker/gpu/model_runner.py",
    "vllm/v1/worker/gpu_model_runner.py", "vllm/v1/worker/gpu/attn_utils.py",
    "vllm/v1/worker/gpu/model_states/default.py", "vllm/v1/attention/backend.py",
    "vllm/v1/attention/backends/flashinfer.py", "vllm/v1/kv_cache_interface.py",
    "vllm/v1/attention/backends/utils.py",
    "vllm/v1/kv_cache_layout.py", "vllm/utils/math_utils.py",
    "csrc/mwf_candidate_index.cpp", "csrc/mwf_candidate_index.h",
    "csrc/mwf_candidate_index_bindings.cpp",
)
TEST_FILES = (
    "tests/v1/core/test_single_type_kv_cache_manager.py",
    "tests/v1/core/test_mwf_candidate_index.py", "tests/v1/core/test_mwf_eviction.py",
    "tests/v1/test_multi_segment_metadata.py",
    "tests/v1/worker/test_gpu_input_batch_v2.py",
    "tests/v1/attention/test_flashinfer_msa_metadata.py",
)
GATE_FILES = tuple("tests/msa_layer2/" + name for name in (
    "local_cpu_audit.py", "package_cpu_gate.py", "gpu_metadata_probe.py",
    "run_metadata_gate.py", "run_gpu_checks.sh", "test_metadata_gate.py",
    "test_package_cpu_gate.py", "test_gpu_metadata_probe.py",
))
SOURCE_FILES = IMPLEMENTATION_FILES + TEST_FILES + GATE_FILES
REQUIRED_CPU_CLASSES = {
    "TestMultiSegAllocationPlan", "TestMultiSegHitRefresh",
    "TestMultiSegPlanningAdapter", "TestMultiSegInstallation",
    "TestMultiSegOracleGateRegressions", "TestMultiSegmentMetadata",
    "TestMultiSegmentRequestState", "TestMSAQueryPreparation",
    "TestFlashInferMSAMetadata",
}


def selected_stages(stage):
    if stage == "all":
        return STAGES
    if stage not in STAGES:
        raise ValueError("Unknown metadata gate stage")
    return STAGES[:STAGES.index(stage) + 1]


def validate_output(repo, output):
    repo, candidate = Path(repo).resolve(), Path(output)
    if not candidate.is_absolute():
        raise ValueError("Use an absolute result directory")
    if candidate.exists() or candidate.is_symlink():
        raise ValueError("Result directory already exists; preserve prior evidence")
    output = candidate.resolve()
    if output == repo or repo in output.parents:
        raise ValueError("Results must be outside the source checkout")
    for root in (Path("/tmp").resolve(), Path("/private/tmp").resolve()):
        if output == root or root in output.parents:
            raise ValueError("Use a persistent result directory, not /tmp")
    for suffix in (".tar.gz", ".tar.gz.partial"):
        archive = output.with_name(output.name + suffix)
        if archive.exists() or archive.is_symlink():
            raise ValueError("Evidence archive already exists")
    return output


def source_manifest(repo):
    repo = Path(repo).resolve()
    manifest = {}
    for relative in SOURCE_FILES:
        path = (repo / relative).resolve()
        if not path.is_relative_to(repo) or not path.is_file():
            raise ValueError("Missing or foreign source: " + relative)
        manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest


def manifest_digest(manifest):
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def collect_git_state(repo):
    def git(*arguments):
        return subprocess.check_output(
            ["git", "-C", str(repo), *arguments], text=True
        ).strip()
    return {
        "head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "status": git("status", "--short"),
        "describe": git("describe", "--always", "--dirty", "--tags"),
    }


def _method(repo, relative, owner, name):
    tree = ast.parse((repo / relative).read_text(), filename=relative)
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == owner)
    return next(node for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def inspect_production_gate(repo):
    init = _method(
        repo, "vllm/v1/core/kv_cache_manager.py", "KVCacheManager", "__init__"
    )
    defaults = dict(zip(
        (arg.arg for arg in init.args.args[-len(init.args.defaults):]),
        init.args.defaults,
    ))
    cache_off = isinstance(defaults.get("enable_multiseg_cache"), ast.Constant)
    cache_off = cache_off and defaults["enable_multiseg_cache"].value is False
    v2 = _method(
        repo, "vllm/v1/worker/gpu/model_runner.py", "GPUModelRunner", "execute_model"
    )
    first_if = v2.body[0]
    v2_guard = (
        isinstance(first_if, ast.If)
        and isinstance(first_if.body[0], ast.Expr)
        and isinstance(first_if.body[0].value, ast.Call)
        and isinstance(first_if.body[0].value.func, ast.Attribute)
        and first_if.body[0].value.func.attr == "_check_msa_execution_gate"
    )
    legacy = _method(
        repo, "vllm/v1/worker/gpu_model_runner.py", "GPUModelRunner", "execute_model"
    )
    legacy_guard = any(
        isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Name)
        and node.exc.func.id == "NotImplementedError"
        for node in ast.walk(ast.Module(body=legacy.body[:4], type_ignores=[]))
    )
    forward = _method(
        repo, "vllm/v1/attention/backends/flashinfer.py", "FlashInferImpl", "forward"
    )
    forward_guard = any(
        isinstance(node, ast.Attribute) and node.attr == "msa_calls"
        for node in ast.walk(ast.Module(body=forward.body[:3], type_ignores=[]))
    ) and any(
        isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Name)
        and node.exc.func.id == "NotImplementedError"
        for node in ast.walk(ast.Module(body=forward.body[:3], type_ignores=[]))
    )
    checks = dict(
        cache_default_off=cache_off, v2_guard_before_updates=v2_guard,
        legacy_runner_guard=legacy_guard, flashinfer_forward_guard=forward_guard,
    )
    if not all(checks.values()):
        raise ValueError(
            "Production MSA guard is open or unrecognized: " + repr(checks)
        )
    return checks


def validate_stage(name, report, manifest, page_sizes):
    if report.get("success") is not True:
        raise ValueError("Child stage did not pass")
    hashes = report.get("source_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Stage lacks source provenance")
    anchors = {"vllm/v1/request.py", "vllm/v1/core/kv_cache_manager.py",
               "vllm/v1/attention/backends/flashinfer.py"}
    if not anchors <= hashes.keys():
        raise ValueError("Stage source provenance lacks implementation anchors")
    for path, value in hashes.items():
        if path in manifest and manifest[path] != value:
            raise ValueError("Stage source mismatch: " + path)
    if report.get("sources_stable") is not True:
        raise ValueError("Stage source changed during execution")
    if name == "cpu":
        if report.get("tests_run", 0) <= 0 or any(
            report.get(key) != 0 for key in ("failures", "errors", "skips")
        ):
            raise ValueError("CPU contracts failed, skipped, or were not executed")
        classes = {item.get("class") for item in report.get("classes", [])
                   if item.get("tests_run", 0) > 0}
        if not REQUIRED_CPU_CLASSES <= classes:
            raise ValueError("Missing CPU contract classes")
    elif name == "package":
        if (report.get("stage") != "package"
                or report.get("normal_package_imports_passed") is not True
                or report.get("environment_binding_passed") is not True
                or report.get("native_smoke_passed") is not True
                or report.get("cuda_initialized") is not False):
            raise ValueError("Normal package/native CPU evidence is incomplete")
        pytest = report.get("pytest", {})
        if pytest.get("tests", 0) <= 0 or any(
            pytest.get(key) != 0
            for key in ("failures", "errors", "skipped", "exit_code")
        ) or pytest.get("selection_coverage", {}).get("passed") is not True:
            raise ValueError("Normal-package pytest did not fully pass")
    else:
        if report.get("stage") != name or not report.get("modules"):
            raise ValueError("Probe stage/import provenance mismatch")
        if report.get("attention_run_calls") != 0:
            raise ValueError("Layer-2 gate must not execute attention")
        if name != "plan" and report.get("real_plan_calls") != 0:
            raise ValueError("This probe stage must not plan attention")
        if report.get("modules_stable") is not True:
            raise ValueError("Probe import/binary provenance changed")
        if name == "import":
            if report.get("cuda_initialized") is not False:
                raise ValueError("Import-only stage initialized CUDA")
        else:
            if report.get("case_count", 0) <= 0:
                raise ValueError("GPU probe did not execute required cases")
            expected = report.get("expected", {})
            actual = report.get("coverage", {})
            probe_path = Path(__file__).with_name("gpu_metadata_probe.py")
            spec = importlib.util.spec_from_file_location(
                "_msa_gate_probe_schema", probe_path
            )
            probe = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(probe)
            required = probe.expected_matrix(name, page_sizes)
            if expected != required:
                raise ValueError("GPU expected matrix does not match this gate")
            if (not expected or not actual
                    or actual.get("case_ids") != expected.get("case_ids")
                    or len(actual.get("case_ids", [])) != report["case_count"]):
                raise ValueError("GPU case coverage is incomplete")
            cases = report.get("cases", [])
            if ([item.get("case_id") for item in cases] != required["case_ids"]
                    or len(cases) != report["case_count"] or any(
                not item.get("checks") or any(value is not True
                    for value in item["checks"].values()) for item in cases
            )):
                raise ValueError("GPU case checks are missing or failed")
            dense = report.get("dense_progress_controls", [])
            if (len(dense) != required["dense_progress_count"]
                    or [item.get("case_id") for item in dense]
                    != required["dense_case_ids"]
                    or actual.get("dense_case_ids") != required["dense_case_ids"]
                    or any(not item.get("checks") or any(
                        value is not True for value in item["checks"].values()
                    ) for item in dense)):
                raise ValueError("Dense progress control coverage is incomplete")
            if sorted(report.get("page_sizes", [])) != sorted(page_sizes):
                raise ValueError("GPU page-size matrix differs from the request")
            if name == "plan":
                calls = [item.get("calls", []) for item in cases]
                if (report.get("plan_count") != 28 * len(page_sizes)
                        or report["case_count"] != 12 * len(page_sizes)
                        or report.get("real_plan_calls") != required["plan_count"]
                        or sum(map(len, calls)) != required["plan_count"]
                        or any(len(item.get("calls", [])) != (
                            3 if item["case_id"].endswith("/C") else 2
                        ) for item in cases)):
                    raise ValueError(
                        "FlashInfer plan/configuration coverage is incomplete"
                    )


def run_child(command, log, cwd, env, timeout):
    def signal_group(sig):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass

    def stop_child():
        signal_group(signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            signal_group(signal.SIGKILL)
        signal_group(signal.SIGKILL)
        process.wait()

    started = time.monotonic()
    with Path(log).open("x") as stream:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timed_out = False
        try:
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                stop_child()
                code = process.returncode
            residual = False
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                pass
            else:
                residual = True
        finally:
            stop_child()
    return dict(exit_code=code, timed_out=timed_out, residual_process_group=residual,
                seconds_not_benchmark=time.monotonic() - started)


def handle_termination(signum, _frame):
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, signal.SIG_IGN)
    raise InterruptedError(f"Metadata gate interrupted by signal {signum}")


def child_environment(output, stage):
    env = os.environ.copy()
    cache = output / "cache" / stage
    for name, folder in {
        "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp",
        "XDG_CACHE_HOME": "xdg", "CUDA_CACHE_PATH": "cuda",
        "TRITON_CACHE_DIR": "triton", "TORCHINDUCTOR_CACHE_DIR": "inductor",
        "TORCH_EXTENSIONS_DIR": "torch-extensions", "TVM_FFI_CACHE_DIR": "tvm-ffi",
        "FLASHINFER_WORKSPACE_BASE": "flashinfer",
    }.items():
        path = cache / folder
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    env.update(PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1", PYTHONHASHSEED="0")
    return env


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def archive_run(output):
    def evidence_files(folder):
        for path in sorted(folder.iterdir()):
            if folder == output and path.name == "cache":
                continue
            if path.is_symlink():
                relative = path.relative_to(output)
                raise ValueError("Unexpected evidence symlink: " + str(relative))
            if path.is_dir():
                yield from evidence_files(path)
            elif path.is_file():
                yield path

    target = output.with_name(output.name + ".tar.gz")
    partial = target.with_name(target.name + ".partial")
    with (
        partial.open("xb") as stream,
        tarfile.open(fileobj=stream, mode="w:gz") as archive,
    ):
        for path in evidence_files(output):
            relative = path.relative_to(output)
            archive.add(
                path, arcname=str(Path(output.name) / relative), recursive=False
            )
    os.link(partial, target)
    partial.unlink()
    return target


def execute_gate(args, executor=run_child):
    repo = Path(args.repo).resolve()
    output = validate_output(repo, args.output)
    stages = selected_stages(args.stage)
    if any(name in stages for name in ("kernels", "plan")) and not args.gpu:
        raise ValueError("GPU stages require an explicit idle physical --gpu")
    output.mkdir(parents=True)
    report = {
        "scope": "v0.31 MSA metadata component gate",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {"executable": sys.executable, "version": sys.version},
        "requested_stage": args.stage, "requested_stages": list(stages),
        "stages": {name: {"status": "not_run"} for name in STAGES},
        "success": False, "metadata_component_gate_passed": False,
        "gpu_stages_started": [],
        "attention_run_calls": 0, "attention_numerics": "not_run",
        "full_runner_integration": "not_run",
        "model_forward": "not_run", "end_to_end": "not_run", "performance": "not_run",
    }
    current = "inspect"
    try:
        manifest = source_manifest(repo)
        report["source_sha256"] = manifest
        report["source_digest"] = manifest_digest(manifest)
        report["git"] = collect_git_state(repo)
        for relative, expected in manifest.items():
            payload = (repo / relative).read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError("Source changed during input snapshot")
            target = output / "inputs" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        checks = inspect_production_gate(repo)
        write_json(output / "inspect.json", dict(
            success=True, production_gates=checks, source_sha256=manifest,
            sources_stable=True, git=report["git"],
        ))
        report["stages"]["inspect"] = {"status": "passed", "production_gates": checks}
        scripts = repo / "tests" / "msa_layer2"
        for current in stages[1:]:
            if source_manifest(repo) != manifest:
                raise ValueError("Source changed between gate stages")
            result_path = output / (current + ".json")
            if current == "cpu":
                command = [sys.executable, "-B", str(scripts / "local_cpu_audit.py")]
            elif current == "package":
                command = [sys.executable, "-B", str(scripts / "package_cpu_gate.py"),
                           "--timeout", str(args.timeout)]
            else:
                command = [sys.executable, "-B", str(scripts / "gpu_metadata_probe.py"),
                           "--stage", current, "--page-sizes",
                           ",".join(map(str, args.page_sizes)), "--workspace-mib",
                           str(args.workspace_mib)]
                if current in ("kernels", "plan"):
                    command.extend(("--gpu", args.gpu))
            command.extend(("--repo", str(repo), "--output", str(result_path)))
            if current in ("kernels", "plan"):
                report["gpu_stages_started"].append(current)
                report["attention_run_calls"] = None
            result = executor(
                command, output / (current + ".log"), repo,
                child_environment(output, current),
                2 * args.timeout + 30 if current == "package" else args.timeout,
            )
            report["stages"][current] = {"status": "failed", "process": result}
            child = None
            if result_path.is_file():
                child = json.loads(result_path.read_text())
                if current in ("kernels", "plan"):
                    report["attention_run_calls"] = child.get("attention_run_calls")
            if (result.get("exit_code") != 0 or result.get("timed_out")
                    or result.get("residual_process_group")):
                raise ValueError("Stage process failed: " + current)
            if child is None:
                raise ValueError("Stage did not write its JSON: " + current)
            validate_stage(current, child, manifest, args.page_sizes)
            if source_manifest(repo) != manifest:
                raise ValueError("Source changed during stage: " + current)
            report["stages"][current]["status"] = "passed"
            report["stages"][current]["result"] = result_path.name
        report["sources_stable"] = source_manifest(repo) == manifest
        report["success"] = report["sources_stable"]
        report["metadata_component_gate_passed"] = (
            report["success"] and all(
                item["status"] == "passed" for item in report["stages"].values()
            )
        )
    except BaseException:
        report["failure_stage"] = current
        report["error"] = traceback.format_exc()
        report["stages"][current]["status"] = "failed"
    report["archive_requested"] = getattr(args, "archive", True)
    report["delivery_success"] = True
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "summary.json", report)
    if report["archive_requested"]:
        try:
            archive_run(output)
        except Exception:
            report["delivery_success"] = False
            report["delivery_error"] = traceback.format_exc()
            report["success"] = False
            report["metadata_component_gate_passed"] = False
            failure_summary = output / "summary.archive-failure.json"
            write_json(failure_summary, report)
            failure_summary.replace(output / "summary.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New persistent directory"
    )
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="cpu")
    parser.add_argument(
        "--gpu", help="One physical NVIDIA GPU index, required for GPU stages"
    )
    parser.add_argument("--page-sizes", default="16,32,64")
    parser.add_argument("--workspace-mib", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--no-archive", action="store_false", dest="archive")
    args = parser.parse_args()
    try:
        args.page_sizes = tuple(int(value) for value in args.page_sizes.split(","))
        if (not args.page_sizes or len(set(args.page_sizes)) != len(args.page_sizes)
                or any(value not in (16, 32, 64) for value in args.page_sizes)
                or args.workspace_mib <= 0 or not math.isfinite(args.timeout)
                or args.timeout <= 0):
            raise ValueError("Use distinct 16/32/64 pages and positive limits")
        if any(
            stage in selected_stages(args.stage) for stage in ("kernels", "plan")
        ):
            if args.gpu is None or not args.gpu.isdecimal():
                raise ValueError("GPU stages require one explicit physical --gpu index")
        validate_output(args.repo, args.output)
    except ValueError as exc:
        parser.error(str(exc))
    previous = {
        number: signal.signal(number, handle_termination)
        for number in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        report = execute_gate(args)
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
    print("RESULT_DIR=" + str(args.output.resolve()))
    print(
        "METADATA_COMPONENT_GATE_PASSED="
        + str(report["metadata_component_gate_passed"])
    )
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
