# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""User-operated MWF gates; inspect first, then CPU and GPU in explicit stages."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

GPU_STAGES = ("gpu-basic", "gpu-shared_cancel", "gpu-preemption")
SOURCE_FILES = (
    "CMakeLists.txt", "setup.py", "pyproject.toml",
    "requirements/common.txt", "requirements/cuda.txt",
    "csrc/mwf_candidate_index.h", "csrc/mwf_candidate_index.cpp",
    "csrc/mwf_candidate_index_bindings.cpp", "vllm/config/cache.py",
    "vllm/engine/arg_utils.py", "vllm/v1/core/mwf_eviction.py",
    "vllm/v1/core/block_pool.py", "vllm/v1/core/kv_cache_utils.py",
    "vllm/v1/core/kv_cache_manager.py", "vllm/v1/core/kv_cache_coordinator.py",
    "vllm/v1/core/single_type_kv_cache_manager.py",
    "vllm/v1/core/sched/scheduler.py", "vllm/v1/engine/llm_engine.py",
    "vllm/v1/engine/core_client.py", "vllm/v1/sample/sampler.py",
    "vllm/v1/attention/backends/flashinfer.py",
    "tests/v1/core/test_mwf_candidate_index.cpp",
    "tests/v1/core/test_mwf_candidate_index.py",
    "tests/v1/core/test_mwf_eviction.py", "tests/v1/core/test_prefix_caching.py",
    "tests/engine/test_arg_utils.py",
    "tests/config/test_config_utils.py",
    "tests/v1/core/run_mwf_native_check.py", "tests/v1/core/run_mwf_gate.py",
    "tests/v1/core/mwf_gate_env.py", "tests/v1/core/mwf_gate_gpu.py",
    "tests/v1/core/test_mwf_gate.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def validate_result_root(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("Result root must be an absolute existing directory")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError("Result root does not exist") from error
    if not resolved.is_dir() or any(
        resolved.is_relative_to(root) for root in (Path("/tmp"), Path("/private/tmp"))
    ):
        raise ValueError("Use a persistent directory under wyt/msa, outside /tmp")
    return resolved


def require_origin(path: Path, repo: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
        repo = repo.resolve(strict=True)
        package = (repo / "vllm").resolve(strict=True)
    except OSError as error:
        raise ValueError("Module origin does not exist") from error
    if (
        not package.is_relative_to(repo) or not resolved.is_file()
        or not resolved.is_relative_to(package)
    ):
        raise ValueError(f"Module is outside the requested vLLM source: {resolved}")
    return resolved


def run_stages(stage_names: list[str], execute: Callable[[str], dict]) -> dict:
    stages = {name: {"status": "not_run"} for name in stage_names}
    for name in stage_names:
        try:
            stages[name] = execute(name)
        except Exception as error:
            stages[name] = {
                "status": "failed", "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        if stages[name].get("status") != "passed":
            break
    passed = bool(stage_names) and all(
        value.get("status") == "passed" for value in stages.values()
    )
    gpu = [stages.get(name, {}).get("status") for name in GPU_STAGES]
    gpu_status = (
        "passed" if all(state == "passed" for state in gpu)
        else "incomplete" if any(name in stages for name in GPU_STAGES)
        else "not_run"
    )
    return {
        "status": "passed" if passed else "failed", "scope": stage_names,
        "stages": stages, "gpu_gate": gpu_status,
        "full_build_gate": "not_run", "performance_gate": "not_run",
    }


def source_manifest(repo: Path) -> dict:
    files = {relative: sha256(repo / relative) for relative in SOURCE_FILES}
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"files": files, "source_digest": hashlib.sha256(canonical).hexdigest()}


def create_run_directory(root: Path, name: str | None) -> Path:
    if name is None:
        return Path(tempfile.mkdtemp(prefix="mwf-v0310-gate-", dir=root))
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Run name must be a single directory name")
    run = root / name
    run.mkdir()
    return run


def child_environment(run: Path) -> dict[str, str]:
    env = os.environ.copy()
    directories = {
        "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp",
        "XDG_CACHE_HOME": "cache", "TORCH_EXTENSIONS_DIR": "torch-extensions",
        "TRITON_CACHE_DIR": "triton", "VLLM_CACHE_ROOT": "vllm-cache",
        "FLASHINFER_WORKSPACE_BASE": "flashinfer", "CUDA_CACHE_PATH": "cuda-cache",
    }
    for key, folder in directories.items():
        path = run / folder
        path.mkdir(exist_ok=True)
        env[key] = str(path)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    return env


def logged(command: list[str], log: Path, env: dict, cwd: Path, timeout: int) -> int:
    print(f"RUN {command!r}\nLOG {log}", flush=True)
    with log.open("w") as stream:
        stream.write(json.dumps(command) + "\n")
        stream.flush()
        result = subprocess.run(
            command, cwd=cwd, env=env, stdout=stream,
            stderr=subprocess.STDOUT, timeout=timeout, check=False,
        )
    print(f"EXIT {result.returncode}: {log.name}", flush=True)
    return result.returncode


def check_idle_gpu(device: str, output: Path, env: dict) -> None:
    observations = {}
    queries = (("gpu", "gpu=uuid,memory.used"), ("processes", "compute-apps=pid"))
    for key, field in queries:
        command = ["nvidia-smi", "-i", device, f"--query-{field}",
                   "--format=csv,noheader,nounits"]
        result = subprocess.run(command, env=env, text=True, capture_output=True,
                                timeout=30, check=False)
        observations[key] = {"command": command, "returncode": result.returncode,
                             "stdout": result.stdout, "stderr": result.stderr}
    write_json(output / "gpu-before.json", observations)
    gpu, processes = observations["gpu"], observations["processes"]
    rows = gpu["stdout"].strip().splitlines()
    if gpu["returncode"] or processes["returncode"] or len(rows) != 1:
        raise ValueError("Cannot verify one selected GPU; inspect gpu-before.json")
    if processes["stdout"].strip() or float(rows[0].split(",")[1].strip()) > 256:
        raise ValueError("Selected GPU is occupied; choose an idle device")


def verify_cpu_evidence(evidence: Path, manifest: dict, environment: dict) -> None:
    previous = json.loads((evidence / "manifest.json").read_text())
    summary = json.loads((evidence / "summary.json").read_text())
    prior_env = json.loads((evidence / "environment.json").read_text())
    if previous["source_digest"] != manifest["source_digest"]:
        raise ValueError("CPU evidence belongs to different source files")
    if summary["status"] != "passed" or any(
        summary["stages"].get(name, {}).get("status") != "passed"
        for name in ("native", "package")
    ):
        raise ValueError("GPU requires passing native and normal-package CPU gates")
    if prior_env["python"] != environment["python"]:
        raise ValueError("The interpreter changed since the CPU gate")
    if prior_env["modules"] != environment["modules"]:
        raise ValueError("Imported modules changed since the CPU gate")
    if prior_env["packages"] != environment["packages"] or (
        prior_env["loaded_vllm_binaries"] != environment["loaded_vllm_binaries"]
    ):
        raise ValueError("Packages or loaded binaries changed since CPU tests")
    for field in ("version", "cuda_build_version", "cxx11_abi"):
        if prior_env["torch"].get(field) != environment["torch"].get(field):
            raise ValueError("Torch runtime/ABI changed since the CPU gate")


def compare_suite(folder: Path, atol: float, rtol: float) -> dict:
    import numpy as np

    records = {
        policy: json.loads((folder / policy / "records.json").read_text())
        for policy in ("lru", "mwf")
    }
    names = [case["name"] for case in records["lru"]]
    if not names or names != [case["name"] for case in records["mwf"]]:
        raise ValueError("Policies must produce identical nonempty case sets")
    pairs = [("lru", name, "mwf", name, "cross_policy") for name in names]
    for policy in ("lru", "mwf"):
        coverage = json.loads((folder / policy / "coverage.json").read_text())
        if not coverage.get("passed"):
            raise ValueError(f"Missing mechanism coverage for {policy}")
        pairs.extend(
            (policy, ref, policy, case, "within_policy")
            for ref, case in coverage.get("reference_pairs", [])
        )
    results = []
    for ref_policy, ref_name, policy, name, category in pairs:
        with np.load(folder / ref_policy / f"{ref_name}.npz") as reference, np.load(
            folder / policy / f"{name}.npz"
        ) as candidate:
            if not np.array_equal(
                reference["prompt_token_ids"], candidate["prompt_token_ids"]
            ):
                raise ValueError("Numerical comparisons require the same prompt")
            ids = candidate["token_ids"].tolist()
            ref_ids = reference["token_ids"].tolist()
            partial = name == "cancel_partial"
            if partial:
                ref_ids = ref_ids[:len(ids)]
            a = reference["logits"][:len(ids)] if partial else reference["logits"]
            b = candidate["logits"]
            if (
                not ids or a.shape != b.shape or a.ndim != 2
                or a.shape[0] != len(ids) or len(ref_ids) != len(ids)
                or a.shape[1] == 0 or not np.isfinite(a).all()
                or not np.isfinite(b).all()
            ):
                raise ValueError("Invalid, incomplete or non-finite logits")
            different = next(
                (i for i, (x, y) in enumerate(zip(ref_ids, ids)) if x != y), None
            )
            aligned = len(ids) if different is None else different + 1
            a, b = a[:aligned].astype(np.float64), b[:aligned].astype(np.float64)
            error = np.abs(a - b)
            bad = error > atol + rtol * np.abs(a)
            results.append({
                "reference": f"{ref_policy}/{ref_name}",
                "candidate": f"{policy}/{name}", "category": category,
                "passed": different is None and not bool(bad.any()),
                "token_ids_equal": different is None,
                "first_different_token": different, "aligned_steps": aligned,
                "max_abs": float(error.max()), "mean_abs": float(error.mean()),
                "rmse": float(np.sqrt(np.mean(error ** 2))),
                "bad_elements": int(bad.sum()), "elements_compared": int(error.size),
            })
    report = {"atol": atol, "rtol": rtol, "comparisons": results}
    report["passed"] = all(item["passed"] for item in results)
    write_json(folder / "comparison.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--run-name", help="Fixed result directory name; must not exist")
    parser.add_argument(
        "--stage", choices=("inspect", "native", "package", "cpu", "gpu", "all"),
        default="inspect",
    )
    parser.add_argument("--expected-source-digest")
    parser.add_argument("--cxx", help="Explicit compiler for the native component gate")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--gpu", help="Explicit CUDA device index or UUID")
    parser.add_argument("--cpu-evidence", type=Path)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument(
        "--suite", choices=("basic", "shared_cancel", "preemption", "all"),
        default="basic",
    )
    parser.add_argument("--timeout", type=int, default=2700)
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--rtol", type=float, default=0.001)
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    result_root = validate_result_root(args.result_root)
    if args.timeout <= 0 or not (0 <= args.atol < float("inf")) or not (
        0 <= args.rtol < float("inf")
    ):
        parser.error("Timeout must be positive; tolerances must be finite/nonnegative")
    run = create_run_directory(result_root, args.run_name)
    print(f"RESULT_DIR={run}", flush=True)
    manifest = source_manifest(repo)
    manifest.update(repo=str(repo), python=sys.executable, stage=args.stage)
    for name, command in (
        ("head", ["git", "-C", str(repo), "rev-parse", "HEAD"]),
        ("git_status", ["git", "-C", str(repo), "status", "--porcelain=v1"]),
    ):
        manifest[name] = subprocess.check_output(command, text=True).strip()
    write_json(run / "manifest.json", manifest)
    print(f"SOURCE_DIGEST={manifest['source_digest']}", flush=True)
    if args.expected_source_digest and (
        manifest["source_digest"] != args.expected_source_digest
    ):
        write_json(run / "summary.json", {
            "status": "failed", "error": "Source mismatch",
        })
        raise ValueError("Source digest mismatch; synchronize the gate sources")
    for relative in SOURCE_FILES:
        target = run / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / relative, target)
    env = child_environment(run)
    base = repo / "tests/v1/core"
    selected_gpu = list(GPU_STAGES) if args.suite == "all" else [f"gpu-{args.suite}"]
    stage_names = {
        "inspect": ["inspect"], "native": ["native"],
        "package": ["inspect", "package"],
        "cpu": ["inspect", "native", "package"],
        "gpu": ["inspect", *selected_gpu],
        "all": ["inspect", "native", "package", *selected_gpu],
    }[args.stage]

    def execute(stage: str) -> dict:
        start = time.monotonic()
        if source_manifest(repo)["source_digest"] != manifest["source_digest"]:
            raise RuntimeError("Sources changed while the gate was running")
        output = run / stage
        output.mkdir()
        command = [sys.executable, "-B"]
        stage_env = env.copy()
        if stage == "inspect":
            command += [str(base / "mwf_gate_env.py"), "--repo", str(repo),
                        "--output", str(run / "environment.json")]
        elif stage == "native":
            if not args.cxx:
                raise ValueError("Native gate requires an explicit --cxx compiler")
            command += [str(base / "run_mwf_native_check.py"), "--snapshot-from",
                        str(repo), "--output-dir", str(output), "--cxx", args.cxx]
        elif stage == "package":
            command += ["-m", "pytest", "-q",
                        str(base / "test_mwf_candidate_index.py"),
                        str(base / "test_mwf_eviction.py"),
                        str(base / "test_prefix_caching.py"),
                        str(repo / "tests/engine/test_arg_utils.py"),
                        str(repo / "tests/config/test_config_utils.py"),
                        str(base / "test_mwf_gate.py"), "-k", "mwf",
                        f"--junitxml={output / 'tests.xml'}",
                        f"--basetemp={output / 'pytest-tmp'}",
                        "-o", f"cache_dir={output / 'pytest-cache'}"]
        else:
            environment = json.loads((run / "environment.json").read_text())
            if args.stage == "gpu":
                if args.cpu_evidence is None:
                    raise ValueError("GPU requires --cpu-evidence from --stage cpu")
                verify_cpu_evidence(args.cpu_evidence, manifest, environment)
            if args.model is None or args.gpu is None:
                raise ValueError("GPU requires an explicit --model and --gpu")
            if not (args.model / "config.json").is_file():
                raise ValueError("Use an existing local model snapshot")
            stage_env.update(
                CUDA_VISIBLE_DEVICES=args.gpu, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                VLLM_ENABLE_V1_MULTIPROCESSING="0", VLLM_USE_V2_MODEL_RUNNER="0",
                HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1", PYTHONHASHSEED="0",
            )
            for policy in ("lru", "mwf"):
                folder = output / policy
                folder.mkdir()
                check_idle_gpu(args.gpu, folder, env)
                worker = command + [
                    str(base / "mwf_gate_gpu.py"), "--repo", str(repo),
                    "--model", str(args.model.resolve()), "--output", str(folder),
                    "--policy", policy, "--suite", stage.removeprefix("gpu-"),
                    "--dtype", args.dtype, "--attention-backend", "FLASHINFER",
                ]
                code = logged(worker, output / f"{policy}.log", stage_env,
                              run, args.timeout)
                if code:
                    return {"status": "failed", "policy": policy, "exit_code": code}
            compared = compare_suite(output, args.atol, args.rtol)
            return {"status": "passed" if compared["passed"] else "failed",
                    "comparisons": str(output / "comparison.json")}
        code = logged(command, output / "run.log", stage_env, run, args.timeout)
        if stage == "package" and code == 0:
            nodes = list(ET.parse(output / "tests.xml").iter("testcase"))
            if not nodes or any(node.find("skipped") is not None for node in nodes):
                raise ValueError("Package gate requires nonempty tests with no skips")
            required = (
                "test_mwf_candidate_index", "test_mwf_eviction", "test_prefix_caching",
                "test_arg_utils", "test_config_utils", "test_mwf_gate",
            )
            if any(not any(stem in node.get("classname", "") for node in nodes)
                   for stem in required):
                raise ValueError("A required package test file was not exercised")
        return {"status": "passed" if code == 0 else "blocked" if stage == "inspect"
                else "failed", "exit_code": code, "log": str(output / "run.log"),
                "seconds": time.monotonic() - start}

    summary = run_stages(stage_names, execute)
    if source_manifest(repo)["source_digest"] != manifest["source_digest"]:
        summary.update(status="failed", error="Sources changed during this run")
    write_json(run / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"RESULT_DIR={run}", flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
