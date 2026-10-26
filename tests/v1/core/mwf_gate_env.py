# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe natural package imports and provenance before the MWF package gate."""

import argparse
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

from run_mwf_gate import require_origin, validate_result_root


def collect_environment(repo: Path) -> dict:
    """Collect import evidence without installing or building any dependency."""
    repo = repo.resolve()
    report = {
        "repo": str(repo), "probe_cwd": os.getcwd(),
        "ready": False, "errors": [], "failedModules": {},
        "python": {
            "executable": sys.executable,
            "real_executable": os.path.realpath(sys.executable),
            "version": sys.version, "prefix": sys.prefix,
            "base_prefix": sys.base_prefix, "soabi": sysconfig.get_config_var("SOABI"),
        },
        "packages": {}, "modules": {}, "loaded_vllm_binaries": {},
        "expected": {}, "torch": {}, "flashinfer": {}, "commands": {},
        "environment": {name: os.environ.get(name) for name in (
            "PYTHONPATH", "LD_LIBRARY_PATH", "CUDA_HOME", "VLLM_TARGET_DEVICE",
            "VLLM_USE_PRECOMPILED", "CUDA_VISIBLE_DEVICES",
        )},
        "notes": [
            "Full CMake build and source/binary correspondence remain unverified.",
            "Torch ABI and loaded-binary hashes are import evidence only.",
            "CUDA unavailability does not fail the CPU gate.",
        ],
    }
    errors = report["errors"]

    def file_record(path):
        real = Path(path).resolve()
        return {"realfile": str(real),
                "sha256": hashlib.sha256(real.read_bytes()).hexdigest()}

    def source_module(name, relative=None):
        try:
            module = importlib.import_module(name)
            record = file_record(module.__file__)
            report["modules"][name] = record
            real = Path(record["realfile"])
            if name == "vllm" or name.startswith("vllm."):
                require_origin(real, repo)
            if relative is not None and real != (repo / relative).resolve():
                errors.append(f"Unexpected source file for {name}: {real}")
            return module
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            report["failedModules"][name] = failure
            errors.append(f"Import {name}: {failure}")
            return None

    requirements = repo / "requirements/cuda.txt"
    pins = {}
    try:
        for name, version in re.findall(
            r"(?m)^\s*(torch|flashinfer-python|flashinfer-cubin)\s*==\s*([^\s#]+)",
            requirements.read_text(),
        ):
            pins[name] = version
        report["expected"] = {**file_record(requirements), "pins": pins}
        for name in ("torch", "flashinfer-python", "flashinfer-cubin"):
            if name not in pins:
                errors.append(f"Missing explicit source requirement: {name}")
    except Exception as exc:
        errors.append(f"Source requirements: {type(exc).__name__}: {exc}")

    for name in ("vllm", "torch", "flashinfer-python", "flashinfer-cubin",
                 "transformers", "huggingface_hub", "pytest", "tblib"):
        try:
            distribution = importlib.metadata.distribution(name)
            entry = {"version": distribution.version,
                     "root": str(Path(distribution.locate_file("")).resolve())}
            direct = distribution.read_text("direct_url.json")
            if direct:
                data = json.loads(direct)
                parsed = urlsplit(data.get("url", ""))
                host = parsed.netloc.rsplit("@", 1)[-1]
                entry["direct_url"] = {
                    "url": urlunsplit((parsed.scheme, host, parsed.path, "", "")),
                    "editable": bool(data.get("dir_info", {}).get("editable", False)),
                }
                if name == "vllm" and entry["direct_url"]["editable"]:
                    installed = Path(unquote(parsed.path)).resolve()
                    if parsed.scheme != "file" or installed != repo:
                        errors.append("Editable vllm points outside repository")
            report["packages"][name] = entry
            if name in pins and distribution.version.split("+", 1)[0] != pins[name]:
                errors.append(
                    f"{name} version {distribution.version} != source pin {pins[name]}"
                )
        except Exception as exc:
            report["packages"][name] = {"error": f"{type(exc).__name__}: {exc}"}
            errors.append(f"Distribution {name}: {type(exc).__name__}: {exc}")

    try:
        from packaging.requirements import Requirement

        common = repo / "requirements/common.txt"
        bounds = {}
        for line in common.read_text().splitlines():
            if re.match(r"^(transformers|huggingface_hub)\s", line):
                requirement = Requirement(line.split("#", 1)[0].strip())
                actual = importlib.metadata.version(requirement.name)
                bounds[requirement.name] = str(requirement.specifier)
                if actual not in requirement.specifier:
                    errors.append(f"{requirement.name} {actual} violates {requirement}")
        report["expected"]["model_dependencies"] = bounds
    except Exception as exc:
        errors.append(f"Model dependency bounds: {type(exc).__name__}: {exc}")

    torch = source_module("torch")
    if torch is not None:
        report["torch"] = {
            "version": torch.__version__, "cuda_build_version": torch.version.cuda,
            "cxx11_abi": getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", None),
        }
        try:
            report["torch"].update(cuda_available=torch.cuda.is_available(),
                                     device_count=torch.cuda.device_count())
        except Exception as exc:
            report["torch"]["cuda_probe_error"] = f"{type(exc).__name__}: {exc}"
        if "torch" in pins and torch.__version__.split("+", 1)[0] != pins["torch"]:
            errors.append(
                f"Imported Torch version {torch.__version__} differs from source pin"
            )

    flashinfer = source_module("flashinfer")
    if flashinfer is not None:
        report["flashinfer"] = {
            "version": getattr(flashinfer, "__version__", None),
            **report["modules"]["flashinfer"],
        }
        actual = report["flashinfer"]["version"]
        if (actual is None
                or str(actual).split("+", 1)[0] != pins.get("flashinfer-python")):
            errors.append(
                f"Imported FlashInfer version {actual} differs from source pin"
            )
        jit = source_module("flashinfer.jit.env")
        if jit is not None:
            paths = {
                name: str(Path(value).expanduser().resolve())
                for name, value in vars(jit).items()
                if name.startswith("FLASHINFER_")
                and name.endswith(("_DIR", "_PATH", "_BASE"))
                and isinstance(value, (str, os.PathLike))
            }
            targets = {name: path for name, path in paths.items() if any(
                token in name for token in ("WORKSPACE", "CACHE", "JIT", "GEN_SRC")
            )}
            base = os.environ.get("FLASHINFER_WORKSPACE_BASE")
            redirected = None
            if base:
                root = Path(base).expanduser().resolve()
                redirected = bool(targets) and all(
                    Path(path).is_relative_to(root) for path in targets.values()
                )
                if not redirected:
                    errors.append("FlashInfer cache redirection failed; see jit_paths")
            report["flashinfer"].update(jit_paths=paths, checked_cache_paths=targets,
                                         workspace_env=base,
                                         workspace_redirected=redirected)

    for name, relative in {
        "vllm": "vllm/__init__.py",
        "vllm._C_stable_libtorch": None,
        "vllm._mwf_candidate_index": None,
        "vllm.v1.core.block_pool": "vllm/v1/core/block_pool.py",
        "vllm.v1.core.kv_cache_manager": "vllm/v1/core/kv_cache_manager.py",
        "vllm.v1.core.sched.scheduler": "vllm/v1/core/sched/scheduler.py",
        "vllm.v1.core.mwf_eviction": "vllm/v1/core/mwf_eviction.py",
    }.items():
        module = source_module(name, relative)
        if name == "vllm" and module is not None:
            report["modules"][name]["version"] = getattr(module, "__version__", None)
        elif module is not None and relative is None and not str(
            module.__file__
        ).endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES)):
            errors.append(f"Required native module is not a compiled extension: {name}")

    native_paths = set()
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("vllm.") and path and any(
            str(path).endswith(suffix)
            for suffix in importlib.machinery.EXTENSION_SUFFIXES
        ):
            native_paths.add(str(path))
    try:
        if torch is not None:
            native_paths.update(path for path in torch.ops.loaded_libraries
                                if "vllm" in Path(path).parts
                                or "vllm" in Path(path).name.lower())
        maps = Path("/proc/self/maps")
        if maps.is_file():
            for line in maps.read_text().splitlines():
                fields = line.split(None, 5)
                if len(fields) == 6 and ".so" in fields[5]:
                    path = fields[5].removesuffix(" (deleted)")
                    if ("vllm" in Path(path).parts
                            or "vllm" in Path(path).name.lower()):
                        native_paths.add(path)
    except Exception as exc:
        errors.append(f"Loaded binary inventory: {type(exc).__name__}: {exc}")
    for path in sorted(native_paths):
        try:
            record = file_record(path)
            report["loaded_vllm_binaries"][path] = record
            if not Path(record["realfile"]).is_relative_to(repo):
                errors.append(
                    f"Loaded vLLM binary outside requested repository: {path}"
                )
        except Exception as exc:
            errors.append(f"Loaded binary {path}: {type(exc).__name__}: {exc}")

    for name, arguments in {"nvidia-smi": [], "nvcc": ["--version"]}.items():
        executable = shutil.which(name)
        entry = {"executable": executable}
        if executable:
            try:
                result = subprocess.run([executable, *arguments], capture_output=True,
                                        text=True, timeout=15, check=False)
                entry.update(returncode=result.returncode, stdout=result.stdout,
                             stderr=result.stderr)
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
        report["commands"][name] = entry
    report["ready"] = not errors
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    validate_result_root(output.parent)
    os.chdir(output.parent)
    report = collect_environment(repo)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"MWF environment ready={report['ready']}; report={output}")
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    sys.exit(main())
