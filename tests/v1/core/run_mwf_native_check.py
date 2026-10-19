# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a fresh native index and run component tests from verified source files."""

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import unittest
from pathlib import Path

CPP_SOURCES = (
    "csrc/mwf_candidate_index.cpp",
    "csrc/mwf_candidate_index_bindings.cpp",
)
SCORE_SOURCE = "vllm/v1/core/mwf_eviction.py"
TEST_FILES = (
    "tests/v1/core/test_mwf_candidate_index.py",
    "tests/v1/core/test_mwf_eviction.py",
)
INPUTS = (
    "csrc/mwf_candidate_index.h",
    *CPP_SOURCES,
    SCORE_SOURCE,
    *TEST_FILES,
)
REQUIRED_INPUTS = (*INPUTS, "tests/v1/core/run_mwf_native_check.py")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs(root):
    manifest = root / "manifest.json"
    files = json.loads(manifest.read_text())["files"]
    if not isinstance(files, dict) or not set(REQUIRED_INPUTS).issubset(files):
        raise ValueError("Manifest files must include every build/test input")
    for relative, expected in sorted(files.items()):
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Invalid manifest input: {relative}")
        if sha256(path) != expected:
            raise ValueError(f"SHA256 mismatch: {relative}")
    print(f"Manifest verified: {len(files)} inputs", flush=True)
    for relative in INPUTS:
        print(f"SOURCE {sha256(root / relative)} {relative}", flush=True)
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    print(f"SOURCE_DIGEST: {digest}", flush=True)
    return digest


def load_file(name, path):
    path = path.resolve()
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path:
        raise ImportError(f"Unexpected module origin: {module.__file__}")
    return module


def snapshot_and_run(args, root, output_parent):
    def git(*arguments):
        return subprocess.check_output(
            ["git", "-C", str(root), *arguments], text=True
        ).strip()

    metadata = {
        "source_root": str(root),
        "source_git_head": git("rev-parse", "HEAD"),
        "source_git_status": git("status", "--porcelain=v1"),
    }
    snapshot = Path(
        tempfile.mkdtemp(prefix="mwf-source-check-", dir=output_parent)
    ).resolve() / "source"
    for relative in REQUIRED_INPUTS:
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    metadata["files"] = {
        relative: sha256(snapshot / relative) for relative in REQUIRED_INPUTS
    }
    (snapshot / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"SNAPSHOT: {snapshot}\nSOURCE_GIT_HEAD: {metadata['source_git_head']}",
        flush=True,
    )
    command = [
        sys.executable, "-B", str(snapshot / REQUIRED_INPUTS[-1]),
        "--root", str(snapshot), "--output-dir", str(output_parent),
        "--cxx", args.cxx,
    ]
    if args.verify_only:
        command.append("--verify-only")
    if args.expected_source_digest is not None:
        command.extend(["--expected-source-digest", args.expected_source_digest])
    print(f"RUN_SNAPSHOT {shlex.join(command)}", flush=True)
    return subprocess.run(command, check=False).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", required=True, help="Compiler command, shell quoted")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--root", type=Path, help="Existing manifest source root")
    source.add_argument("--snapshot-from", type=Path, help="Snapshot a Git checkout")
    parser.add_argument(
        "--output-dir", type=Path, help="Existing output parent directory"
    )
    parser.add_argument(
        "--expected-source-digest", help="Expected canonical files hash"
    )
    parser.add_argument("--verify-only", action="store_true", help="Only verify inputs")
    args = parser.parse_args()
    root = (
        args.snapshot_from or args.root or Path(__file__).resolve().parents[3]
    ).resolve()
    output_parent = (args.output_dir or root.parent).resolve()
    if not output_parent.is_dir():
        parser.error(f"Output parent directory does not exist: {output_parent}")
    if args.snapshot_from:
        return snapshot_and_run(args, root, output_parent)
    digest = verify_inputs(root)
    if (
        args.expected_source_digest is not None
        and args.expected_source_digest != digest
    ):
        raise ValueError(
            f"Source digest mismatch: {digest} != {args.expected_source_digest}"
        )
    if args.verify_only:
        return 0
    if sys.platform != "linux":
        parser.error("Native compilation is supported on Linux; use --verify-only here")
    cxx = shlex.split(args.cxx)
    if not cxx:
        parser.error("--cxx must contain a compiler command")
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None or torch_spec.origin is None:
        raise RuntimeError("Cannot locate installed Torch's pybind11 headers")
    torch_include = Path(torch_spec.origin).resolve().parent / "include"
    includes = list(
        dict.fromkeys(
            [sysconfig.get_path("include"), sysconfig.get_path("platinclude")]
        )
    )
    for header in (
        Path(includes[0]) / "Python.h",
        torch_include / "pybind11/pybind11.h",
    ):
        if not header.is_file():
            raise FileNotFoundError(header)
        print(f"HEADER {header} SHA256={sha256(header)}", flush=True)
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not suffix:
        raise RuntimeError("Python did not provide EXT_SUFFIX")
    output = Path(
        tempfile.mkdtemp(prefix="mwf-native-check-", dir=output_parent)
    ).resolve()
    compile_env = {**os.environ, "TMPDIR": str(output)}
    extension = output / f"_mwf_candidate_index{suffix}"
    print(f"OUTPUT_DIR: {output}", flush=True)
    print(f"Python: {sys.executable}\nVersion: {sys.version}", flush=True)
    print(f"SOABI: {sysconfig.get_config_var('SOABI')}", flush=True)
    print(f"Torch package: {torch_spec.origin}", flush=True)
    subprocess.run([*cxx, "--version"], check=True, env=compile_env)
    command = [*cxx, "-std=c++20", "-O2", "-shared", "-fPIC", "-fvisibility=hidden"]
    for include in [*includes, str(torch_include), str(root / "csrc")]:
        command.extend(["-I", include])
    command.extend(str(root / relative) for relative in CPP_SOURCES)
    command.extend(["-o", str(extension)])
    print(f"COMPILE {shlex.join(command)}", flush=True)
    subprocess.run(command, check=True, env=compile_env)
    print(f"EXTENSION {extension} SHA256={sha256(extension)}", flush=True)
    if "vllm" in sys.modules:
        raise ImportError("Standalone check must not import the vLLM root package")
    sys.dont_write_bytecode = True
    native = load_file("_mwf_candidate_index", extension)
    print(f"MODULE_FILE: {native.__file__}", flush=True)
    sys.modules["vllm._mwf_candidate_index"] = native
    print(
        "NATIVE_ALIAS_IDENTICAL:",
        sys.modules["vllm._mwf_candidate_index"] is native,
        flush=True,
    )
    score = load_file("vllm.v1.core.mwf_eviction", root / SCORE_SOURCE)
    print(f"SCORE_MODULE_FILE: {score.__file__}", flush=True)
    suite = unittest.TestSuite()
    for number, relative in enumerate(TEST_FILES):
        tests = load_file(f"mwf_component_tests_{number}", root / relative)
        component = unittest.defaultTestLoader.loadTestsFromModule(tests)
        if not component.countTestCases():
            raise ValueError(f"No component tests collected: {relative}")
        print(
            f"TEST_FILE: {tests.__file__} COUNT={component.countTestCases()}",
            flush=True,
        )
        suite.addTests(component)
    print(
        "Scope: fresh standalone native component; full vLLM import/build untested.",
        flush=True,
    )
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    if "vllm" in sys.modules:
        raise ImportError("Component tests unexpectedly imported the vLLM root package")
    print("VLLM_ROOT_IMPORTED: False", flush=True)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
