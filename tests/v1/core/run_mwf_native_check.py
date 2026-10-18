# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a fresh standalone native index and run the existing binding tests."""

import argparse
import ast
import hashlib
import importlib.util
import json
import shlex
import subprocess
import sys
import sysconfig
import tempfile
import unittest
from pathlib import Path

INPUTS = (
    "csrc/mwf_candidate_index.h",
    "csrc/mwf_candidate_index.cpp",
    "csrc/mwf_candidate_index_bindings.cpp",
    "tests/v1/core/test_mwf_candidate_index.py",
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", required=True, help="Compiler command, shell quoted")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument("--verify-only", action="store_true", help="Only verify inputs")
    args = parser.parse_args()
    root = args.root.resolve()
    verify_inputs(root)
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
    output = Path(tempfile.mkdtemp(prefix="mwf-native-check-")).resolve()
    extension = output / f"_mwf_candidate_index{suffix}"
    print(f"Python: {sys.executable}\nVersion: {sys.version}", flush=True)
    print(f"SOABI: {sysconfig.get_config_var('SOABI')}", flush=True)
    print(f"Torch package: {torch_spec.origin}", flush=True)
    subprocess.run([*cxx, "--version"], check=True)
    command = [*cxx, "-std=c++20", "-O2", "-shared", "-fPIC", "-fvisibility=hidden"]
    for include in [*includes, str(torch_include), str(root / "csrc")]:
        command.extend(["-I", include])
    command.extend(str(root / relative) for relative in INPUTS[1:3])
    command.extend(["-o", str(extension)])
    print(f"COMPILE {shlex.join(command)}", flush=True)
    subprocess.run(command, check=True)
    print(f"EXTENSION {extension} SHA256={sha256(extension)}", flush=True)
    spec = importlib.util.spec_from_file_location("_mwf_candidate_index", extension)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load extension: {extension}")
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    if Path(native.__file__).resolve() != extension:
        raise ImportError(f"Unexpected native origin: {native.__file__}")
    print(f"MODULE_FILE: {native.__file__}", flush=True)
    test_path = root / INPUTS[3]
    tree = ast.parse(test_path.read_text(), filename=str(test_path))
    imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "vllm._mwf_candidate_index"
    ]
    if len(imports) != 1:
        raise ValueError("Expected exactly one native import in the existing tests")
    tree.body = [node for node in tree.body if node not in imports]
    namespace = {
        "__name__": "mwf_native_component_tests",
        "CandidateKind": native.CandidateKind,
        "MWFCandidateIndex": native.MWFCandidateIndex,
    }
    exec(compile(tree, str(test_path), "exec"), namespace)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        namespace["TestMWFCandidateIndexBindings"]
    )
    if suite.countTestCases() != 5:
        raise ValueError("Expected the existing five binding tests")
    print(
        "Scope: fresh standalone native component; full vLLM import/build untested.",
        flush=True,
    )
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
