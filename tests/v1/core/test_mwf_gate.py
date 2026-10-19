# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Test gate orchestration with executor doubles; no native or GPU execution."""

import copy
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


RUNNER = Path(__file__).with_name("run_mwf_gate.py")
SPEC = importlib.util.spec_from_file_location("mwf_gate_runner_tests", RUNNER)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot load gate runner: {RUNNER}")
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)
RESULTS_ROOT = Path(__file__).resolve().parents[3].parent / "mwf gate"


class TestMWFGate(unittest.TestCase):
    def setUp(self):
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        directory = tempfile.TemporaryDirectory(
            prefix="mwf-gate-unit-", dir=RESULTS_ROOT
        )
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name)
        self.repo = self.folder / "repo"
        self.package = self.repo / "vllm"
        self.package.mkdir(parents=True)
        self.origin = self.package / "module.py"
        self.origin.write_text("value = 1\n")
        self.results = self.folder / "results"
        self.results.mkdir()

    def test_result_root_returns_the_existing_real_directory(self):
        alias = self.folder / "results-alias"
        alias.symlink_to(self.results, target_is_directory=True)
        expected = self.results.resolve()
        self.assertEqual(gate.validate_result_root(self.results), expected)
        self.assertEqual(gate.validate_result_root(alias), expected)

    def test_result_root_rejects_missing_files_and_temporary_locations(self):
        temporary_alias = self.folder / "temporary-alias"
        temporary_alias.symlink_to(Path("/tmp"), target_is_directory=True)
        for path in (
            self.folder / "missing",
            self.origin,
            Path(os.path.relpath(self.results)),
            Path("/tmp"),
            Path("/private/tmp"),
            temporary_alias,
        ):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    gate.validate_result_root(path)

    def test_origin_accepts_real_package_files_and_internal_symlinks(self):
        alias = self.package / "alias.py"
        alias.symlink_to(self.origin)
        expected = self.origin.resolve()
        self.assertEqual(gate.require_origin(self.origin, self.repo), expected)
        self.assertEqual(gate.require_origin(alias, self.repo), expected)

    def test_origin_rejects_other_trees_prefix_lookalikes_and_escaping_symlinks(self):
        foreign = self.folder / "old-vllm.py"
        foreign.write_text("value = 1\n")
        lookalike = self.repo / "vllm_other" / "module.py"
        lookalike.parent.mkdir()
        lookalike.write_text("value = 1\n")
        escaped = self.package / "escaped.py"
        escaped.symlink_to(foreign)
        for path in (
            foreign, lookalike, escaped, self.package / "missing.py", self.package
        ):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    gate.require_origin(path, self.repo)
        linked_repo = self.folder / "linked-repo"
        linked_repo.mkdir()
        (linked_repo / "vllm").symlink_to(self.package, target_is_directory=True)
        with self.assertRaises(ValueError):
            gate.require_origin(self.origin, linked_repo)

    def test_success_reports_only_requested_scope_and_keeps_gpu_not_run(self):
        names = ["inspect", "native", "package"]
        calls = []

        def execute(stage):
            calls.append(stage)
            return {"status": "passed", "unit_test_double": True}

        result = gate.run_stages(names, execute)
        self.assertEqual(calls, names)
        self.assertEqual(result["scope"], names)
        self.assertEqual(list(result["stages"]), names)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["gpu_gate"], "not_run")
        self.assertEqual(result["full_build_gate"], "not_run")
        self.assertEqual(result["performance_gate"], "not_run")

    def test_nonpassing_stage_stops_later_stages_and_cannot_count_as_pass(self):
        names = ["inspect", "native", "package"]
        for failure in ("failed", "skipped"):
            with self.subTest(status=failure):
                calls = []

                def execute(stage, failure=failure, calls=calls):
                    calls.append(stage)
                    return {"status": failure if stage == "native" else "passed"}

                result = gate.run_stages(names, execute)
                self.assertEqual(calls, ["inspect", "native"])
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["stages"]["inspect"]["status"], "passed")
                self.assertNotEqual(result["stages"]["native"]["status"], "passed")
                self.assertEqual(result["stages"]["package"]["status"], "not_run")

    def test_executor_exception_is_recorded_and_stops_remaining_stages(self):
        calls = []

        def execute(stage):
            calls.append(stage)
            if stage == "native":
                raise RuntimeError("planned executor failure")
            return {"status": "passed"}

        result = gate.run_stages(["inspect", "native", "package"], execute)
        self.assertEqual(calls, ["inspect", "native"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["stages"]["native"]["status"], "failed")
        self.assertIn("planned executor failure", str(result["stages"]["native"]))
        self.assertEqual(result["stages"]["package"]["status"], "not_run")

    def test_gpu_pass_requires_all_three_requested_suites_to_pass(self):
        suites = ["gpu-basic", "gpu-shared_cancel", "gpu-preemption"]
        for count in (1, 2, 3):
            with self.subTest(suite_count=count):
                result = gate.run_stages(suites[:count], lambda _: {"status": "passed"})
                self.assertEqual(result["scope"], suites[:count])
                if count == 3:
                    self.assertEqual(result["gpu_gate"], "passed")
                else:
                    self.assertNotEqual(result["gpu_gate"], "passed")
        failed = gate.run_stages(
            suites,
            lambda stage: {"status": "failed" if stage == suites[-1] else "passed"},
        )
        self.assertEqual(failed["status"], "failed")
        self.assertNotEqual(failed["gpu_gate"], "passed")

    def test_gpu_rejects_cpu_evidence_from_different_sources_or_runtime(self):
        manifest = {"source_digest": "current"}
        environment = {
            "python": {"version": "test-interpreter"}, "modules": {},
            "packages": {}, "loaded_vllm_binaries": {},
            "torch": {"version": "test-version", "cuda_build_version": "test-cuda",
                      "cxx11_abi": True},
        }
        summary = {"status": "passed", "stages": {
            name: {"status": "passed"} for name in ("native", "package")
        }}
        for name, value in (("manifest", manifest), ("environment", environment),
                            ("summary", summary)):
            (self.results / f"{name}.json").write_text(json.dumps(value))
        gate.verify_cpu_evidence(self.results, manifest, environment)
        for field in ("python", "modules", "packages", "loaded_vllm_binaries", "torch"):
            with self.subTest(field=field):
                changed = copy.deepcopy(environment)
                changed[field] = {"version": "changed"}
                with self.assertRaises(ValueError):
                    gate.verify_cpu_evidence(self.results, manifest, changed)
        with self.assertRaises(ValueError):
            gate.verify_cpu_evidence(
                self.results, {"source_digest": "other"}, environment
            )
        summary["stages"]["package"]["status"] = "not_run"
        (self.results / "summary.json").write_text(json.dumps(summary))
        with self.assertRaises(ValueError):
            gate.verify_cpu_evidence(self.results, manifest, environment)


if __name__ == "__main__":
    unittest.main()
