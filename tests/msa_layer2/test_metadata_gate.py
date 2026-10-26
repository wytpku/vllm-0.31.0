# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata gate acceptance and orchestration fixtures; no native/GPU calls."""

import copy
import hashlib
import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


FOLDER = Path(__file__).resolve().parent
REPO = FOLDER.parents[1]


def load_tool(name, filename):
    spec = importlib.util.spec_from_file_location(name, FOLDER / filename)
    if spec is None or spec.loader is None:
        raise ImportError(filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gate = load_tool("metadata_gate_orchestration_tests", "run_metadata_gate.py")
audit = load_tool("metadata_gate_cpu_contracts", "local_cpu_audit.py")
probe = load_tool("metadata_gate_probe_contracts", "gpu_metadata_probe.py")


class TestMetadataGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            cls.cpu_report = audit.run_audit(REPO)
        if not cls.cpu_report.get("success"):
            raise AssertionError(
                "Actual isolated CPU contracts failed before gate tests"
            )

    def setUp(self):
        root = Path(os.environ.get("MSA_GATE_TEST_ROOT", REPO.parent / "metadata gate"))
        root.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="gate-fixture-", dir=root)
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name).resolve()
        self.repo = self.folder / "repo"
        self.repo.mkdir()

    def test_stage_dependencies_are_ordered_and_unknown_stage_is_rejected(self):
        for index, name in enumerate(gate.STAGES):
            with self.subTest(stage=name):
                self.assertEqual(
                    tuple(gate.selected_stages(name)), gate.STAGES[:index + 1]
                )
        self.assertEqual(tuple(gate.selected_stages("all")), gate.STAGES)
        with self.assertRaises(ValueError):
            gate.selected_stages("unrecognized")

    def test_output_rejects_repository_temporary_and_symlinked_locations(self):
        alias = self.folder / "repo-alias"
        alias.symlink_to(self.repo, target_is_directory=True)
        for location in (
            self.repo, self.repo / "results", alias / "results",
            Path("/tmp/metadata-gate-fixture"),
            Path("/private/tmp/metadata-gate-fixture"),
        ):
            with self.subTest(location=location), self.assertRaises(ValueError):
                gate.validate_output(self.repo, location)
        output = self.folder / "new-run"
        self.assertEqual(gate.validate_output(self.repo, output), output)
        self.assertFalse(output.exists())

    def test_existing_evidence_is_preserved(self):
        existing = self.folder / "old-run"
        existing.mkdir()
        marker = existing / "summary.json"
        marker.write_text("original evidence")
        for location in (existing, marker):
            with (
                self.subTest(location=location),
                self.assertRaises((ValueError, FileExistsError)),
            ):
                gate.validate_output(self.repo, location)
        self.assertEqual(marker.read_text(), "original evidence")

    def test_manifest_binds_real_source_bytes_using_relative_paths(self):
        manifest = gate.source_manifest(REPO)
        for relative in (
            "vllm/v1/core/kv_cache_manager.py",
            "vllm/v1/worker/gpu/model_runner.py",
            "vllm/v1/attention/backends/flashinfer.py",
        ):
            self.assertEqual(
                manifest[relative],
                hashlib.sha256((REPO / relative).read_bytes()).hexdigest(),
            )
        self.assertTrue(all(
            not Path(name).is_absolute() and ".." not in Path(name).parts
            for name in manifest
        ))

    def test_cpu_validator_accepts_current_actual_contract_report(self):
        gate.validate_stage("cpu", copy.deepcopy(self.cpu_report),
                            gate.source_manifest(REPO), (16,))

    def test_exit_success_cannot_override_failed_or_incomplete_cpu_evidence(self):
        for changes in (
            {"success": False}, {"tests_run": 0}, {"failures": 1},
            {"errors": 1}, {"skips": 1}, {"sources_stable": False},
        ):
            with self.subTest(changes=changes):
                report = copy.deepcopy(self.cpu_report)
                report.update(changes)
                with self.assertRaises(ValueError):
                    gate.validate_stage(
                        "cpu", report, gate.source_manifest(REPO), (16,)
                    )

    def test_cpu_source_drift_is_rejected_even_when_tests_pass(self):
        report = copy.deepcopy(self.cpu_report)
        relative = "vllm/v1/core/kv_cache_manager.py"
        report["source_sha256"][relative] = "0" * 64
        with self.assertRaises(ValueError):
            gate.validate_stage("cpu", report, gate.source_manifest(REPO), (16,))

    def test_cpu_requires_implementation_sources_and_explicit_stability(self):
        for missing in ("source_sha256", "sources_stable"):
            with self.subTest(missing=missing):
                report = copy.deepcopy(self.cpu_report)
                report.pop(missing)
                with self.assertRaises(ValueError):
                    gate.validate_stage(
                        "cpu", report, gate.source_manifest(REPO), (16,)
                    )
        report = copy.deepcopy(self.cpu_report)
        report["source_sha256"] = {"unrelated.py": "0" * 64}
        with self.assertRaises(ValueError):
            gate.validate_stage("cpu", report, gate.source_manifest(REPO), (16,))

    def test_package_requires_actual_selection_coverage_without_skips(self):
        report = {
            "stage": "package", "success": True,
            "source_sha256": gate.source_manifest(REPO), "sources_stable": True,
            "normal_package_imports_passed": True,
            "environment_binding_passed": True, "native_smoke_passed": True,
            "cuda_initialized": False,
            "pytest": {"tests": 1, "failures": 0, "errors": 0, "skipped": 0,
                       "exit_code": 0, "selection_coverage": {"passed": True}},
        }
        gate.validate_stage("package", report, gate.source_manifest(REPO), (16,))
        for fault in ("missing_selection", "omitted_tests", "skipped"):
            with self.subTest(fault=fault):
                invalid = copy.deepcopy(report)
                if fault == "missing_selection":
                    invalid["pytest"].pop("selection_coverage")
                elif fault == "omitted_tests":
                    invalid["pytest"]["selection_coverage"]["passed"] = False
                else:
                    invalid["pytest"]["skipped"] = 1
                with self.assertRaises(ValueError):
                    gate.validate_stage(
                        "package", invalid, gate.source_manifest(REPO), (16,)
                    )

    def probe_fixture(self, stage="plan", pages=(16,)):
        """Synthetic validator input, never evidence of native/GPU execution."""
        expected = probe.expected_matrix(stage, pages)
        modules = {"fixture.only": {"path": "/fixture/module.py",
                                    "sha256": "0" * 64, "version": "fixture"}}
        cases = [dict(case_id=name, checks={"fixture_assertion": True}, calls=[])
                 for name in expected["case_ids"]]
        # Match the independent 2/2/3 wrapper counts for each A/B/C plan group.
        if stage == "plan":
            for case in cases:
                case["calls"] = [{} for _ in range(
                    3 if case["case_id"].endswith("/C") else 2
                )]
        return {
            "success": True, "stage": stage,
            "source_sha256": gate.source_manifest(REPO), "sources_stable": True,
            "modules": modules, "modules_after": copy.deepcopy(modules),
            "modules_stable": True, "attention_run_calls": 0,
            "cuda_initialized": stage != "import", "expected": expected,
            "coverage": {"case_ids": expected["case_ids"].copy(),
                         "dense_case_ids": expected["dense_case_ids"].copy()},
            "page_sizes": list(pages), "case_count": len(cases), "cases": cases,
            "plan_count": expected["plan_count"],
            "plan_calls": expected["plan_count"],
            "real_plan_calls": expected["plan_count"],
            "dense_progress_controls": [
                dict(case_id=name, checks={"logical_progress": True})
                for name in expected["dense_case_ids"]
            ],
        }

    def test_probe_fixture_acceptance_only_tests_report_schema(self):
        for stage in ("import", "kernels", "plan"):
            with self.subTest(stage=stage):
                gate.validate_stage(stage, self.probe_fixture(stage),
                                    gate.source_manifest(REPO), (16,))

    def test_gpu_expected_matrix_cannot_be_reduced_by_a_child(self):
        report = self.probe_fixture()
        report["expected"]["case_ids"].pop()
        report["expected"]["case_count"] -= 1
        report["coverage"]["case_ids"].pop()
        report["case_count"] -= 1
        report["cases"].pop()
        with self.assertRaises(ValueError):
            gate.validate_stage("plan", report, gate.source_manifest(REPO), (16,))

    def test_plan_requires_real_calls_and_all_independent_metadata_calls(self):
        for fault in ("missing_count", "skipped_plan", "missing_calls"):
            with self.subTest(fault=fault):
                report = self.probe_fixture()
                if fault == "missing_count":
                    report.pop("real_plan_calls")
                elif fault == "skipped_plan":
                    report["real_plan_calls"] -= 1
                else:
                    report["cases"][0]["calls"].pop()
                with self.assertRaises(ValueError):
                    gate.validate_stage(
                        "plan", report, gate.source_manifest(REPO), (16,)
                    )

    def test_gpu_missing_case_or_failed_assertion_cannot_pass(self):
        for fault in ("coverage", "cases", "check_false", "check_non_boolean"):
            with self.subTest(fault=fault):
                report = self.probe_fixture()
                if fault in ("coverage", "cases"):
                    target = (report["coverage"]["case_ids"] if fault == "coverage"
                              else report["cases"])
                    target.pop()
                else:
                    report["cases"][0]["checks"]["fixture_assertion"] = (
                        False if fault == "check_false" else 1
                    )
                with self.assertRaises(ValueError):
                    gate.validate_stage(
                        "plan", report, gate.source_manifest(REPO), (16,)
                    )

    def test_gpu_case_records_must_match_claimed_coverage(self):
        report = self.probe_fixture()
        report["cases"][0]["case_id"] = report["cases"][1]["case_id"]
        with self.assertRaises(ValueError):
            gate.validate_stage("plan", report, gate.source_manifest(REPO), (16,))

    def test_dense_controls_require_correct_ids_and_true_checks(self):
        for fault in ("missing", "wrong_id", "failed", "non_boolean"):
            with self.subTest(fault=fault):
                report = self.probe_fixture()
                dense = report["dense_progress_controls"]
                if fault == "missing":
                    dense.pop()
                elif fault == "wrong_id":
                    dense[0]["case_id"] = "dense/incorrect"
                else:
                    dense[0]["checks"]["logical_progress"] = (
                        False if fault == "failed" else 1
                    )
                with self.assertRaises(ValueError):
                    gate.validate_stage(
                        "plan", report, gate.source_manifest(REPO), (16,)
                    )

    def test_probe_execution_and_module_boundaries_are_enforced(self):
        for stage, changes in (
            ("plan", {"attention_run_calls": 1}),
            ("plan", {"modules_stable": False}),
            ("import", {"cuda_initialized": True}),
            ("plan", {"page_sizes": [32]}),
        ):
            with self.subTest(stage=stage, changes=changes):
                report = self.probe_fixture(stage)
                report.update(changes)
                with self.assertRaises(ValueError):
                    gate.validate_stage(
                        stage, report, gate.source_manifest(REPO), (16,)
                    )

    def args(self, **changes):
        args = SimpleNamespace(
            repo=REPO, output=self.folder / "run", stage="all", gpu="7",
            timeout=10, page_sizes=(16,), workspace_mib=128,
        )
        for name, value in changes.items():
            setattr(args, name, value)
        return args

    def cpu_executor(self, calls, *, success=False, timed_out=False, after=None):
        """Pretend only a child report was returned; never execute commands."""
        def execute(command, log, cwd, env, timeout):
            report_path = Path(command[command.index("--output") + 1])
            stage = report_path.stem
            calls.append(stage)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            if stage == "cpu":
                report = copy.deepcopy(self.cpu_report)
                report["success"] = success
            else:
                report = {"success": True, "stage": stage}
            report_path.write_text(json.dumps(report))
            log.write_text("Orchestration fixture; child command was not executed.\n")
            if after is not None and stage == "cpu":
                after()
            return {"exit_code": None if timed_out else 0,
                    "timed_out": timed_out, "elapsed_seconds": 0.0}
        return execute

    def stage_status(self, summary, name):
        stages = summary["stages"]
        if isinstance(stages, dict):
            return stages[name]["status"]
        return next(item["status"] for item in stages
                    if item.get("stage", item.get("name")) == name)

    def test_failed_cpu_report_blocks_every_later_child_even_with_exit_zero(self):
        calls = []
        summary = gate.execute_gate(self.args(), executor=self.cpu_executor(calls))
        self.assertFalse(summary["success"])
        self.assertIn("cpu", calls)
        self.assertFalse(set(calls) & {"package", "import", "kernels", "plan"})
        self.assertEqual(self.stage_status(summary, "cpu"), "failed")
        for stage in ("package", "import", "kernels", "plan"):
            self.assertEqual(self.stage_status(summary, stage), "not_run")

    def test_timed_out_child_cannot_pass_using_a_successful_report(self):
        calls = []
        summary = gate.execute_gate(
            self.args(stage="cpu"),
            executor=self.cpu_executor(calls, success=True, timed_out=True),
        )
        self.assertFalse(summary["success"])
        self.assertEqual(self.stage_status(summary, "cpu"), "failed")

    def test_nonzero_residual_process_or_missing_json_preserves_failed_evidence(self):
        for fault in ("nonzero", "residual", "missing_json"):
            with self.subTest(fault=fault):
                calls = []
                fixture = self.cpu_executor(calls, success=True)

                def execute(command, log, cwd, env, timeout):
                    result = fixture(command, log, cwd, env, timeout)
                    if fault == "missing_json":
                        Path(command[command.index("--output") + 1]).unlink()
                    elif fault == "nonzero":
                        result["exit_code"] = 2
                    else:
                        result["residual_process_group"] = True
                    return result

                output = self.folder / fault
                summary = gate.execute_gate(
                    self.args(output=output), executor=execute,
                )
                self.assertFalse(summary["success"])
                self.assertEqual(calls, ["cpu"])
                self.assertEqual(self.stage_status(summary, "cpu"), "failed")
                self.assertTrue((output / "summary.json").is_file())
                self.assertTrue((output / "cpu.log").is_file())
                self.assertTrue(output.with_name(output.name + ".tar.gz").is_file())

    def test_static_inspection_success_does_not_claim_complete_metadata_gate(self):
        calls = []
        summary = gate.execute_gate(
            self.args(stage="inspect"), executor=self.cpu_executor(calls),
        )
        self.assertTrue(summary["success"])
        self.assertFalse(summary["metadata_component_gate_passed"])
        self.assertEqual(calls, [])
        self.assertEqual(self.stage_status(summary, "inspect"), "passed")
        for stage in gate.STAGES[1:]:
            self.assertEqual(self.stage_status(summary, stage), "not_run")

    def test_source_drift_during_cpu_is_not_accepted(self):
        manifest = gate.source_manifest(REPO)
        drifted = dict(manifest)
        drifted["vllm/v1/core/kv_cache_manager.py"] = "0" * 64
        changed = False

        def drift():
            nonlocal changed
            changed = True

        def fingerprint(repo):
            return drifted if changed else manifest

        calls = []
        with patch.object(gate, "source_manifest", side_effect=fingerprint):
            summary = gate.execute_gate(
                self.args(stage="cpu"),
                executor=self.cpu_executor(calls, success=True, after=drift),
            )
        self.assertFalse(summary["success"])
        self.assertFalse(set(calls) & {"package", "import", "kernels", "plan"})

    def test_gpu_stages_without_explicit_device_never_start_a_child(self):
        for stage in ("plan", "all"):
            with self.subTest(stage=stage):
                calls = []
                try:
                    result = gate.execute_gate(
                        self.args(stage=stage, gpu=None),
                        executor=self.cpu_executor(calls),
                    )
                except ValueError:
                    pass
                else:
                    self.assertFalse(result["success"])
                self.assertEqual(calls, [])

    def test_archive_keeps_evidence_and_excludes_nested_cache_symlinks(self):
        output = self.folder / "archive-run"
        output.mkdir()
        (output / "summary.json").write_text('{"success": true}\n')
        cache = output / "cache" / "nested"
        cache.mkdir(parents=True)
        (cache / "external").symlink_to(self.repo, target_is_directory=True)
        log = output / "package-artifacts" / "pytest.log"
        log.parent.mkdir()
        log.write_text("retained failure details\n")
        archive_path = gate.archive_run(output)
        with tarfile.open(archive_path, "r:gz") as archive:
            self.assertEqual(set(archive.getnames()), {
                "archive-run/summary.json",
                "archive-run/package-artifacts/pytest.log",
            })
            self.assertEqual(
                archive.extractfile("archive-run/package-artifacts/pytest.log")
                .read(), b"retained failure details\n",
            )

    def test_archive_rejects_symlinks_in_evidence(self):
        output = self.folder / "foreign-evidence"
        output.mkdir()
        (output / "foreign.log").symlink_to(self.repo)
        with self.assertRaisesRegex(ValueError, "evidence symlink"):
            gate.archive_run(output)
        self.assertFalse(output.with_name(output.name + ".tar.gz").exists())
        self.assertTrue(
            output.with_name(output.name + ".tar.gz.partial").is_file()
        )

    def test_archive_failure_retains_stage_result_and_marks_delivery_failed(self):
        output = self.folder / "archive-failure"
        with patch.object(gate, "archive_run", side_effect=OSError("disk full")):
            summary = gate.execute_gate(self.args(stage="inspect", output=output))
        self.assertFalse(summary["success"])
        self.assertFalse(summary["delivery_success"])
        self.assertFalse(summary["metadata_component_gate_passed"])
        self.assertEqual(self.stage_status(summary, "inspect"), "passed")
        self.assertIn("disk full", summary["delivery_error"])
        self.assertEqual(json.loads((output / "summary.json").read_text()), summary)

    def test_real_child_success_retains_log_and_reaps_owned_group(self):
        log = self.folder / "child-success.log"
        result = gate.run_child(
            [sys.executable, "-B", "-c", "print('cpu child completed')"],
            log, REPO, os.environ.copy(), 10,
        )
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["timed_out"])
        self.assertFalse(result["residual_process_group"])
        self.assertIn("cpu child completed", log.read_text())

    def test_real_child_timeout_terminates_owned_process(self):
        result = gate.run_child(
            [sys.executable, "-B", "-c", "import time; time.sleep(30)"],
            self.folder / "child-timeout.log", REPO, os.environ.copy(), 0.05,
        )
        self.assertTrue(result["timed_out"])
        self.assertNotEqual(result["exit_code"], 0)

    def test_real_child_residual_group_is_detected_and_stopped(self):
        program = (
            "import subprocess, sys\n"
            "child = subprocess.Popen([sys.executable, '-B', '-c', "
            "'import time; time.sleep(30)'])\n"
            "print(child.pid, flush=True)\n"
        )
        result = gate.run_child(
            [sys.executable, "-B", "-c", program],
            self.folder / "child-residual.log", REPO, os.environ.copy(), 10,
        )
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["timed_out"])
        self.assertTrue(result["residual_process_group"])

    def test_interrupted_wait_cleans_owned_group_before_propagating(self):
        process = Mock(pid=123456)
        process.wait.side_effect = [KeyboardInterrupt, 0, 0]
        with (
            patch.object(gate.subprocess, "Popen", return_value=process),
            patch.object(gate.os, "killpg") as kill,
            self.assertRaises(KeyboardInterrupt),
        ):
            gate.run_child(
                [sys.executable, "-B", "-c", "pass"],
                self.folder / "cancelled.log", REPO, os.environ.copy(), 10,
            )
        self.assertEqual(kill.call_args_list[0].args, (123456, signal.SIGTERM))
        self.assertEqual(kill.call_args_list[-1].args, (123456, signal.SIGKILL))

    def test_term_of_orchestrator_reaps_real_owned_child(self):
        pidfile = self.folder / "worker.pid"
        worker = (
            "import os, sys, time\nfrom pathlib import Path\n"
            "Path(sys.argv[1]).write_text(str(os.getpid()))\ntime.sleep(30)\n"
        )
        owner = (
            "import importlib.util, os, signal, sys\nfrom pathlib import Path\n"
            f"spec = importlib.util.spec_from_file_location('owner_gate', "
            f"{str(FOLDER / 'run_metadata_gate.py')!r})\n"
            "gate = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(gate)\n"
            "signal.signal(signal.SIGTERM, gate.handle_termination)\n"
            "try:\n"
            f"    gate.run_child([sys.executable, '-B', '-c', {worker!r}, "
            f"{str(pidfile)!r}], {str(self.folder / 'owner.log')!r}, "
            f"Path({str(REPO)!r}), os.environ.copy(), 60)\n"
            "except InterruptedError:\n    pass\n"
        )
        process = subprocess.Popen([sys.executable, "-B", "-c", owner])
        worker_pid = None
        try:
            deadline = time.monotonic() + 10
            while not pidfile.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(pidfile.exists(), "Owned worker did not start")
            worker_pid = int(pidfile.read_text())
            process.send_signal(signal.SIGTERM)
            self.assertEqual(process.wait(timeout=15), 0)
            with self.assertRaises(ProcessLookupError):
                os.kill(worker_pid, 0)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            if worker_pid is not None:
                try:
                    os.killpg(worker_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    unittest.main()
