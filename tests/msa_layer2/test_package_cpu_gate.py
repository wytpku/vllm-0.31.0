# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standard-library gate control fixtures; no normal package/native execution."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


SPEC = importlib.util.spec_from_file_location(
    "msa_package_gate_under_test", Path(__file__).with_name("package_cpu_gate.py")
)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)
REPO = Path(__file__).resolve().parents[2]


class TestPackageGateHelpers(unittest.TestCase):
    def test_required_test_selection_is_cpu_scoped_and_nonempty(self):
        nodes = gate.selected_tests(REPO)
        self.assertIn("tests/v1/core/test_mwf_candidate_index.py", nodes)
        self.assertIn("tests/v1/core/test_mwf_eviction.py", nodes)
        self.assertIn(
            "tests/v1/worker/test_gpu_input_batch_v2.py::"
            "test_msa_prefill_cpu_packs_true_logical_positions", nodes
        )
        self.assertFalse(any("gpu_metadata_probe" in node for node in nodes))

    def test_origin_rejects_external_sources_and_python_native_substitute(self):
        source = Path(__file__).resolve()
        self.assertEqual(gate.require_origin(source, REPO), source)
        with self.assertRaises(ValueError):
            gate.require_origin(source, REPO, native=True)
        with self.assertRaises(ValueError):
            gate.require_origin(source, REPO / "vllm")

    def test_binding_requires_matching_031_editable_source(self):
        direct = {"url": REPO.as_uri(), "dir_info": {"editable": True}}
        result = gate.validate_binding(
            "0.31.1.dev17+g61dfe326", "0.31.1.dev17+g61dfe326", direct, REPO
        )
        self.assertTrue(result["environment_binding_passed"])
        self.assertEqual(result["editable_source"], str(REPO))
        self.assertEqual(
            result["normalized_runtime_version"], "0.31.1.dev17+g61dfe326"
        )

    def test_binding_decodes_file_url_and_normalizes_only_precompiled_suffix(self):
        repo = REPO / "fixture source"
        direct = {"url": repo.as_uri(), "dir_info": {"editable": True}}
        result = gate.validate_binding(
            "0.31.0+precompiled", "0.31.0", direct, repo
        )
        self.assertEqual(result["normalized_distribution_version"], "0.31.0")
        result = gate.validate_binding(
            "0.31.1.dev3+g123.precompiled", "0.31.1.dev3+g123", direct, repo
        )
        self.assertEqual(result["normalized_runtime_version"], "0.31.1.dev3+g123")
        self.assertEqual(result["editable_source"], str(repo))

    def test_binding_rejects_wrong_versions_and_stale_editable_install(self):
        direct = {"url": REPO.as_uri(), "dir_info": {"editable": True}}
        for distribution, runtime in (
            ("0.32.0", "0.32.0"), ("0.22.1", "0.22.1"),
            ("0.31.0", "0.31.1.dev1"), ("0.31.0.fake", "0.31.0.fake"),
            ("0.31.0.precompiled", "0.31.0.precompiled"),
            ("fixture", "fixture"),
        ):
            with self.subTest(distribution=distribution, runtime=runtime):
                with self.assertRaises(ValueError):
                    gate.validate_binding(distribution, runtime, direct, REPO)
        for invalid in (
            None, {},
            {"url": REPO.as_uri(), "dir_info": {"editable": False}},
            {"url": REPO.as_uri(), "dir_info": {"editable": 1}},
            {"url": REPO.parent.as_uri(), "dir_info": {"editable": True}},
            {"url": "https://example.org/repo", "dir_info": {"editable": True}},
            {"url": "file://other-host/repo", "dir_info": {"editable": True}},
        ):
            with self.subTest(direct_url=invalid):
                with self.assertRaises(ValueError):
                    gate.validate_binding("0.31.0", "0.31.0", invalid, REPO)

    def test_junit_rejects_success_with_skips_errors_or_empty_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "junit.xml"
            for kind in ("failure", "error", "skipped"):
                with self.subTest(kind=kind):
                    path.write_text(
                        '<testsuites><testsuite tests="1"><testcase name="case">'
                        f'<{kind} message="diagnostic">details</{kind}>'
                        '</testcase></testsuite></testsuites>'
                    )
                    result = gate.parse_junit(path)
                    self.assertFalse(gate.pytest_passed(0, result))
            path.write_text('<testsuite tests="0"/>')
            self.assertFalse(gate.pytest_passed(0, gate.parse_junit(path)))
            path.write_text(
                '<testsuite tests="1"><testcase name="case"/></testsuite>'
            )
            self.assertTrue(gate.pytest_passed(0, gate.parse_junit(path)))
            self.assertFalse(gate.pytest_passed(1, gate.parse_junit(path)))

    def test_junit_declared_collection_failure_is_not_lost(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "junit.xml"
            path.write_text(
                '<testsuite tests="1" errors="1"><testcase name="case"/></testsuite>'
            )
            self.assertEqual(gate.parse_junit(path)["errors"], 1)
            self.assertFalse(gate.pytest_passed(0, gate.parse_junit(path)))

    def test_output_and_json_writes_preserve_existing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            gate.write_json_new(path, {"original": True})
            with self.assertRaises(FileExistsError):
                gate.write_json_new(path, {"original": False})
            self.assertEqual(json.loads(path.read_text()), {"original": True})
        with self.assertRaises(ValueError):
            gate.validate_output(Path("/tmp/new-metadata-result.json"))

    def test_child_environment_removes_inherited_pytest_selection(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"PYTEST_ADDOPTS": "-k one_test -m subset",
                         "PYTEST_PLUGINS": "external_selector"}
        ):
            env = gate.child_environment(Path(directory))
        self.assertNotIn("PYTEST_ADDOPTS", env)
        self.assertNotIn("PYTEST_PLUGINS", env)

    def test_required_node_coverage_rejects_deselection_and_missing_execution(self):
        required = ["tests/fixture.py::TestGroup", "tests/other.py::test_param"]
        items = [
            SimpleNamespace(path=REPO / "tests/fixture.py",
                            nodeid="tests/fixture.py::TestGroup::test_first"),
            SimpleNamespace(path=REPO / "tests/other.py",
                            nodeid="tests/other.py::test_param[case]"),
        ]
        coverage = gate.RequiredPytestCoverage(REPO, required)
        coverage.pytest_collection_finish(SimpleNamespace(items=items))
        coverage.pytest_runtest_logreport(SimpleNamespace(
            when="call", nodeid=items[0].nodeid, context={"msg": "subcase"}
        ))
        self.assertEqual(coverage.result()["executed_nodeids"], [])
        coverage.pytest_runtest_logreport(
            SimpleNamespace(when="call", nodeid=items[0].nodeid)
        )
        self.assertFalse(coverage.result()["passed"])
        self.assertEqual(coverage.result()["unexecuted_nodeids"],
                         [items[1].nodeid])
        coverage.pytest_runtest_logreport(
            SimpleNamespace(when="call", nodeid=items[1].nodeid)
        )
        self.assertTrue(coverage.result()["passed"])
        coverage.pytest_deselected([items[0]])
        self.assertFalse(coverage.result()["passed"])
        missing = gate.RequiredPytestCoverage(REPO, required)
        missing.pytest_collection_finish(SimpleNamespace(items=items[:1]))
        self.assertEqual(missing.result()["missing_required_nodes"], required[1:])

    def test_owned_worker_group_is_cleaned_on_timeout_and_cancellation(self):
        for error in (InterruptedError("cancelled"),
                      subprocess.TimeoutExpired(["fixture"], 10)):
            with self.subTest(error=type(error).__name__):
                process = Mock(pid=12345)
                process.wait.side_effect = [
                    error, subprocess.TimeoutExpired(["fixture"], 2), 0,
                ]
                with tempfile.TemporaryDirectory() as directory, (
                    patch.object(gate.subprocess, "Popen", return_value=process)
                ), patch.object(gate.os, "killpg") as killed:
                    with self.assertRaises(type(error)):
                        gate.run_child(["fixture"], Path(directory) / "worker.log",
                                       Path(directory), {}, 10)
                self.assertEqual(killed.call_args_list, [
                    unittest.mock.call(12345, signal.SIGTERM),
                    unittest.mock.call(12345, signal.SIGKILL),
                ])

    def test_term_of_gate_owner_reaps_real_isolated_worker(self):
        driver = """
import os, runpy, signal, sys
from pathlib import Path
gate = runpy.run_path(sys.argv[1], run_name="control_fixture")
signal.signal(signal.SIGTERM, gate["handle_termination"])
folder = Path(sys.argv[2])
child = "import os, pathlib, sys, time; " + \\
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)"
try:
    gate["run_child"]([sys.executable, "-B", "-c", child, str(folder / "pid")],
                      folder / "worker.log", folder, os.environ.copy(), 60)
except InterruptedError:
    pass
"""
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            process = subprocess.Popen(
                [sys.executable, "-B", "-c", driver, gate.__file__, directory],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            worker_pid = None
            try:
                deadline = time.monotonic() + 10
                while not (folder / "pid").is_file():
                    if process.poll() is not None or time.monotonic() > deadline:
                        self.fail("Isolated fixture worker did not start")
                    time.sleep(0.02)
                worker_pid = int((folder / "pid").read_text())
                process.send_signal(signal.SIGTERM)
                _, stderr = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 0, stderr.decode())
                with self.assertRaises(ProcessLookupError):
                    os.kill(worker_pid, 0)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
                if worker_pid is not None:
                    try:
                        os.killpg(worker_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_alias_aware_audit_accepts_alias_and_bounds_a_corrupted_queue(self):
        pool = SimpleNamespace(mwf_eviction_manager=None)
        null = SimpleNamespace(block_id=0, pool=pool, is_null=True, ref_cnt=-1,
                               block_hash=None)
        block = SimpleNamespace(block_id=1, pool=pool, is_null=False, ref_cnt=0,
                                block_hash=b"primary")
        head, tail = SimpleNamespace(), SimpleNamespace()
        head.next_free_block = block
        block.prev_free_block, block.next_free_block = head, tail
        tail.prev_free_block = block
        pool.blocks = [null, block]
        pool.free_block_queue = SimpleNamespace(
            fake_free_list_head=head, fake_free_list_tail=tail, num_free_blocks=1
        )
        mapping = {b"primary": block, b"alias": block}
        pool.cached_block_hash_to_block = SimpleNamespace(
            _cache=mapping,
            contain=lambda key, block_id: key in mapping
            and mapping[key].block_id == block_id,
        )
        # The helper uses len(map), matching the actual cache map's API.
        class Map(SimpleNamespace):
            def __len__(self):
                return len(self._cache)

        pool.cached_block_hash_to_block = Map(
            **vars(pool.cached_block_hash_to_block)
        )
        pool.cached_block_hashes_by_block = {1: {b"alias"}}
        self.assertEqual(gate.audit_pool(pool)["aliases"], 1)
        block.next_free_block = block
        with self.assertRaises(AssertionError):
            gate.audit_pool(pool)


class TestPackageGateOrchestration(unittest.TestCase):
    def fixture(self):
        folder = tempfile.TemporaryDirectory(
            prefix="msa-package-unit-", dir=REPO.parent
        )
        self.addCleanup(folder.cleanup)
        return Path(folder.name) / "result.json"

    def child(self, mode, *, skipped=0, success=True):
        modules = {"fixture": {"path": "fixture", "sha256": "hash"}}
        result = {
            "success": success, "cuda_initialized": False, "modules": modules,
            "runtime_version": "0.31.1.dev3+g61dfe326",
            "distribution_version": "0.31.1.dev3+g61dfe326",
            "normalized_runtime_version": "0.31.1.dev3+g61dfe326",
            "normalized_distribution_version": "0.31.1.dev3+g61dfe326",
            "normal_package_imports_passed": True,
            "environment_binding_passed": True,
            "editable_source": str(REPO), "native_smoke_passed": True,
        }
        if mode == "pytest":
            result["pytest"] = {
                "tests": 3, "failures": 0, "errors": 0, "skipped": skipped,
                "exit_code": 0, "junit_path": "fixture.xml",
                "selection_coverage": {"passed": True},
            }
        return result

    def execute(self, results, calls):
        def executor(command, log, cwd, env, timeout):
            mode = command[command.index("--worker") + 1]
            calls.append(mode)
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
            self.assertNotIn("PYTHONPATH", env)
            self.assertNotIn("PYTEST_ADDOPTS", env)
            self.assertTrue(Path(env["TMPDIR"]).is_relative_to(cwd))
            Path(log).write_text("orchestration fixture; no package execution")
            output = Path(command[command.index("--output") + 1])
            gate.write_json_new(output, results[mode])
            return 0 if results[mode]["success"] else 1
        return executor

    def run_fixture(self, results, manifests=None):
        output, calls = self.fixture(), []
        with (
            patch.object(gate, "source_manifest", side_effect=manifests)
            if manifests is not None else patch.object(
                gate, "source_manifest", return_value={"fixture": "hash"}
            ),
            patch.object(gate.subprocess, "check_output", return_value="git-fixture"),
        ):
            report = gate.run_gate(REPO, output, 10, self.execute(results, calls))
        self.assertEqual(json.loads(output.read_text())["success"], report["success"])
        return report, calls, output

    def test_success_requires_both_workers_native_and_zero_skip_pytest(self):
        results = {mode: self.child(mode) for mode in ("inspect", "pytest")}
        report, calls, output = self.run_fixture(results)
        self.assertTrue(report["success"])
        self.assertEqual(calls, ["inspect", "pytest"])
        self.assertTrue(report["native_smoke_passed"])
        self.assertTrue(report["environment_binding_passed"])
        with self.assertRaises(FileExistsError):
            gate.run_gate(REPO, output, 10)

    def test_import_failure_stops_before_pytest_and_preserves_log(self):
        results = {"inspect": self.child("inspect", success=False)}
        report, calls, _ = self.run_fixture(results)
        self.assertFalse(report["success"])
        self.assertEqual(calls, ["inspect"])
        self.assertEqual(report["pytest"]["status"], "not_run")

    def test_missing_import_or_binding_confirmation_stops_before_pytest(self):
        for required in (
            "normal_package_imports_passed", "environment_binding_passed",
        ):
            with self.subTest(required=required):
                child = self.child("inspect")
                child[required] = False
                report, calls, _ = self.run_fixture({"inspect": child})
                self.assertFalse(report["success"])
                self.assertEqual(calls, ["inspect"])
                self.assertEqual(report["pytest"]["status"], "not_run")

    def test_successful_exit_with_required_skip_still_fails(self):
        results = {mode: self.child(mode, skipped=1) for mode in ("inspect", "pytest")}
        report, _, _ = self.run_fixture(results)
        self.assertFalse(report["success"])
        self.assertEqual(report["pytest"]["skipped"], 1)
        self.assertEqual(report["pytest"]["status"], "failed")

    def test_omitted_required_test_fails_even_when_junit_passed(self):
        results = {mode: self.child(mode) for mode in ("inspect", "pytest")}
        results["pytest"]["pytest"]["selection_coverage"]["passed"] = False
        report, _, _ = self.run_fixture(results)
        self.assertFalse(report["success"])
        self.assertEqual(report["pytest"]["status"], "failed")

    def test_source_drift_is_failure_even_when_children_succeeded(self):
        results = {mode: self.child(mode) for mode in ("inspect", "pytest")}
        report, _, _ = self.run_fixture(
            results, manifests=[{"fixture": "before"}, {"fixture": "after"},
                                {"fixture": "after"}],
        )
        self.assertFalse(report["success"])
        self.assertFalse(report["sources_stable"])


if __name__ == "__main__":
    unittest.main()
