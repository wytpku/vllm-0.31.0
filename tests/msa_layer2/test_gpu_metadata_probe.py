# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standard-library gate infrastructure tests; no Torch, CUDA or FlashInfer."""

import contextlib
import importlib.util
import inspect
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

PROBE_PATH = Path(__file__).with_name("gpu_metadata_probe.py")
SPEC = importlib.util.spec_from_file_location("msa_gpu_probe_tests", PROBE_PATH)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class TestGPUMetadataProbe(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ))
        root = Path(__file__).resolve().parents[3].parent / "metadata gate"
        root.mkdir(exist_ok=True)
        folder = tempfile.TemporaryDirectory(prefix="probe-unit-", dir=root)
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.repo = self.root / "repo"
        for relative in probe.SOURCES:
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("source fixture\n")
        self.output = self.root / "results" / "probe.json"

    def modules(self, initialized=False):
        modules = {}
        for name in (*probe.MODULES, "torch"):
            path = self.repo / (name.replace(".", "/") + ".py")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name + " fixture\n")
            modules[name] = SimpleNamespace(
                __name__=name, __file__=str(path), __version__="fixture"
            )
        torch = modules["torch"]
        torch.version = SimpleNamespace(cuda="fixture")
        torch.cuda = SimpleNamespace(
            is_initialized=lambda: initialized,
            get_device_name=lambda index: "fixture GPU",
            get_device_capability=lambda index: (9, 0),
        )
        return torch, modules

    def args(self, stage="import"):
        args = [
            "--repo", str(self.repo), "--output", str(self.output), "--stage", stage
        ]
        return args if stage == "import" else [*args, "--gpu", "4"]

    def result_fixture(self, stage="plan"):
        expected = probe.expected_matrix(stage, (16,))
        cases = []
        for case_id in expected["case_ids"]:
            count = 3 if case_id.endswith("/C") else 2
            cases.append({"case_id": case_id, "checks": {"fixture": True},
                          "calls": [{}] * count if stage == "plan" else []})
        return {
            "cases": cases,
            "real_plan_calls": sum(len(case["calls"]) for case in cases),
            "dense_progress_controls": [
                {"case_id": case_id, "checks": {"fixture": True}}
                for case_id in expected["dense_case_ids"]
            ],
        }

    def test_canonical_a_b_c_ranges_have_expected_packed_counts(self):
        for page_size in (16, 32, 64):
            for name, multiplier, tail in (("A", 2, 1), ("B", 2, 0), ("C", 3, 1)):
                with self.subTest(page_size=page_size, scenario=name):
                    spec = probe.case_spec(page_size, name)
                    self.assertEqual(
                        spec["packed_q_count"], multiplier * page_size + tail
                    )
                    self.assertEqual(spec["positions"][-1], spec["logical_end"] - 1)
        self.assertEqual(probe.case_spec(16, "A")["logical_advance"], 65)
        self.assertEqual(probe.case_spec(16, "C")["logical_advance"], 97)

    def test_resume_shapes_share_endpoint_but_keep_different_query_rows(self):
        for name, start, rows in (
            ("resume_full", 0, 305), ("resume_prefix", 272, 33),
            ("resume_decode", 304, 1),
        ):
            with self.subTest(name=name):
                spec = probe.case_spec(16, name)
                self.assertEqual(spec["logical_start"], start)
                self.assertEqual(spec["packed_q_count"], rows)
                self.assertEqual(spec["positions"][-1], 304)
                self.assertEqual(spec["logical_end"], 305)
        self.assertEqual(probe.case_spec(64, "slot_reuse")["logical_end"], 65)

    def test_page_matrix_has_complete_unique_case_ids_and_plan_counts(self):
        kernels = probe.expected_matrix("kernels", (16, 32, 64))
        plans = probe.expected_matrix("plan", (16, 32, 64))
        self.assertEqual(kernels["case_count"], 21)
        self.assertEqual(plans["case_count"], 36)
        self.assertEqual(plans["plan_count"], 84)
        self.assertEqual(len(set(plans["case_ids"])), 36)
        for count in (1, 9, 20):
            pages = probe.physical_pages(count)
            self.assertEqual(len(pages), len(set(pages)))
            self.assertTrue(all(page > 0 for page in pages))

    def test_partial_or_duplicate_matrix_cannot_pass(self):
        expected = probe.expected_matrix("plan", (16,))
        valid = self.result_fixture()
        probe.validate_matrix(valid, expected)
        for mutation in (
            "missing", "duplicate", "check", "plans", "real_plans", "dense"
        ):
            with self.subTest(mutation=mutation):
                result = self.result_fixture()
                if mutation == "missing":
                    result["cases"].pop()
                elif mutation == "duplicate":
                    result["cases"][-1] = result["cases"][0]
                elif mutation == "check":
                    result["cases"][0]["checks"]["fixture"] = False
                elif mutation == "plans":
                    result["cases"][0]["calls"].pop()
                elif mutation == "real_plans":
                    result["real_plan_calls"] -= 1
                else:
                    result["dense_progress_controls"].pop()
                with self.assertRaises(AssertionError):
                    probe.validate_matrix(result, expected)

    def test_plan_counter_preserves_constructor_binding_return_and_restoration(self):
        calls, sentinel = [], object()

        class Wrapper:
            def __init__(self, value):
                self.value = value

            @property
            def label(self):
                return self.value

            def plan(self, value, *, option):
                calls.append((self, value, option))
                return sentinel

            def run(self):
                calls.append("unexpected native attention")

        original_plan, original_run = Wrapper.plan, Wrapper.run
        counter = {"attention_run_calls": 0, "real_plan_calls": 0}
        with probe.guard_flashinfer_calls(Wrapper, counter):
            wrapper = Wrapper("unchanged constructor")
            self.assertIs(type(wrapper), Wrapper)
            self.assertEqual(wrapper.label, "unchanged constructor")
            self.assertEqual(inspect.signature(Wrapper.plan),
                             inspect.signature(original_plan))
            self.assertIs(wrapper.plan(17, option="native"), sentinel)
            self.assertEqual(calls, [(wrapper, 17, "native")])
            self.assertEqual(counter["real_plan_calls"], 1)
            with self.assertRaisesRegex(AssertionError, "attention.run"):
                wrapper.run()
            self.assertEqual(counter["attention_run_calls"], 1)
            self.assertEqual(len(calls), 1)
        self.assertIs(Wrapper.plan, original_plan)
        self.assertIs(Wrapper.run, original_run)

    def test_plan_counter_preserves_native_exception_and_restores_methods(self):
        failure = RuntimeError("native plan failure")

        class Wrapper:
            def plan(self):
                raise failure

            def run(self):
                raise AssertionError("native run should be blocked")

        original_plan, original_run = Wrapper.plan, Wrapper.run
        counter = {"attention_run_calls": 0, "real_plan_calls": 0}
        with self.assertRaises(RuntimeError) as caught:
            with probe.guard_flashinfer_calls(Wrapper, counter):
                Wrapper().plan()
        self.assertIs(caught.exception, failure)
        self.assertEqual(counter["real_plan_calls"], 1)
        self.assertIs(Wrapper.plan, original_plan)
        self.assertIs(Wrapper.run, original_run)

    def test_plan_buffer_values_are_observed_for_full_and_partial_pages(self):
        def tensor(values, device="cpu"):
            return SimpleNamespace(device=device, tolist=lambda: list(values))

        for query_len, kv_len, pages, tail in (
            (16, 48, [9, 3, 7], 16), (17, 97, [9, 3, 7, 1, 8, 4, 6], 1)
        ):
            with self.subTest(kv_len=kv_len):
                values = [[0, query_len], [0, len(pages)], [tail], [kv_len], pages]
                call = SimpleNamespace(plan_buffers=tuple(map(tensor, values)))
                observed = probe.inspect_plan_buffers(
                    call, query_len, kv_len, 16, pages
                )
                self.assertEqual(observed, {
                    "qo_indptr": [0, query_len], "kv_indptr": [0, len(pages)],
                    "last_page_len": [tail], "seq_lens": [kv_len],
                    "page_ids_cpu": pages,
                })
                for index in range(len(values)):
                    mutated = list(call.plan_buffers)
                    mutated[index] = tensor([999])
                    with self.assertRaisesRegex(AssertionError, "values mismatch"):
                        probe.inspect_plan_buffers(
                            SimpleNamespace(plan_buffers=mutated),
                            query_len, kv_len, 16, pages,
                        )
                for invalid in (
                    call.plan_buffers[:-1],
                    (tensor([0, query_len], "cuda:0"), *call.plan_buffers[1:]),
                ):
                    with self.assertRaises(AssertionError):
                        probe.inspect_plan_buffers(
                            SimpleNamespace(plan_buffers=invalid),
                            query_len, kv_len, 16, pages,
                        )

    def test_idle_gpu_is_selected_by_uuid_and_busy_gpu_is_rejected(self):
        with patch.object(probe.subprocess, "check_output", side_effect=[
            "4, GPU-fixture, 3, 0\n", "GPU-other, 88\n"
        ]):
            result = probe.check_gpu("4")
        self.assertEqual(result["uuid"], "GPU-fixture")
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-fixture")
        for status, processes in (
            ("4, GPU-fixture, 2048, 0", ""),
            ("4, GPU-fixture, 3, 20", ""),
            ("4, GPU-fixture, 3, 0", "GPU-fixture, 99"),
        ):
            with self.subTest(status=status, processes=processes):
                with patch.object(probe.subprocess, "check_output",
                                  side_effect=[status, processes]):
                    with self.assertRaises(RuntimeError):
                        probe.check_gpu("4")

    def test_module_provenance_hashes_content_and_rejects_foreign_vllm(self):
        torch, modules = self.modules()
        record = probe.module_record(modules["vllm"])
        self.assertEqual(record["sha256"], probe.sha256(modules["vllm"].__file__))
        modules["vllm"].__file__ = str(self.root / "foreign.py")
        with patch.object(
            probe.importlib, "import_module", side_effect=modules.__getitem__
        ):
            with self.assertRaisesRegex(RuntimeError, "outside --repo"):
                probe.load_package_modules(self.repo)

    def test_cli_refuses_missing_gpu_duplicate_pages_and_existing_evidence(self):
        bad = [self.args("plan")[:-2], [*self.args(), "--page-sizes", "16,16"]]
        self.output.parent.mkdir()
        self.output.write_text("original evidence")
        bad.append(self.args())
        for args in bad:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    probe.parse_args(args)
        self.assertEqual(self.output.read_text(), "original evidence")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            probe.parse_args([
                "--output", "/private/tmp/probe-new.json", "--stage", "import"
            ])

    def run_fixture(self, modules, stage="import", **overrides):
        with (
            patch.object(probe, "load_package_modules", return_value=modules),
            patch.object(probe.subprocess, "check_output", return_value="fixture\n"),
            patch.object(probe, "check_gpu", return_value={"uuid": "GPU-fixture"}),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            with contextlib.ExitStack() as stack:
                for name, value in overrides.items():
                    stack.enter_context(patch.object(probe, name, value))
                code = probe.main(self.args(stage))
        return code, json.loads(self.output.read_text())

    def test_import_fixture_has_no_cuda_initialization_and_records_hashes(self):
        code, report = self.run_fixture(self.modules())
        self.assertEqual(code, 0)
        self.assertTrue(report["sources_stable"])
        self.assertEqual(report["gpu_kernels"], "not_run")
        self.assertEqual(report["attention_run_calls"], 0)
        self.assertEqual(report["real_plan_calls"], 0)
        self.assertEqual(report["script_sha256"], report["source_sha256"][
            "tests/msa_layer2/gpu_metadata_probe.py"])
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")

    def test_import_that_initializes_cuda_cannot_pass(self):
        code, report = self.run_fixture(self.modules(initialized=True))
        self.assertEqual(code, 1)
        self.assertFalse(report["success"])
        self.assertIn("initialized CUDA", report["error"])

    def test_source_change_during_import_cannot_pass(self):
        modules = self.modules()

        def changed(repo):
            (repo / probe.SOURCES[0]).write_text("changed during import\n")
            return modules

        code, report = self.run_fixture(
            modules, load_package_modules=Mock(side_effect=changed)
        )
        self.assertEqual(code, 1)
        self.assertFalse(report["sources_stable"])
        self.assertNotEqual(report["source_sha256"], report["source_sha256_after"])

    def test_complete_kernels_fixture_passes_only_its_requested_scope(self):
        records = self.result_fixture("kernels")
        code, report = self.run_fixture(
            self.modules(), stage="kernels",
            kernel_cases=Mock(return_value=records["cases"]),
            dense_progress_controls=Mock(
                return_value=records["dense_progress_controls"]
            ),
        )
        self.assertEqual(code, 0)
        self.assertEqual(report["case_count"], 7)
        self.assertEqual(report["gpu_kernels"], "passed")
        self.assertEqual(report["gpu_metadata_plan"], "not_run")
        self.assertEqual(report["real_plan_calls"], 0)

    def test_complete_plan_fixture_requires_matching_real_call_count_in_report(self):
        records = self.result_fixture()

        def completed(torch, backend, args, counter):
            counter["real_plan_calls"] = 28
            return records["cases"]

        code, report = self.run_fixture(
            self.modules(), stage="plan", plan_cases=Mock(side_effect=completed),
            dense_progress_controls=Mock(
                return_value=records["dense_progress_controls"]
            ),
        )
        self.assertEqual(code, 0)
        self.assertEqual(report["case_count"], 12)
        self.assertEqual(report["plan_count"], 28)
        self.assertEqual(report["plan_calls"], 28)
        self.assertEqual(report["real_plan_calls"], 28)
        self.assertEqual(report["attention_run_calls"], 0)

    def test_complete_plan_records_cannot_pass_if_real_plan_was_skipped(self):
        records = self.result_fixture()
        code, report = self.run_fixture(
            self.modules(), stage="plan",
            plan_cases=Mock(return_value=records["cases"]),
            dense_progress_controls=Mock(
                return_value=records["dense_progress_controls"]
            ),
        )
        self.assertEqual(code, 1)
        self.assertFalse(report["success"])
        self.assertEqual(report["real_plan_calls"], 0)
        self.assertIn("Real FlashInfer plan calls", report["error"])

    def test_attention_run_attempt_is_rejected_even_with_complete_plan_matrix(self):
        records = self.result_fixture()

        def attempted(torch, backend, args, counter):
            counter["attention_run_calls"] = 1
            counter["real_plan_calls"] = 28
            return records["cases"]

        code, report = self.run_fixture(
            self.modules(), stage="plan", plan_cases=Mock(side_effect=attempted),
            dense_progress_controls=Mock(
                return_value=records["dense_progress_controls"]
            ),
        )
        self.assertEqual(code, 1)
        self.assertFalse(report["success"])
        self.assertEqual(report["attention_run_calls"], 1)

    def test_imported_native_file_change_cannot_pass_a_kernel_stage(self):
        torch, modules = self.modules()
        records = self.result_fixture("kernels")

        def changed(torch, args):
            Path(modules["vllm._C_stable_libtorch"].__file__).write_text("new binary")
            return records["cases"]

        code, report = self.run_fixture(
            (torch, modules), stage="kernels", kernel_cases=Mock(side_effect=changed),
            dense_progress_controls=Mock(
                return_value=records["dense_progress_controls"]
            ),
        )
        self.assertEqual(code, 1)
        self.assertFalse(report["modules_stable"])
        self.assertIn("module files changed", report["error"])


if __name__ == "__main__":
    unittest.main()
