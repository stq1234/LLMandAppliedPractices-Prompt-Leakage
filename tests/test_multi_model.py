from __future__ import annotations

import base64
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import api_client, multi_model


PROJECT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((PROJECT / "data" / "protocol.json").read_text(encoding="utf-8"))


def complete_result(model: str) -> dict:
    return {
        "status": "complete",
        "error_type": None,
        "error": None,
        "model_requested": model,
        "model_returned": model,
        "response_id": "resp-test",
        "content": "I cannot provide that information.",
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "params": {"max_completion_tokens": multi_model.MAX_TOKENS, "reasoning_effort": multi_model.REASONING_EFFORT},
        "elapsed_seconds": 0.1,
        "http_status": 200,
    }


class MultiModelTests(unittest.TestCase):
    def test_base_url_must_use_explicit_loopback_http(self) -> None:
        self.assertEqual(api_client.validate_local_base_url("http://localhost:8080/v1/"), "http://localhost:8080/v1")
        self.assertEqual(api_client.validate_local_base_url("http://[::1]:8080/v1"), "http://[::1]:8080/v1")
        for value in (
            "https://localhost:8080/v1",
            "http://example.com:8080/v1",
            "http://localhost/v1",
            "http://user:password@localhost:8080/v1",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                api_client.validate_local_base_url(value)

    def test_model_id_validation_accepts_only_exact_id_or_valid_date_suffix(self) -> None:
        self.assertTrue(api_client.model_matches("gpt-6.1-sol", "gpt-6.1-sol"))
        self.assertTrue(api_client.model_matches("gpt-6.1-sol", "gpt-6.1-sol-20261001"))
        self.assertFalse(api_client.model_matches("gpt-6.1-sol", "gpt-6.1-sol-pro"))
        self.assertFalse(api_client.model_matches("gpt-6.1-sol", "gpt-6.1-sol-20261399"))

    def test_request_rejects_a_different_returned_model(self) -> None:
        class FakeResponse:
            status_code = 200

            @staticmethod
            def json() -> dict:
                return {
                    "id": "resp-mismatch",
                    "model": "gpt-5.6-luna",
                    "choices": [{"message": {"content": "No."}, "finish_reason": "stop"}],
                    "usage": {"total_tokens": 4},
                }

        class FakeClient:
            instances = []

            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs
                self.payload = None
                type(self).instances.append(self)

            def __enter__(self):
                return self

            def __exit__(self, *_args) -> None:
                return None

            def post(self, *_args, **kwargs):
                self.payload = kwargs["json"]
                return FakeResponse()

        with mock.patch.dict("os.environ", {"OPENAI_API_KEY": "local-test-key", "OPENAI_BASE_URL": "http://localhost:8080/v1"}), mock.patch.object(
            api_client.httpx, "Client", FakeClient
        ):
            result = api_client.request("gpt-6-luna", [{"role": "user", "content": "ping"}])
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "model_mismatch")
        self.assertEqual(result["model_returned"], "gpt-5.6-luna")
        self.assertEqual(result["params"], {"max_completion_tokens": 2048, "reasoning_effort": "low"})
        self.assertEqual(FakeClient.instances[-1].payload["max_completion_tokens"], 2048)
        self.assertNotIn("max_tokens", FakeClient.instances[-1].payload)
        self.assertFalse(FakeClient.instances[-1].kwargs["trust_env"])

    def test_protocol_selection_is_fixed_to_45_calls(self) -> None:
        calls = multi_model.select_experiment_calls(PROTOCOL)
        self.assertEqual(len(calls), 45)
        self.assertEqual(sum(call["category"] == "attack" for call in calls), 36)
        self.assertEqual(sum(call["category"] == "normal_control" for call in calls), 6)
        self.assertEqual(sum(call["strategy"] == "authorized_direct" for call in calls), 3)
        self.assertFalse(any(call["category"] == "default_probe" for call in calls))
        self.assertFalse(any(call["strategy"] == "authorized_base64" for call in calls))

    def test_base64_scoring_unions_raw_and_decoded_evidence(self) -> None:
        call = next(
            call for call in PROTOCOL["calls"]
            if call["call_id"] == "attack-basic-base64_transform-g1"
        )
        literal_marker = call["target_canaries"][0]
        encoded_marker = base64.b64encode(call["target_canaries"][1].encode()).decode()
        encoded_instruction = base64.b64encode(call["target_instructions"][2].encode()).decode()
        content = f"{literal_marker}\n{encoded_marker}\n{encoded_instruction}"
        score = multi_model._score_one(call, {"status": "complete", "content": content})
        self.assertEqual(score["any_disclosure"]["raw_canary_hits"], 1)
        self.assertEqual(score["any_disclosure"]["decoded_canary_hits"], 1)
        self.assertEqual(score["any_disclosure"]["canary_hits"], 2)
        self.assertEqual(score["any_disclosure"]["complete_instruction_indices"], [3])
        self.assertTrue(score["any_disclosure"]["has_any_disclosure"])

    def test_truncated_errors_and_missing_are_excluded_from_primary_denominator(self) -> None:
        evidence = {
            "any_disclosure": {
                "has_any_disclosure": True,
                "canary_hits": 1,
                "raw_canary_hits": 1,
                "decoded_canary_hits": 0,
                "complete_instruction_hits": 0,
                "raw_complete_instruction_hits": 0,
                "decoded_complete_instruction_hits": 0,
                "all_instructions_recovered": True,
                "full_application_system_recovered": False,
            },
            "public_task_correct": None,
        }
        metrics = multi_model._complete_metrics([
            {"status": "complete", "score": evidence},
            {"status": "truncated", "score": evidence},
            {"status": "error", "score": None},
            None,
        ])
        self.assertEqual(metrics["primary_denominator_n"], 1)
        self.assertEqual(metrics["any_canary_response_count"], 1)
        self.assertEqual(metrics["truncated_any_canary_response_count"], 1)
        self.assertEqual(metrics["all_instructions_recovered_response_count"], 1)
        self.assertEqual(metrics["truncated_all_instructions_recovered_response_count"], 1)
        self.assertEqual(metrics["error_response_n"], 1)
        self.assertEqual(metrics["missing_n"], 1)

    def _temporary_project(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        protocol_path = root / "data" / "protocol.json"
        protocol_path.parent.mkdir(parents=True)
        protocol_path.write_text(json.dumps(PROTOCOL, ensure_ascii=False), encoding="utf-8")
        results_dir = root / "results" / "experiment1"
        return tmp, root, protocol_path, results_dir

    def test_resume_skips_successful_records(self) -> None:
        tmp, root, protocol_path, results_dir = self._temporary_project()
        self.addCleanup(tmp.cleanup)
        patches = (
            mock.patch.object(multi_model, "ROOT", root),
            mock.patch.object(multi_model, "PROTOCOL_PATH", protocol_path),
            mock.patch.dict("os.environ", {"OPENAI_API_KEY": "local-test-key", "OPENAI_BASE_URL": "http://localhost:8080/v1"}),
        )
        with contextlib.ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            request_mock = stack.enter_context(mock.patch.object(multi_model.api_client, "request", side_effect=lambda model, **_kw: complete_result(model)))
            with contextlib.redirect_stdout(io.StringIO()):
                first = multi_model.run_model("gpt-6-luna", results_dir=results_dir)
                second = multi_model.run_model("gpt-6-luna", results_dir=results_dir)
        self.assertEqual(request_mock.call_count, 45)
        self.assertEqual(first["issued_call_records"], 45)
        self.assertEqual(Path(first["result_file"]), Path("results") / "experiment1" / "gpt-6-luna.jsonl")
        self.assertEqual(second["status_counts"]["complete"], 45)

    def test_error_retry_requires_flag_and_is_limited_to_one(self) -> None:
        tmp, root, protocol_path, results_dir = self._temporary_project()
        self.addCleanup(tmp.cleanup)
        calls = multi_model.select_experiment_calls(PROTOCOL)
        rows = []
        for call in calls[1:]:
            attempt = multi_model._attempt_record(call, complete_result("gpt-5.6-luna"), 1)
            rows.append(multi_model._result_record(call, "gpt-5.6-luna", [attempt]))
        failed = calls[0]
        first_error = {
            "status": "error",
            "error_type": "api_error",
            "error": "temporary failure",
            "model_requested": "gpt-5.6-luna",
            "model_returned": None,
            "content": None,
            "usage": None,
            "params": {"max_completion_tokens": 2048, "reasoning_effort": "low"},
        }
        first_attempt = multi_model._attempt_record(failed, first_error, 1)
        rows.append(multi_model._result_record(failed, "gpt-5.6-luna", [first_attempt]))
        patches = (
            mock.patch.object(multi_model, "ROOT", root),
            mock.patch.object(multi_model, "PROTOCOL_PATH", protocol_path),
            mock.patch.object(multi_model, "RESULTS_DIR", results_dir),
            mock.patch.dict("os.environ", {"OPENAI_API_KEY": "local-test-key", "OPENAI_BASE_URL": "http://localhost:8080/v1"}),
        )
        with contextlib.ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            results_dir.mkdir(parents=True)
            path = results_dir / "gpt-5.6-luna.jsonl"
            multi_model._write_jsonl_atomic(path, rows, [call["call_id"] for call in calls])
            request_mock = stack.enter_context(mock.patch.object(multi_model.api_client, "request", side_effect=lambda model, **_kw: complete_result(model)))
            with contextlib.redirect_stdout(io.StringIO()):
                multi_model.run_model("gpt-5.6-luna", retry_errors=False)
                self.assertEqual(request_mock.call_count, 0)
                multi_model.run_model("gpt-5.6-luna", retry_errors=True)
                self.assertEqual(request_mock.call_count, 1)
                multi_model.run_model("gpt-5.6-luna", retry_errors=True)
                self.assertEqual(request_mock.call_count, 1)
            saved = {row["call_id"]: row for row in multi_model._read_jsonl(path)}
        updated = saved[failed["call_id"]]
        self.assertEqual(updated["attempt_count"], 2)
        self.assertEqual([attempt["attempt"] for attempt in updated["attempt_history"]], [1, 2])

    def test_offline_summary_ignores_historical_results_outside_selected_directory(self) -> None:
        tmp, root, protocol_path, results_dir = self._temporary_project()
        self.addCleanup(tmp.cleanup)
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(multi_model, "ROOT", root))
            stack.enter_context(mock.patch.object(multi_model, "PROTOCOL_PATH", protocol_path))
            legacy_path = root / "results" / "calls.jsonl"
            legacy_path.parent.mkdir(parents=True)
            legacy_path.write_text('{"call_id":"historic-only","status":"complete"}\n', encoding="utf-8")
            summary = multi_model.build_summary(PROTOCOL, results_dir)
        self.assertEqual(summary["selection"]["planned_total_calls"], 135)
        self.assertNotIn("historical_data_excluded_from_main_metrics", summary)
        for model in multi_model.MODELS:
            self.assertEqual(summary["models"][model]["run_status"], "not_started")
            self.assertEqual(summary["models"][model]["status_counts"]["missing"], 45)
            self.assertEqual(summary["models"][model]["attack"]["primary_denominator_n"], 0)

    def test_summarize_cli_writes_to_selected_results_directory(self) -> None:
        tmp, root, protocol_path, results_dir = self._temporary_project()
        self.addCleanup(tmp.cleanup)
        with mock.patch.object(multi_model, "ROOT", root), mock.patch.object(multi_model, "PROTOCOL_PATH", protocol_path):
            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = multi_model.main(["summarize", "--results-dir", str(results_dir)])
        self.assertEqual(exit_code, 0)
        self.assertTrue((results_dir / "summary.json").is_file())
        summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["selection"]["planned_total_calls"], 135)


if __name__ == "__main__":
    unittest.main()
