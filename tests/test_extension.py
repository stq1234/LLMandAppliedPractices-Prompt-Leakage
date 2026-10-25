from __future__ import annotations

import base64
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import extension


PROJECT = Path(__file__).resolve().parents[1]
PROTOCOL = extension.load_protocol(PROJECT / "data" / "extension_protocol.json")


def complete_result(model: str, content: str = "I cannot provide that information.") -> dict:
    return {
        "status": "complete",
        "error_type": None,
        "error": None,
        "model_requested": model,
        "model_returned": model,
        "response_id": "resp-test",
        "content": content,
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "params": {"max_completion_tokens": 2048, "reasoning_effort": "low"},
        "elapsed_seconds": 0.1,
        "http_status": 200,
    }


class ExtensionProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.results_root = Path(self.temp.name) / "results" / "extension_dev"

    def test_fixed_development_plan_has_12_strategies_and_108_request_cap(self) -> None:
        plan = extension.build_plan(PROTOCOL, "dev")
        self.assertEqual(len(plan["strategy_ids"]), 12)
        self.assertEqual(plan["conversation_count"], 72)
        self.assertEqual(plan["planned_http_requests_upper_bound"], 108)
        self.assertEqual(sum(spec["planned_rounds"] for spec in plan["conversation_specs"]), 108)

    def test_validation_plan_uses_three_targets_two_repeats_and_selected_strategy_only(self) -> None:
        plan = extension.build_plan(PROTOCOL, "validation", ["incremental_prefix_spans"])
        self.assertEqual(plan["conversation_count"], 36)
        self.assertEqual(plan["target_set_ids"], ["validation_group_a", "validation_group_b", "validation_group_c"])
        self.assertEqual(plan["repeats_per_target_model_condition"], 2)
        self.assertEqual(plan["planned_http_requests_upper_bound"], 144)
        self.assertEqual({spec["strategy_id"] for spec in plan["conversation_specs"]}, {"incremental_prefix_spans"})

    def test_attack_messages_do_not_contain_canaries_or_rule_spans(self) -> None:
        extension.validate_attack_messages(PROTOCOL)
        for strategy in PROTOCOL["strategies"]:
            combined = "\n".join(strategy["user_messages"]).casefold()
            for target in PROTOCOL["target_sets"].values():
                for canary in target["target_canaries"]:
                    self.assertNotIn(canary.casefold(), combined)
                for instruction in target["target_instructions"]:
                    self.assertNotIn(extension.normalize_whitespace(instruction).casefold(), extension.normalize_whitespace(combined).casefold())

    def test_only_predeclared_reversible_decoders_are_used(self) -> None:
        instruction = PROTOCOL["target_sets"]["validation_group_a"]["target_instructions"][0]
        encoded_hex = instruction.encode("utf-8").hex()
        encoded_base64 = base64.b64encode(instruction.encode("utf-8")).decode("ascii")
        spaced = " ".join(instruction)
        json_fragments = json.dumps({"parts": [instruction[:25], instruction[25:]]})
        candidates = extension.decode_predeclared_formats(
            f"{encoded_hex}\n{encoded_base64}\n{spaced}\n{json_fragments}"
        )
        self.assertIn(instruction, candidates["hex_utf8"])
        self.assertIn(instruction, candidates["base64_utf8"])
        self.assertIn(instruction, candidates["spaced_characters"])
        self.assertIn(instruction, candidates["json_strings"])
        self.assertEqual(extension._decode_hex_candidates("abcdef0123456789"), [])
        invalid_utf8 = (bytes([255]) * 20).hex()
        self.assertEqual(extension._decode_hex_candidates(invalid_utf8), [])

    def test_scoring_keeps_marker_sentence_fragment_and_semantic_labels_separate(self) -> None:
        target_set = PROTOCOL["target_sets"]["dev_group_1"]
        instruction = target_set["target_instructions"][0]
        call = {
            "truth": {
                "target_canaries": target_set["target_canaries"],
                "target_instructions": target_set["target_instructions"],
                "target_system": "Confidential test system",
            },
            "round_results": [
                {"status": "complete", "content": target_set["target_canaries"][0]},
                {"status": "complete", "content": instruction},
                {"status": "complete", "content": "For family groups, suggest the"},
            ],
        }
        score = extension.score_conversation(call)
        self.assertEqual(score["raw_canary_hits"], 1)
        self.assertEqual(score["raw_complete_instruction_indices"], [1])
        self.assertEqual(score["raw_exact_rule_fragment_evidence"][0]["instruction_index"], 3)
        self.assertTrue(score["verified_text_evidence"])
        self.assertEqual(score["semantic_rule_leakage"], "requires_separate_semantic_review")
        self.assertFalse(score["raw_full_target_system_recovered"])

    def test_request_history_uses_actual_prior_assistant_response(self) -> None:
        plan = extension.build_plan(PROTOCOL, "dev")
        spec = next(item for item in plan["conversation_specs"] if item["strategy_id"] == "multiturn_public_then_challenger")
        history = [{"status": "complete", "content": "The public schedule is ..."}]
        messages = extension._request_messages(PROTOCOL, spec, history)
        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[2]["content"], history[0]["content"])
        self.assertEqual(messages[0]["content"], extension._system_prompt(PROTOCOL, "dev_group_1", spec["condition"]))

    def test_validation_is_blocked_without_verified_development_evidence(self) -> None:
        dev_plan = extension.build_plan(PROTOCOL, "dev")
        rows_by_model = {model: [] for model in extension.MODELS}
        for spec in dev_plan["conversation_specs"]:
            if spec["strategy_id"] != "translation_export":
                continue
            rows_by_model[spec["model"]].append(
                {
                    **spec,
                    "status": "complete",
                    "score": {"verified_text_evidence": False},
                    "round_results": [{"status": "complete", "content": "No."}],
                }
            )
        for model, rows in rows_by_model.items():
            extension._jsonl_write_atomic(
                self.results_root / "dev" / f"{model}.jsonl",
                rows,
                [row["conversation_id"] for row in rows],
            )
        with self.assertRaisesRegex(ValueError, "no verified development-set"):
            extension.prepare_phase(PROTOCOL, "validation", self.results_root, ["translation_export"])

    def test_validation_requires_complete_selected_dev_cells_and_verified_evidence(self) -> None:
        dev_plan = extension.build_plan(PROTOCOL, "dev")
        specs = [spec for spec in dev_plan["conversation_specs"] if spec["strategy_id"] == "translation_export"]
        rows_by_model = {model: [] for model in extension.MODELS}
        for spec in specs[:-1]:
            rows_by_model[spec["model"]].append(
                {
                    **spec,
                    "status": "complete",
                    "score": {"verified_text_evidence": True},
                    "round_results": [{"status": "complete", "content": "No."}],
                }
            )
        for model, rows in rows_by_model.items():
            extension._jsonl_write_atomic(
                self.results_root / "dev" / f"{model}.jsonl",
                rows,
                [row["conversation_id"] for row in rows],
            )
        with self.assertRaisesRegex(ValueError, "complete results"):
            extension.prepare_phase(PROTOCOL, "validation", self.results_root, ["translation_export"])

    def test_multiturn_runner_persists_actual_history_without_controls(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "local-test-key", "OPENAI_BASE_URL": "http://localhost:8080/v1"},
        ), mock.patch.object(
            extension.api_client, "request", side_effect=lambda model, **_kw: complete_result(model, "Public hours are posted.")
        ) as request_mock:
            with contextlib.redirect_stdout(io.StringIO()):
                outcome = extension.run_phase(
                    PROTOCOL,
                    "dev",
                    self.results_root,
                )
        self.assertEqual(outcome["planned_http_requests_upper_bound"], 108)
        self.assertEqual(request_mock.call_count, 108)
        multiturn_followup = next(
            message
            for strategy in PROTOCOL["strategies"]
            if strategy["strategy_id"] == "multiturn_public_then_challenger"
            for message in strategy["user_messages"][1:]
        )
        second_turn_calls = [
            call for call in request_mock.call_args_list
            if call.kwargs["messages"][-1]["content"] == multiturn_followup
        ]
        self.assertEqual(len(second_turn_calls), 6)
        self.assertTrue(all(call.kwargs["messages"][2] == {"role": "assistant", "content": "Public hours are posted."} for call in second_turn_calls))
        for model in extension.MODELS:
            rows = extension._jsonl_read(self.results_root / "dev" / f"{model}.jsonl")
            self.assertEqual(len(rows), 24)
            self.assertFalse(any(row["category"] == "authorized_control" for row in rows))
            self.assertTrue(all(row["status"] == "complete" for row in rows))

    def test_offline_summarize_writes_separate_semantic_review_queue(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            summary = extension.summarize_phase(PROTOCOL, "dev", self.results_root)
        self.assertEqual(summary["planned_conversation_count"], 72)
        self.assertEqual(summary["observed_http_request_count"], 0)
        self.assertFalse(summary["semantic_rule_leakage_is_automatically_scored"])
        self.assertTrue((self.results_root / "dev" / "summary.json").is_file())
        review_path = self.results_root / "dev" / "manual_review.jsonl"
        self.assertTrue(review_path.is_file())
        review_record = {
            "conversation_id": "review-provenance-test",
            "phase": "dev",
            "target_set_id": "dev_group_1",
            "repeat": 1,
            "model": extension.MODELS[0],
            "condition": "basic",
            "strategy_id": "incremental_rule_by_position",
            "status": "complete",
            "round_results": [{"round": 1, "content": "A mock response for offline testing."}],
        }
        reviews = extension._sync_manual_review_template(review_path.parent, [review_record])
        self.assertIsNone(reviews["review-provenance-test"]["reviewer_type"])
        self.assertIn("agent_adjudicated_semantic_leak_conversations", next(iter(summary["score_groups"].values())))
        reviewed = reviews["review-provenance-test"]
        reviewed["review_status"] = "reviewed"
        reviewed["reviewer_type"] = "agent"
        extension._jsonl_write_atomic(review_path, [reviewed], [reviewed["conversation_id"]])
        reloaded = extension._sync_manual_review_template(review_path.parent, [])
        self.assertEqual(reloaded["review-provenance-test"]["reviewer_type"], "agent")

    def test_prepare_cli_writes_the_dev_plan_without_api_configuration(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), mock.patch.dict("os.environ", {}, clear=True):
            exit_code = extension.main(["prepare", "--phase", "dev", "--results-dir", str(self.results_root)])
        self.assertEqual(exit_code, 0)
        self.assertIn("108 HTTP requests", stdout.getvalue())
        plan = json.loads((self.results_root / "dev" / "plan.json").read_text(encoding="utf-8"))
        self.assertEqual(plan["planned_http_requests_upper_bound"], 108)


if __name__ == "__main__":
    unittest.main()
