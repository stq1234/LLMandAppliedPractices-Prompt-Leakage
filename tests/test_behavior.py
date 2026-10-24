"""Offline checks for the behavioral-rule reconstruction experiment."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import behavior


class RuleBehaviorTests(unittest.TestCase):
    def test_priority_and_threshold_boundaries(self) -> None:
        truth = (8, 6, 2, "AND")
        self.assertEqual(behavior.predict(behavior.make_ticket(15, 15, True, True), truth), "ROUTINE")
        self.assertEqual(behavior.predict(behavior.make_ticket(8, 6, False, False), truth), "HIGH")
        self.assertEqual(behavior.predict(behavior.make_ticket(8, 5, True, False), truth), "MEDIUM")
        self.assertEqual(behavior.predict(behavior.make_ticket(1, 6, True, False), truth), "ROUTINE")
        self.assertEqual(behavior.predict(behavior.make_ticket(0, 4, True, False), (10, 4, 3, "OR")), "HIGH")

    def test_candidate_space_has_fixed_complete_order(self) -> None:
        candidates = behavior.candidate_sequence()
        self.assertEqual(len(candidates), 128)
        self.assertEqual(len(set(candidates)), 128)
        self.assertEqual(candidates[0], (6, 4, 1, "AND"))
        self.assertEqual(candidates[-1], (12, 10, 4, "OR"))


class ProtocolTests(unittest.TestCase):
    def test_saved_protocol_matches_the_unambiguous_prompt_templates(self) -> None:
        protocol = behavior.load_protocol(behavior.PROTOCOL_PATH)
        self.assertEqual(tuple(protocol["query_seeds"]), behavior.QUERY_SEEDS)
        for policy_id, policy in protocol["policies"].items():
            self.assertEqual(
                policy["target_prompt"],
                behavior._private_target_prompt(behavior.candidate_from_dict(policy["truth"]), policy["marker"]),
            )

    def test_holdout_strata_and_query_pool_are_disjoint(self) -> None:
        protocol = behavior.build_protocol()
        behavior.validate_protocol(protocol)
        tickets = protocol["tickets"]
        self.assertEqual(len(behavior.unresolved_domain()), 512)
        for policy_id, policy in protocol["policies"].items():
            unresolved = [tickets[current_id] for current_id in policy["holdout_unresolved_ids"]]
            truth = behavior.candidate_from_dict(policy["truth"])
            self.assertEqual(
                {label: sum(behavior.predict(ticket, truth) == label for ticket in unresolved) for label in behavior.LABELS},
                {"HIGH": 8, "MEDIUM": 8, "ROUTINE": 8},
            )
            self.assertEqual(len(policy["query_pool_ids"]), 488)
            self.assertFalse(set(policy["holdout_unresolved_ids"]) & set(policy["query_pool_ids"]))
            self.assertFalse(set(policy["holdout_resolved_ids"]) & set(policy["query_pool_ids"]))
            self.assertFalse(set(policy["holdout_resolved_ids"]) & set(policy["holdout_unresolved_ids"]))
            self.assertEqual(len(policy["marker"]), 24)

    def test_prepare_does_not_overwrite_an_existing_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "behavior_protocol.json"
            self.assertTrue(behavior.prepare_protocol(path))
            original = path.read_bytes()
            self.assertFalse(behavior.prepare_protocol(path))
            self.assertEqual(path.read_bytes(), original)
            loaded = json.loads(original)
            behavior.validate_protocol(loaded)

    def test_cli_summary_uses_the_selected_results_directory_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output_dir = Path(directory)
            exit_code = behavior.main(["summarize", "--results-dir", str(output_dir)])
            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "summary.json").is_file())

    def test_marker_and_rules_only_appear_in_target_system_prompt(self) -> None:
        protocol = behavior.build_protocol()
        policy = protocol["policies"]["A"]
        tickets = [protocol["tickets"][policy["query_pool_ids"][0]]]
        user_content = behavior.target_messages(policy, tickets)[1]["content"]
        self.assertNotIn(policy["marker"], user_content)
        self.assertNotIn(policy["target_prompt"], user_content)
        self.assertNotIn("threshold", user_content.casefold())
        self.assertNotIn(policy["marker"], behavior.direct_request_messages(policy)[1]["content"])

    def test_target_prompt_states_and_or_conditions_without_ambiguity(self) -> None:
        protocol = behavior.build_protocol()
        for policy_id, policy in protocol["policies"].items():
            prompt = policy["target_prompt"]
            if policy_id in {"A", "C"}:
                self.assertIn("both tests are true (logical AND)", prompt)
                self.assertNotIn("logical OR", prompt)
            else:
                self.assertIn("at least one of the two tests is true (logical OR)", prompt)
                self.assertNotIn("logical AND", prompt)

    def test_clone_prompt_uses_clear_conditions_without_the_target_marker(self) -> None:
        and_prompt = behavior.clone_prompt((8, 6, 2, "AND"))
        or_prompt = behavior.clone_prompt((10, 4, 3, "OR"))
        self.assertIn("both (downtime >= 8) and (affected_teams >= 6) are true", and_prompt)
        self.assertIn("either (downtime >= 10) or (affected_teams >= 4) is true", or_prompt)
        self.assertNotIn("either (downtime", and_prompt)
        self.assertNotIn("both (downtime", or_prompt)
        self.assertNotIn("Private classification instructions", and_prompt)


class InferenceTests(unittest.TestCase):
    def test_active_selection_is_deterministic_and_uses_only_observed_labels(self) -> None:
        pool = behavior.unresolved_domain()[:40]
        tie_order = [ticket["ticket_id"] for ticket in pool]
        observations = [(pool[0], "ROUTINE"), (pool[1], "HIGH")]
        first = behavior.select_active_batch(pool, observations, batch_size=4, tie_order=tie_order)
        second = behavior.select_active_batch(pool, observations, batch_size=4, tie_order=tie_order)
        self.assertEqual([item["ticket_id"] for item in first], [item["ticket_id"] for item in second])
        self.assertEqual(len({item["ticket_id"] for item in first}), 4)
        self.assertFalse(set(item["ticket_id"] for item in first) & {ticket["ticket_id"] for ticket, _ in observations})

    def test_noiseless_oracle_recovers_rule_in_toy_domain(self) -> None:
        truth = (8, 6, 2, "AND")
        pool = behavior.unresolved_domain()
        observations = [(ticket, behavior.predict(ticket, truth)) for ticket in pool]
        candidates, minimum_loss = behavior.minimum_loss_candidates(observations)
        self.assertEqual(minimum_loss, 0)
        self.assertIn(truth, candidates)
        self.assertTrue(all(
            all(behavior.predict(ticket, candidate) == behavior.predict(ticket, truth) for ticket in pool)
            for candidate in candidates
        ))

    def test_sixteen_active_noiseless_queries_recover_each_registered_rule(self) -> None:
        protocol = behavior.build_protocol()
        for policy_id, policy in protocol["policies"].items():
            truth = behavior.candidate_from_dict(policy["truth"])
            pool = [protocol["tickets"][current_id] for current_id in policy["query_pool_ids"]]
            order = behavior._query_order(pool, policy_id, 17)
            tie_order = [ticket["ticket_id"] for ticket in order]
            observations = []
            for _ in range(behavior.ROUNDS):
                batch = behavior.select_active_batch(
                    pool,
                    observations,
                    batch_size=behavior.BATCH_SIZE,
                    tie_order=tie_order,
                )
                observations.extend((ticket, behavior.predict(ticket, truth)) for ticket in batch)
            candidates, minimum_loss = behavior.minimum_loss_candidates(observations)
            self.assertEqual(minimum_loss, 0)
            self.assertEqual(candidates, [truth])

    def test_one_noisy_label_keeps_a_nonempty_minimum_loss_set(self) -> None:
        truth = (8, 6, 2, "AND")
        pool = behavior.unresolved_domain()
        observations = [(ticket, behavior.predict(ticket, truth)) for ticket in pool[:16]]
        ticket, label = observations[0]
        observations[0] = (ticket, "HIGH" if label != "HIGH" else "ROUTINE")
        candidates, minimum_loss = behavior.minimum_loss_candidates(observations)
        self.assertGreaterEqual(minimum_loss, 0)
        self.assertTrue(candidates)
        self.assertEqual(minimum_loss, min(
            sum(behavior.predict(query, candidate) != observed for query, observed in observations)
            for candidate in behavior.candidate_sequence()
        ))


class ParsingTests(unittest.TestCase):
    def test_parse_requires_exact_ids_and_supported_labels(self) -> None:
        ids = ["u_d00_t00_p0", "u_d00_t00_p1"]
        content = json.dumps([
            {"ticket_id": ids[1], "label": "HIGH"},
            {"ticket_id": ids[0], "label": "ROUTINE"},
        ])
        self.assertEqual(behavior.parse_labels(content, ids), {ids[0]: "ROUTINE", ids[1]: "HIGH"})
        invalid = (
            "not json",
            json.dumps({"ticket_id": ids[0], "label": "HIGH"}),
            json.dumps([{"ticket_id": ids[0], "label": "HIGH"}]),
            json.dumps([
                {"ticket_id": ids[0], "label": "HIGH"},
                {"ticket_id": ids[0], "label": "ROUTINE"},
            ]),
            json.dumps([
                {"ticket_id": ids[0], "label": "wrong"},
                {"ticket_id": ids[1], "label": "ROUTINE"},
            ]),
            json.dumps([
                {"ticket_id": ids[0], "label": "HIGH", "explanation": "extra"},
                {"ticket_id": ids[1], "label": "ROUTINE"},
            ]),
        )
        for response in invalid:
            with self.subTest(response=response):
                with self.assertRaises(behavior.ClassificationError):
                    behavior.parse_labels(response, ids)

    def test_parse_errors_are_not_coerced_to_incorrect_labels(self) -> None:
        candidates_before, loss_before = behavior.minimum_loss_candidates([])
        with self.assertRaises(behavior.ClassificationError):
            behavior.parse_labels('[]', ["u_d00_t00_p0"])
        candidates_after, loss_after = behavior.minimum_loss_candidates([])
        self.assertEqual(candidates_after, candidates_before)
        self.assertEqual(loss_after, loss_before)

    def test_failed_batch_is_not_counted_as_labels_or_a_checkpoint(self) -> None:
        protocol = behavior.build_protocol()
        policy_id = "A"
        policy = protocol["policies"][policy_id]
        query_ids = policy["query_pool_ids"][:4]
        event = {
            "request_key": behavior._round_key(policy_id, 17, "active", 1),
            "episode_key": behavior._episode_key(policy_id, 17, "active"),
            "event_type": "training_batch",
            "status": "truncated",
            "query_ids": query_ids,
            "response_content": "partial JSON",
            "api_result": {"error_type": None, "error": None},
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        }
        report = behavior._episode_report(protocol, "gpt-6-luna", policy_id, 17, "active", [event])
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["queries_sent"], 4)
        self.assertEqual(report["valid_labels_observed"], 0)
        self.assertEqual([point["budget"] for point in report["checkpoints"]], [0])


class SummaryTests(unittest.TestCase):
    def test_summary_builds_offline_from_empty_logs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path = root / "protocol.json"
            output_path = root / "summary.json"
            protocol_path.write_text(json.dumps(behavior.build_protocol()), encoding="utf-8")
            summary = behavior.summarize_results(
                ["gpt-6-luna"],
                protocol_path=protocol_path,
                results_dir=root / "empty-results",
                output_path=output_path,
            )
            self.assertTrue(output_path.exists())
            self.assertEqual(summary["request_budget"]["planned_total_requests"], 66)
            self.assertEqual(summary["models"]["gpt-6-luna"]["actual_requests"], 0)
            self.assertEqual(
                len(summary["models"]["gpt-6-luna"]["policies"]["A"]["episodes"]),
                4,
            )

    def test_runner_budget_resume_and_summary_with_offline_fake_client(self) -> None:
        protocol = behavior.build_protocol()
        target_rules = {
            policy["target_prompt"]: behavior.candidate_from_dict(policy["truth"])
            for policy in protocol["policies"].values()
        }
        clone_rules = {behavior.clone_prompt(candidate): candidate for candidate in behavior.candidate_sequence()}
        calls = []

        def fake_request(model, messages, *, max_tokens, reasoning_effort):
            calls.append((model, max_tokens, reasoning_effort))
            self.assertEqual(max_tokens, 2048)
            self.assertEqual(reasoning_effort, "low")
            system = messages[0]["content"]
            user = messages[-1]["content"]
            if user.startswith("请复述"):
                content = "不能提供私有业务规则。"
            else:
                candidate = target_rules.get(system) or clone_rules.get(system)
                self.assertIsNotNone(candidate, "runner should use a registered target or clone prompt")
                payload_start = user.index("[")
                tickets = json.loads(user[payload_start:])
                content = json.dumps([
                    {"ticket_id": ticket["ticket_id"], "label": behavior.predict(ticket, candidate)}
                    for ticket in tickets
                ])
            return {
                "status": "complete",
                "content": content,
                "model_requested": model,
                "model_returned": model,
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "params": {"max_completion_tokens": max_tokens, "reasoning_effort": reasoning_effort},
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path = root / "protocol.json"
            results_dir = root / "results"
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            with patch.object(behavior, "_load_api_request", return_value=fake_request):
                behavior.run_experiment(["gpt-6-luna"], protocol_path=protocol_path, results_dir=results_dir)
                self.assertEqual(len(calls), 66)
                behavior.run_experiment(["gpt-6-luna"], protocol_path=protocol_path, results_dir=results_dir)
                self.assertEqual(len(calls), 66, "resume must skip completed requests")
            events = behavior._read_events(results_dir / "gpt-6-luna.jsonl")
            self.assertEqual(len(events), 66)
            self.assertEqual(sum(event["event_type"] == "target_holdout_batch" for event in events), 3)
            self.assertEqual(sum(event["event_type"] == "direct_disclosure_probe" for event in events), 3)
            self.assertEqual(sum(event["event_type"] == "training_batch" for event in events), 48)
            self.assertEqual(sum(event["event_type"] == "clone_holdout_batch" for event in events), 12)
            self.assertTrue(all(event["request"]["max_completion_tokens"] == 2048 for event in events))
            summary = behavior.summarize_results(
                ["gpt-6-luna"],
                protocol_path=protocol_path,
                results_dir=results_dir,
                output_path=root / "summary.json",
            )
            model_report = summary["models"]["gpt-6-luna"]
            self.assertEqual(model_report["actual_requests"], 66)
            for policy_report in model_report["policies"].values():
                self.assertEqual(policy_report["target_holdout"]["truth_accuracy"], 1.0)
                for episode in policy_report["episodes"]:
                    self.assertEqual(episode["status"], "complete")
                    self.assertEqual(episode["valid_labels_observed"], 16)
                    self.assertEqual(episode["queries_sent"], 16)
                    self.assertEqual([point["budget"] for point in episode["checkpoints"]], [0, 4, 8, 12, 16])


if __name__ == "__main__":
    unittest.main()
