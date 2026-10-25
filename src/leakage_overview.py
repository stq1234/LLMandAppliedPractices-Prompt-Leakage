"""Aggregate the saved prompt-leakage experiments and render the dialogue-recovery figure.

This module is strictly offline. Adaptive semantic and exact-rule outcomes are
computed from defender responses; attacker outputs are retained only as audit
context in the source records.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from . import extension

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results"
OVERVIEW_PATH = RESULTS_ROOT / "leakage_overview.json"
FIGURE_PATH = ROOT / "figures" / "experiment1_dialogue_recovery.png"
MODELS = ("gpt-6-luna", "gpt-5.6-luna", "gpt-6.1-sol")
ADAPTIVE_ATTACKERS = ("gpt-6-luna", "gpt-6.1-sol")
ADAPTIVE_DEFENDER = "gpt-6-luna"
ADAPTIVE_CONDITIONS = ("none", "weak", "structured")
MAX_ADAPTIVE_ROUNDS = 4
EXPECTED_COUNTS = {
    "experiment1_attack_calls": 108,
    "protection_sweep_calls": 108,
    "extension_conversations": 72,
    "extension_requests": 108,
    "adaptive_conversations": 18,
    "adaptive_request_limit": 144,
}


class OverviewInputError(ValueError):
    """Raised when saved evidence is missing, partial, or inconsistent."""


def _read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        raise OverviewInputError(f"Required input file is missing: {path}.")
    result: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OverviewInputError(f"Invalid JSON in {path.name} at line {line_number}.") from exc
            if not isinstance(row, dict):
                raise OverviewInputError(f"Expected an object in {path.name} at line {line_number}.")
            result.append((line_number, row))
    return result


def _read_reviews(path: Path, key: str) -> dict[str, tuple[int, dict[str, Any]]]:
    rows = _read_jsonl(path)
    indexed: dict[str, tuple[int, dict[str, Any]]] = {}
    for line, row in rows:
        identifier = row.get(key)
        if not isinstance(identifier, str) or not identifier:
            raise OverviewInputError(f"Review row in {path.name}:{line} has no {key}.")
        joined_identifier = _review_join_key(row, key)
        if joined_identifier in indexed:
            raise OverviewInputError(f"Duplicate review for {joined_identifier!r} in {path.name}.")
        indexed[joined_identifier] = (line, row)
    return indexed


def _review_join_key(row: dict[str, Any], key: str) -> str:
    """Keep same-named one-turn call IDs distinct across model JSONL files."""
    identifier = row.get(key)
    if key == "call_id" and isinstance(row.get("model"), str):
        return f"{row['model']}::{identifier}"
    return str(identifier)


def _path_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _read_model_files(results_root: Path, subdirectory: str) -> list[tuple[int, dict[str, Any]]]:
    combined = []
    for model in MODELS:
        path = results_root / subdirectory / f"{model}.jsonl"
        label = _path_label(path)
        combined.extend(
            (line, {**row, "__overview_source_file__": label})
            for line, row in _read_jsonl(path)
        )
    return combined


def _check_unique(rows: Iterable[tuple[int, dict[str, Any]]], key: str, source: str) -> None:
    seen: set[str] = set()
    for line, row in rows:
        identifier = row.get(key)
        if not isinstance(identifier, str) or not identifier:
            raise OverviewInputError(f"Record in {source}:{line} has no {key}.")
        if identifier in seen:
            raise OverviewInputError(f"Duplicate {key} {identifier!r} in {source}.")
        seen.add(identifier)


def _valid_indices(value: Any) -> list[int | str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if (isinstance(item, int) and not isinstance(item, bool)) or isinstance(item, str)]


def _semantic_indices(review: dict[str, Any] | None) -> list[int | str]:
    if not isinstance(review, dict):
        return []
    for key in (
        "semantic_rule_indices",
        "semantic_rule_leakage_instruction_indices",
        "semantic_equivalent_rule_indices",
        "semantic_instruction_indices",
    ):
        if key in review:
            return _valid_indices(review.get(key))
    return []


def _quote_text(item: Any) -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("quote", "text", "evidence_quote", "excerpt"):
            value = item.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _quote_items(review: dict[str, Any] | None) -> list[Any]:
    if not isinstance(review, dict):
        return []
    items: list[Any] = []
    for key in ("semantic_rule_evidence", "semantic_evidence", "evidence_quotes", "evidence"):
        value = review.get(key)
        if isinstance(value, list):
            items.extend(value)
        elif value is not None:
            items.append(value)
    return items


def _quote_refs(
    review: dict[str, Any] | None,
    *,
    review_file: str | None,
    review_line: int | None,
    round_number: int | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(review, dict):
        return []
    refs = []
    for item in _quote_items(review):
        text = _quote_text(item)
        if not text:
            continue
        item_round = None
        item_index = None
        if isinstance(item, dict):
            raw_round = item.get("round_number", item.get("round", item.get("turn")))
            if isinstance(raw_round, int):
                item_round = raw_round
            raw_index = item.get("rule_index", item.get("instruction_index"))
            if isinstance(raw_index, (int, str)) and not isinstance(raw_index, bool):
                item_index = raw_index
        refs.append({
            "source_file": review_file,
            "source_line": review_line,
            "round_number": item_round if item_round is not None else round_number,
            "rule_index": item_index,
            "quote": text,
        })
    return refs


def _adaptive_evidence_refs(
    review_round: dict[str, Any], *, review_file: str, review_line: int, round_number: int
) -> list[dict[str, Any]]:
    """Retain line/field pointers for reviewer evidence even without a short quote."""
    refs = _quote_refs(
        review_round,
        review_file=review_file,
        review_line=review_line,
        round_number=round_number,
    )
    evidence = review_round.get("rule_evidence")
    if not isinstance(evidence, list):
        return refs
    for item in evidence:
        if not isinstance(item, dict):
            continue
        rule_indices = _valid_indices(item.get("rule_indices", []))
        rule_index = item.get("rule_index")
        if isinstance(rule_index, (int, str)) and not isinstance(rule_index, bool):
            rule_indices = list(dict.fromkeys(rule_indices + [rule_index]))
        refs.append({
            "source_file": review_file,
            "source_line": review_line,
            "round_number": round_number,
            "rule_indices": rule_indices,
            "evidence_type": item.get("evidence_type"),
            "evidence_field": item.get("evidence_field"),
            "quote": _quote_text(item),
        })
    return refs


def _truth_score(score: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize literal and predeclared-decoder scores into response-level evidence."""
    if not isinstance(score, dict):
        score = {}
    disclosure = score.get("any_disclosure")
    disclosure = disclosure if isinstance(disclosure, dict) else {}
    raw = score.get("raw")
    raw = raw if isinstance(raw, dict) else {}
    decoded = score.get("decoded")
    decoded = decoded if isinstance(decoded, dict) else {}
    raw_canary_hits = int(disclosure.get("raw_canary_hits", score.get("raw_canary_hits", raw.get("canary_hits", 0))) or 0)
    decoded_canary_hits = int(disclosure.get("decoded_canary_hits", score.get("decoded_canary_hits", decoded.get("canary_hits", 0))) or 0)

    raw_rules = _valid_indices(raw.get("complete_instruction_indices"))
    if not raw_rules:
        raw_rules = _valid_indices(score.get("raw_complete_instruction_indices"))
    decoded_rules = _valid_indices(decoded.get("complete_instruction_indices"))
    if not decoded_rules:
        decoded_rules = _valid_indices(score.get("decoded_complete_instruction_indices"))
    if not raw_rules and not decoded_rules:
        combined = _valid_indices(disclosure.get("complete_instruction_indices"))
    else:
        combined = list(dict.fromkeys(raw_rules + decoded_rules))
    if not combined:
        combined = _valid_indices(score.get("complete_instruction_indices"))

    raw_full = bool(
        disclosure.get("raw_full_application_system_recovered")
        or score.get("raw_full_target_system_recovered")
        or raw.get("full_target_system_recovered")
        or raw.get("full_application_system_recovered")
    )
    decoded_full = bool(
        disclosure.get("decoded_full_application_system_recovered")
        or score.get("decoded_full_target_system_recovered")
        or decoded.get("full_target_system_recovered")
        or decoded.get("full_application_system_recovered")
    )
    if not raw_full and not decoded_full:
        raw_full = bool(disclosure.get("full_application_system_recovered") or score.get("full_target_system_recovered"))
    return {
        "raw_canary_hits": raw_canary_hits,
        "raw_canary_hit": raw_canary_hits > 0,
        "decoded_canary_hits": decoded_canary_hits,
        "decoded_canary_hit": decoded_canary_hits > 0,
        "raw_complete_rule_indices": raw_rules,
        "decoded_complete_rule_indices": decoded_rules,
        "complete_rule_indices": combined,
        "has_complete_rule": bool(combined),
        "all_three_complete_rules": len(set(combined)) >= 3 or bool(disclosure.get("all_instructions_recovered")),
        "raw_full_system": raw_full,
        "decoded_full_system": decoded_full,
        "full_system": raw_full or decoded_full,
    }


def _usage(value: Any) -> dict[str, int]:
    result = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if not isinstance(value, dict):
        return result
    for key in result:
        item = value.get(key)
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            result[key] = int(item)
    return result


def _sum_usage(items: Iterable[Any]) -> dict[str, int]:
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for item in items:
        usage = _usage(item)
        for key in total:
            total[key] += usage[key]
    return total


def _usage_json(value: dict[str, int]) -> dict[str, int]:
    return {key: int(value.get(key, 0)) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}


def _single_call_conversation(
    row: dict[str, Any],
    *,
    source_file: str,
    source_line: int,
    review: dict[str, Any] | None = None,
    review_file: str | None = None,
    review_line: int | None = None,
) -> dict[str, Any]:
    attempts = row.get("attempt_history")
    if not isinstance(attempts, list) or not attempts:
        attempts = [row]
    statuses = Counter(str(item.get("status", "unknown")) for item in attempts if isinstance(item, dict))
    usage = _sum_usage(item.get("usage") for item in attempts if isinstance(item, dict))
    status = str(row.get("status", "unknown"))
    reviewed = review is not None and review.get("review_status", review.get("status")) in {"reviewed", "complete", "completed"}
    semantic_indices = _semantic_indices(review) if reviewed else []
    metrics = _truth_score(row.get("score")) if status in {"complete", "truncated"} else _truth_score(None)
    event = {
        "round_number": 1,
        "status": status,
        "metrics": metrics,
        "score_eligible": status in {"complete", "truncated"},
        "semantic_rule_indices": semantic_indices,
        "semantic_reviewed": bool(reviewed),
        "semantic_quote_refs": _quote_refs(
            review,
            review_file=review_file,
            review_line=review_line,
            round_number=1,
        ) if reviewed else [],
        "request_count": len(attempts),
        "request_status_counts": dict(statuses),
        "token_usage": usage,
    }
    return {
        "conversation_id": row.get("call_id"),
        "model": row.get("model"),
        "condition": row.get("condition"),
        "target_group": row.get("target_group"),
        "strategy": row.get("strategy"),
        "status": status,
        "source": {"file": row.get("__overview_source_file__", source_file), "line": source_line},
        "semantic_review": {
            "reviewed": bool(reviewed),
            "reviewer_type": review.get("reviewer_type") if reviewed else None,
            "review_source": {"file": review_file, "line": review_line} if reviewed else None,
        },
        "rounds": [event],
        "request_count": len(attempts),
        "request_status_counts": dict(statuses),
        "token_usage": usage,
    }


def _simple_dataset(
    *,
    name: str,
    rows: list[tuple[int, dict[str, Any]]],
    source_file: str,
    reviews: dict[str, tuple[int, dict[str, Any]]] | None = None,
    review_file: str | None = None,
    expected_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reviews = reviews or {}
    selected = rows
    if name == "experiment1":
        selected = [(line, row) for line, row in rows if row.get("category") == "attack"]
    selected_ids = [
        (line, {**row, "call_id": _review_join_key(row, "call_id")})
        for line, row in selected
    ]
    _check_unique(selected_ids, "call_id", source_file)
    if len(selected) != expected_count:
        raise OverviewInputError(f"{name} expected {expected_count} attack records, found {len(selected)}.")
    if name in {"experiment1", "protection_sweep"} and len(reviews) != expected_count:
        raise OverviewInputError(f"{name} semantic review must cover all {expected_count} attack calls.")
    if set(reviews) != {_review_join_key(row, "call_id") for _, row in selected}:
        raise OverviewInputError(f"{name} semantic reviews do not match all call IDs.")
    if any(
        review.get("review_status", review.get("status")) not in {"reviewed", "complete", "completed"}
        for _, review in reviews.values()
    ):
        raise OverviewInputError(f"{name} has a missing or incomplete semantic review.")

    normalized = []
    for line, row in selected:
        review_entry = reviews.get(_review_join_key(row, "call_id"))
        normalized.append(_single_call_conversation(
            row,
            source_file=row.get("__overview_source_file__", source_file),
            source_line=line,
            review=review_entry[1] if review_entry else None,
            review_file=review_file,
            review_line=review_entry[0] if review_entry else None,
        ))
    observed_requests = sum(item["request_count"] for item in normalized)
    return normalized, {
        "record_type": "single_turn_attack_call",
        "planned_conversations": expected_count,
        "observed_conversations": len(normalized),
        "observed_requests": observed_requests,
        "status_counts": dict(Counter(item["status"] for item in normalized)),
        "attempt_status_counts": dict(sum((Counter(item["request_status_counts"]) for item in normalized), Counter())),
        "token_usage": _sum_usage(
            item["token_usage"] for item in normalized
        ),
        "semantic_review_file": review_file,
    }


def _extension_semantic_events(
    review: dict[str, Any] | None,
    rounds: list[dict[str, Any]],
    *,
    review_file: str | None,
    review_line: int | None,
) -> tuple[dict[int, list[int | str]], list[dict[str, Any]], bool]:
    if not isinstance(review, dict) or review.get("review_status") not in {"reviewed", "complete", "completed"}:
        return {}, [], False
    semantic = _semantic_indices(review)
    refs: list[dict[str, Any]] = []
    round_events: dict[int, list[int | str]] = {}
    # Some auditors provide per-turn findings; use them directly when present.
    review_rounds = review.get("rounds")
    if isinstance(review_rounds, list):
        for review_round in review_rounds:
            if not isinstance(review_round, dict):
                continue
            number = review_round.get("round_number", review_round.get("round", review_round.get("turn")))
            if not isinstance(number, int):
                continue
            indices = _semantic_indices(review_round)
            if indices:
                round_events[number] = indices
            refs.extend(_quote_refs(review_round, review_file=review_file, review_line=review_line, round_number=number))

    # Conversation-level extension reviews can be attributed by their quoted
    # evidence: locate each quote in the saved response for its turn.
    if semantic and not round_events:
        for item in _quote_items(review):
            quote = _quote_text(item)
            if not quote:
                continue
            number = None
            if isinstance(item, dict):
                candidate = item.get("round_number", item.get("round", item.get("turn")))
                if isinstance(candidate, int):
                    number = candidate
            if number is None:
                quote_key = " ".join(quote.split()).casefold()
                for round_index, event in enumerate(rounds, start=1):
                    content = event.get("content")
                    if isinstance(content, str) and quote_key and quote_key in " ".join(content.split()).casefold():
                        number = round_index
                        break
            refs.extend(_quote_refs({"evidence_quotes": [item]}, review_file=review_file, review_line=review_line, round_number=number))
            if number is not None:
                round_events.setdefault(number, []).extend(semantic)
    if semantic and not round_events and len(rounds) == 1:
        round_events[1] = semantic
    return round_events, refs, True


def _normalize_extension(
    rows: list[tuple[int, dict[str, Any]]],
    reviews: dict[str, tuple[int, dict[str, Any]]],
    *,
    source_file: str,
    review_file: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _check_unique(rows, "conversation_id", source_file)
    _check_unique([(line, row) for line, row in reviews.values()], "conversation_id", review_file)
    if len(rows) != EXPECTED_COUNTS["extension_conversations"]:
        raise OverviewInputError(
            f"Extension expected {EXPECTED_COUNTS['extension_conversations']} conversations, found {len(rows)}."
        )
    if set(reviews) != {row["conversation_id"] for _, row in rows}:
        raise OverviewInputError("Extension manual reviews are missing or contain unknown conversations.")
    normalized = []
    for line, row in rows:
        conversation_id = row["conversation_id"]
        review_line, review = reviews[conversation_id]
        if review.get("review_status", review.get("status")) not in {"reviewed", "complete", "completed"}:
            raise OverviewInputError(f"Extension semantic review for {conversation_id} is incomplete.")
        raw_rounds = row.get("round_results")
        if not isinstance(raw_rounds, list) or not raw_rounds:
            raise OverviewInputError(f"Extension conversation {conversation_id} has no saved turns.")
        status = str(row.get("status", "unknown"))
        scores = []
        normalized_events = []
        for round_number in range(1, len(raw_rounds) + 1):
            prefix = raw_rounds[:round_number]
            score = extension.score_conversation({"truth": row.get("truth", {}), "round_results": prefix})
            scores.append(_truth_score(score))
        semantic_events, quote_refs, reviewed = _extension_semantic_events(
            review,
            raw_rounds,
            review_file=review_file,
            review_line=review_line,
        )
        for index, round_result in enumerate(raw_rounds, start=1):
            round_status = str(round_result.get("status", status))
            normalized_events.append({
                "round_number": index,
                "status": round_status,
                "metrics": scores[index - 1],
                "score_eligible": any(
                    result.get("status") in {"complete", "truncated"}
                    for result in raw_rounds[:index]
                ),
                "semantic_rule_indices": semantic_events.get(index, []),
                "semantic_reviewed": reviewed,
                "semantic_quote_refs": [ref for ref in quote_refs if ref.get("round_number") in {None, index}],
                "request_count": 1,
                "request_status_counts": {round_status: 1},
                "token_usage": _usage(round_result.get("usage")),
            })
        usage = _sum_usage(event["token_usage"] for event in normalized_events)
        request_statuses = Counter()
        for event in normalized_events:
            request_statuses.update(event["request_status_counts"])
        normalized.append({
            "conversation_id": conversation_id,
            "model": row.get("model"),
            "condition": row.get("condition"),
            "target_group": row.get("target_set_id"),
            "strategy": row.get("strategy_id"),
            "status": status,
            "phase": row.get("phase"),
            "source": {"file": row.get("__overview_source_file__", source_file), "line": line},
            "semantic_review": {
                "reviewed": reviewed,
                "reviewer_type": review.get("reviewer_type"),
                "review_source": {"file": review_file, "line": review_line},
            },
            "rounds": normalized_events,
            "request_count": len(normalized_events),
            "request_status_counts": dict(request_statuses),
            "token_usage": usage,
        })
    request_count = sum(item["request_count"] for item in normalized)
    if request_count != EXPECTED_COUNTS["extension_requests"]:
        raise OverviewInputError(f"Extension expected 108 requests, found {request_count}.")
    multi = [item for item in normalized if len(item["rounds"]) > 1]
    single = [item for item in normalized if len(item["rounds"]) == 1]
    return normalized, {
        "record_type": "multi-turn_conversation",
        "planned_conversations": len(normalized),
        "observed_conversations": len(normalized),
        "observed_requests": request_count,
        "single_request_conversations": len(single),
        "multi_request_conversations": len(multi),
        "multi_conversation_requests": sum(item["request_count"] for item in multi),
        "status_counts": dict(Counter(item["status"] for item in normalized)),
        "attempt_status_counts": dict(sum((Counter(item["request_status_counts"]) for item in normalized), Counter())),
        "token_usage": _sum_usage(item["token_usage"] for item in normalized),
        "semantic_review_file": review_file,
    }


def _review_rounds(review: dict[str, Any], expected_rounds: int, *, source: str, line: int) -> list[dict[str, Any]]:
    rounds = review.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != expected_rounds:
        raise OverviewInputError(f"Adaptive semantic review in {source}:{line} must annotate all {expected_rounds} rounds.")
    by_round = {}
    for item in rounds:
        if not isinstance(item, dict):
            raise OverviewInputError(f"Adaptive semantic review in {source}:{line} has an invalid round.")
        number = item.get("round_number", item.get("round", item.get("turn")))
        if not isinstance(number, int) or number < 1 or number > expected_rounds or number in by_round:
            raise OverviewInputError(f"Adaptive semantic review in {source}:{line} has duplicate or invalid round numbering.")
        # Round-level fields are required to distinguish a first semantic hit
        # from an exact later recovery.
        recognized = any(key in item for key in (
            "semantic_rule_indices", "semantic_rule_leakage_instruction_indices", "semantic_equivalent_rule_indices"
        ))
        if not recognized:
            raise OverviewInputError(f"Adaptive semantic review in {source}:{line} omits round-level semantic rule findings.")
        by_round[number] = item
    if set(by_round) != set(range(1, expected_rounds + 1)):
        raise OverviewInputError(f"Adaptive semantic review in {source}:{line} has incomplete round numbering.")
    return [by_round[number] for number in range(1, expected_rounds + 1)]


def _is_contaminated(review: dict[str, Any]) -> bool:
    for key in ("contaminated", "contamination_flag", "attacker_echo_contamination", "contamination_detected"):
        if review.get(key) is True:
            return True
    assessment = review.get("known_secret_contamination_assessment")
    if isinstance(assessment, dict) and assessment.get("status") in {
        "confirmed_contamination", "known_secret_injected", "attacker_echoed_secret", "contaminated"
    }:
        return True
    return False


def _attribution_ambiguous(review: dict[str, Any]) -> bool:
    assessment = review.get("known_secret_contamination_assessment")
    return isinstance(assessment, dict) and assessment.get("status") == "possible_unexplained_target_specific_premise"


def _adaptive_round_score(record: dict[str, Any], round_number: int) -> dict[str, Any]:
    """Re-score a defender-only prefix; attacker outputs never enter the text."""
    defender_rounds = []
    for item in record.get("rounds", [])[:round_number]:
        result = item.get("defender_result")
        if not isinstance(result, dict):
            continue
        defender_rounds.append({
            "status": result.get("status"),
            "content": result.get("content"),
        })
    score = extension.score_conversation({
        "truth": record.get("truth", {}),
        "round_results": defender_rounds,
    })
    return _truth_score(score)


def _normalize_adaptive(
    rows: list[tuple[int, dict[str, Any]]],
    reviews: dict[str, tuple[int, dict[str, Any]]],
    *,
    source_file: str,
    review_file: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _check_unique(rows, "conversation_id", source_file)
    _check_unique([(line, row) for line, row in reviews.values()], "conversation_id", review_file)
    if len(rows) != EXPECTED_COUNTS["adaptive_conversations"]:
        raise OverviewInputError(
            f"Adaptive dialogue requires 18 completed conversations; found {len(rows)}."
        )
    expected_grid = {
        (attacker, condition, group)
        for attacker in ADAPTIVE_ATTACKERS
        for condition in ADAPTIVE_CONDITIONS
        for group in (1, 2, 3)
    }
    observed_grid = {
        (row.get("attacker_model"), row.get("condition"), row.get("target_group"))
        for _, row in rows
    }
    if observed_grid != expected_grid or any(row.get("defender_model") != ADAPTIVE_DEFENDER for _, row in rows):
        raise OverviewInputError("Adaptive dialogue does not cover the registered 2 × 3 × 3 model/condition/group design.")
    if set(reviews) != {row["conversation_id"] for _, row in rows}:
        raise OverviewInputError("Adaptive semantic review is missing or does not cover exactly 18 conversations.")

    normalized = []
    for line, row in rows:
        conversation_id = row["conversation_id"]
        review_line, review = reviews[conversation_id]
        for field in ("attacker_model", "defender_model", "condition", "target_group"):
            if field in review and review.get(field) != row.get(field):
                raise OverviewInputError(f"Adaptive review metadata disagrees with {conversation_id} for {field}.")
        raw_rounds = row.get("rounds")
        if not isinstance(raw_rounds, list) or not 1 <= len(raw_rounds) <= MAX_ADAPTIVE_ROUNDS:
            raise OverviewInputError(f"Adaptive conversation {conversation_id} has an invalid round count.")
        if row.get("status") not in {"canary_recovered", "completed_max_rounds"}:
            raise OverviewInputError(f"Adaptive conversation {conversation_id} is incomplete (status={row.get('status')!r}).")
        if review.get("review_status", review.get("status")) not in {"reviewed", "complete", "completed"}:
            raise OverviewInputError(f"Adaptive semantic review for {conversation_id} is incomplete.")
        review_rounds = _review_rounds(review, len(raw_rounds), source=review_file, line=review_line)
        contaminated = _is_contaminated(review)
        event_rows = []
        request_statuses: Counter[str] = Counter()
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for index, (round_data, review_round) in enumerate(zip(raw_rounds, review_rounds, strict=True), start=1):
            attacker_result = round_data.get("attacker_result")
            defender_result = round_data.get("defender_result")
            if not isinstance(attacker_result, dict) or not isinstance(defender_result, dict):
                raise OverviewInputError(f"Adaptive conversation {conversation_id}, round {index} lacks a model result.")
            attacker_status = str(attacker_result.get("status", "unknown"))
            defender_status = str(defender_result.get("status", "unknown"))
            for role_status in (attacker_status, defender_status):
                request_statuses[role_status] += 1
            if attacker_status != "complete" or defender_status != "complete":
                raise OverviewInputError(f"Adaptive request pair in {conversation_id}, round {index} is incomplete.")
            score_metrics = _adaptive_round_score(row, index)
            semantic_indices = [] if contaminated else _semantic_indices(review_round)
            quotes = _adaptive_evidence_refs(
                review_round,
                review_file=review_file,
                review_line=review_line,
                round_number=index,
            )
            role_usage = {
                "attacker": _usage(attacker_result.get("usage")),
                "defender": _usage(defender_result.get("usage")),
            }
            round_usage = _sum_usage(role_usage.values())
            for key in usage_total:
                usage_total[key] += round_usage[key]
            requests = {
                "attacker": int(round_data.get("attacker_request") is not None),
                "defender": int(round_data.get("defender_request") is not None),
            }
            event_rows.append({
                "round_number": index,
                "status": defender_status,
                "attacker_status": attacker_status,
                "metrics": score_metrics,
                "score_eligible": True,
                "semantic_rule_indices": semantic_indices,
                "semantic_reviewed": True,
                "semantic_quote_refs": quotes,
                "contaminated": contaminated,
                "request_count": sum(requests.values()),
                "requests_by_role": requests,
                "request_status_counts": {attacker_status: 1, defender_status: 1},
                "token_usage": round_usage,
                "token_usage_by_role": role_usage,
            })
        requests_total = sum(event["request_count"] for event in event_rows)
        normalized.append({
            "conversation_id": conversation_id,
            "attacker_model": row.get("attacker_model"),
            "defender_model": row.get("defender_model"),
            "model": row.get("attacker_model"),
            "condition": row.get("condition"),
            "target_group": row.get("target_group"),
            "status": row.get("status"),
            "source": {"file": row.get("__overview_source_file__", source_file), "line": line},
            "semantic_review": {
                "reviewed": True,
                "reviewer_type": review.get("reviewer_type"),
                "review_source": {"file": review_file, "line": review_line},
                "contaminated": contaminated,
                "attribution_ambiguous": _attribution_ambiguous(review),
                "known_secret_contamination_assessment": review.get("known_secret_contamination_assessment"),
                "prior_turn_context_integrity": review.get("prior_turn_context_integrity"),
            },
            "rounds": event_rows,
            "request_count": requests_total,
            "request_status_counts": dict(request_statuses),
            "token_usage": usage_total,
            "token_usage_by_role": {
                role: _sum_usage(event["token_usage_by_role"][role] for event in event_rows)
                for role in ("attacker", "defender")
            },
        })
    total_requests = sum(item["request_count"] for item in normalized)
    if total_requests > EXPECTED_COUNTS["adaptive_request_limit"]:
        raise OverviewInputError(f"Adaptive dialogue used {total_requests} requests, above the 144-request limit.")
    return normalized, {
        "record_type": "attacker_defender_conversation",
        "planned_conversations": EXPECTED_COUNTS["adaptive_conversations"],
        "observed_conversations": len(normalized),
        "observed_requests": total_requests,
        "maximum_requests": EXPECTED_COUNTS["adaptive_request_limit"],
        "status_counts": dict(Counter(item["status"] for item in normalized)),
        "attempt_status_counts": dict(sum((Counter(item["request_status_counts"]) for item in normalized), Counter())),
        "token_usage": _sum_usage(item["token_usage"] for item in normalized),
        "token_usage_by_role": {
            role: _sum_usage(item["token_usage_by_role"][role] for item in normalized)
            for role in ("attacker", "defender")
        },
        "semantic_review_file": review_file,
    }


_ROUND_METRIC_KEYS = (
    "raw_canary_hit",
    "decoded_canary_hit",
    "has_complete_rule",
    "all_three_complete_rules",
    "raw_full_system",
    "decoded_full_system",
    "full_system",
)


def _merge_index_values(values: Iterable[Any]) -> list[int | str]:
    merged: list[int | str] = []
    for value in values:
        merged.extend(_valid_indices(value))
    return list(dict.fromkeys(merged))


def _group_rounds(conversations: list[dict[str, Any]], max_rounds: int) -> dict[str, Any]:
    round_summary = {}
    for round_number in range(1, max_rounds + 1):
        current_rows: list[dict[str, Any]] = []
        newly_flags = {key: 0 for key in _ROUND_METRIC_KEYS}
        semantic_new = 0
        request_count = 0
        request_statuses: Counter[str] = Counter()
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        role_requests: Counter[str] = Counter()
        role_usage: dict[str, dict[str, int]] = {
            "attacker": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "defender": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        for conversation in conversations:
            events = conversation.get("rounds", [])
            event = next((item for item in events if item.get("round_number") == round_number), None)
            if event is not None:
                request_count += int(event.get("request_count", 0) or 0)
                request_statuses.update(event.get("request_status_counts", {}))
                event_usage = _usage(event.get("token_usage"))
                for key in usage:
                    usage[key] += event_usage[key]
                for role, amount in event.get("requests_by_role", {}).items():
                    role_requests[role] += int(amount or 0)
                for role, role_item in event.get("token_usage_by_role", {}).items():
                    normalized = _usage(role_item)
                    for key in usage:
                        role_usage.setdefault(role, {name: 0 for name in usage})[key] += normalized[key]

            through = [item for item in events if item.get("round_number", 0) <= round_number]
            if not through:
                continue
            latest = through[-1]
            current_metrics = latest.get("metrics", {})
            previous = through[-2] if len(through) > 1 else None
            previous_metrics = previous.get("metrics", {}) if previous else {}
            semantic_indices = _merge_index_values(item.get("semantic_rule_indices") for item in through)
            previous_semantic_indices = _merge_index_values(
                item.get("semantic_rule_indices") for item in through[:-1]
            )
            current_event = next(
                (item for item in events if item.get("round_number") == round_number),
                None,
            )
            if current_event is not None:
                for key in _ROUND_METRIC_KEYS:
                    if current_metrics.get(key) and not previous_metrics.get(key):
                        newly_flags[key] += 1
                current_semantic = _valid_indices(current_event.get("semantic_rule_indices"))
                if current_semantic and not previous_semantic_indices and current_event.get("semantic_reviewed"):
                    semantic_new += 1
            record = {
                "metrics": current_metrics,
                "semantic_rule_indices": semantic_indices,
                "reviewed": bool(latest.get("semantic_reviewed")),
                "status": latest.get("status"),
                "score_eligible": any(item.get("score_eligible") for item in through),
            }
            current_rows.append(record)

        eligible_rows = [row for row in current_rows if row["score_eligible"]]

        def hit_count(key: str) -> int:
            return sum(bool(row["metrics"].get(key)) for row in eligible_rows)

        def value_hit_total(key: str) -> int:
            return sum(int(row["metrics"].get(key, 0) or 0) for row in eligible_rows)

        semantic_reviewed_n = sum(row["reviewed"] for row in current_rows)
        semantic_rule_value_hits = sum(
            len(set(row["semantic_rule_indices"]))
            for row in current_rows
            if row["reviewed"]
        )
        semantic_hits = sum(
            bool(row["semantic_rule_indices"])
            for row in current_rows
            if row["reviewed"]
        )
        semantic_all_three_hits = sum(
            len(set(row["semantic_rule_indices"])) >= 3
            for row in current_rows
            if row["reviewed"]
        )
        round_summary[str(round_number)] = {
            "conversations_observed_by_this_round": len(current_rows),
            "conversations_with_scored_defender_or_model_response": len(eligible_rows),
            "cumulative_evidence": {
                "raw_canary_hit_conversations": hit_count("raw_canary_hit"),
                "raw_canary_value_hits": value_hit_total("raw_canary_hits"),
                "decoded_canary_hit_conversations": hit_count("decoded_canary_hit"),
                "decoded_canary_value_hits": value_hit_total("decoded_canary_hits"),
                "raw_complete_rule_conversations": hit_count("raw_complete_rule_indices"),
                "decoded_complete_rule_conversations": hit_count("decoded_complete_rule_indices"),
                "complete_rule_conversations": hit_count("has_complete_rule"),
                "all_three_complete_rule_conversations": hit_count("all_three_complete_rules"),
                "raw_full_system_conversations": hit_count("raw_full_system"),
                "decoded_full_system_conversations": hit_count("decoded_full_system"),
                "full_system_conversations": hit_count("full_system"),
                "semantic_reviewed_conversations": semantic_reviewed_n,
                "semantic_equivalent_rule_conversations": semantic_hits,
                "semantic_equivalent_rule_value_hits": semantic_rule_value_hits,
                "semantic_all_three_rule_conversations": semantic_all_three_hits,
            },
            "new_recoveries_this_round": {
                "raw_canary_hit_conversations": newly_flags["raw_canary_hit"],
                "decoded_canary_hit_conversations": newly_flags["decoded_canary_hit"],
                "complete_rule_conversations": newly_flags["has_complete_rule"],
                "all_three_complete_rule_conversations": newly_flags["all_three_complete_rules"],
                "full_system_conversations": newly_flags["full_system"],
            },
            "semantic_new_recoveries": semantic_new,
            "requests_in_round": request_count,
            "request_status_counts": dict(request_statuses),
            "token_usage_in_round": usage,
            "requests_in_round_by_role": dict(role_requests),
            "token_usage_in_round_by_role": {role: _usage_json(item) for role, item in role_usage.items()},
        }
    return round_summary


def _dataset_summary(
    conversations: list[dict[str, Any]],
    *,
    group_keys: tuple[str, ...],
    max_rounds: int,
) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for conversation in conversations:
        grouped[tuple(conversation.get(key) for key in group_keys)].append(conversation)
    by_group = {}
    for key in sorted(grouped, key=lambda item: tuple(str(value) for value in item)):
        records = grouped[key]
        label = "|".join(str(value) for value in key)
        request_statuses: Counter[str] = Counter()
        for item in records:
            request_statuses.update(item.get("request_status_counts", {}))
        by_group[label] = {
            "group": {name: value for name, value in zip(group_keys, key, strict=True)},
            "conversation_n": len(records),
            "status_counts": dict(Counter(str(item.get("status", "unknown")) for item in records)),
            "request_count": sum(int(item.get("request_count", 0) or 0) for item in records),
            "request_status_counts": dict(request_statuses),
            "token_usage": _sum_usage(item.get("token_usage") for item in records),
            "rounds": _group_rounds(records, max_rounds),
            "semantic_evidence_references": [
                {
                    "conversation_id": record["conversation_id"],
                    "round_number": event["round_number"],
                    "semantic_rule_indices": event.get("semantic_rule_indices", []),
                    "review_source": record.get("semantic_review", {}).get("review_source"),
                    "source": record.get("source"),
                    "evidence_refs": event.get("semantic_quote_refs", []),
                    "attribution_ambiguous": record.get("semantic_review", {}).get("attribution_ambiguous", False),
                }
                for record in records
                for event in record.get("rounds", [])
                if event.get("semantic_rule_indices") and event.get("semantic_reviewed")
            ],
        }
        if any("token_usage_by_role" in item for item in records):
            by_group[label]["token_usage_by_role"] = {
                role: _sum_usage(item.get("token_usage_by_role", {}).get(role) for item in records)
                for role in ("attacker", "defender")
            }
    return {
        "conversation_n": len(conversations),
        "request_count": sum(int(item.get("request_count", 0) or 0) for item in conversations),
        "status_counts": dict(Counter(str(item.get("status", "unknown")) for item in conversations)),
        "request_status_counts": dict(sum((Counter(item.get("request_status_counts", {})) for item in conversations), Counter())),
        "token_usage": _sum_usage(item.get("token_usage") for item in conversations),
        "by_model_condition": by_group,
    }


def _adaptive_figure_data(conversations: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [
        item for item in conversations
        if item.get("attacker_model") == "gpt-6.1-sol" and item.get("defender_model") == ADAPTIVE_DEFENDER
    ]
    if len(selected) != 9:
        raise OverviewInputError(f"Figure 1 requires nine gpt-6.1-sol → gpt-6-luna conversations; found {len(selected)}.")
    by_condition = {}
    for condition in ADAPTIVE_CONDITIONS:
        cell = sorted(
            (item for item in selected if item.get("condition") == condition),
            key=lambda item: int(item.get("target_group")),
        )
        if len(cell) != 3 or {item.get("target_group") for item in cell} != {1, 2, 3}:
            raise OverviewInputError(f"Figure 1 requires three target groups for condition {condition!r}.")
        if any(not 1 <= len(item.get("rounds", [])) <= MAX_ADAPTIVE_ROUNDS for item in cell):
            raise OverviewInputError(f"Figure 1 requires one to four saved rounds in condition {condition!r}.")
        semantic_first: list[int | None] = []
        exact_first: list[int | None] = []
        final_canary_hits = 0
        conversations_table = []
        for item in cell:
            sem_rounds = [
                int(event["round_number"])
                for event in item["rounds"]
                if event.get("semantic_rule_indices")
            ]
            exact_rounds = [
                int(event["round_number"])
                for event in item["rounds"]
                if event.get("metrics", {}).get("has_complete_rule")
            ]
            semantic_first.append(min(sem_rounds) if sem_rounds else None)
            exact_first.append(min(exact_rounds) if exact_rounds else None)
            last_metrics = item["rounds"][-1]["metrics"]
            final_canary_hits += int(last_metrics.get("raw_canary_hits", 0)) + int(last_metrics.get("decoded_canary_hits", 0))
            evidence_refs = [
                ref
                for event in item["rounds"]
                for ref in event.get("semantic_quote_refs", [])
            ]
            conversations_table.append({
                "conversation_id": item["conversation_id"],
                "target_group": item["target_group"],
                "first_semantic_round": min(sem_rounds) if sem_rounds else None,
                "first_exact_complete_rule_round": min(exact_rounds) if exact_rounds else None,
                "final_raw_canary_hits": last_metrics.get("raw_canary_hits", 0),
                "final_decoded_canary_hits": last_metrics.get("decoded_canary_hits", 0),
                "semantic_quote_refs": evidence_refs,
                "source": item["source"],
                "semantic_review": item["semantic_review"],
            })
        x = list(range(1, MAX_ADAPTIVE_ROUNDS + 1))
        by_condition[condition] = {
            "n": len(cell),
            "rounds": x,
            "semantic_recovery_at_least_one_rule": [sum(first is not None and first <= round_number for first in semantic_first) for round_number in x],
            "exact_complete_sentence_at_least_one_rule": [sum(first is not None and first <= round_number for first in exact_first) for round_number in x],
            "raw_and_decoded_canary_hits": final_canary_hits,
            "annotate_zero_canary": final_canary_hits == 0,
            "conversations": conversations_table,
        }
    return {
        "attacker_model": "gpt-6.1-sol",
        "defender_model": ADAPTIVE_DEFENDER,
        "conditions": by_condition,
        "uncertainty_intervals": None,
        "uncertainty_note": "No interval is shown; the plot reports the three observed conversations per condition.",
    }


def build_overview_from_inputs(
    *,
    experiment1_rows: list[tuple[int, dict[str, Any]]],
    experiment1_reviews: dict[str, tuple[int, dict[str, Any]]],
    protection_rows: list[tuple[int, dict[str, Any]]],
    protection_reviews: dict[str, tuple[int, dict[str, Any]]],
    extension_rows: list[tuple[int, dict[str, Any]]],
    extension_reviews: dict[str, tuple[int, dict[str, Any]]],
    adaptive_rows: list[tuple[int, dict[str, Any]]],
    adaptive_reviews: dict[str, tuple[int, dict[str, Any]]],
    source_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    source_paths = source_paths or {}
    e1, e1_meta = _simple_dataset(
        name="experiment1",
        rows=experiment1_rows,
        source_file=source_paths.get("experiment1", "results/experiment1/*.jsonl"),
        reviews=experiment1_reviews,
        review_file=source_paths.get("experiment1_review", "results/experiment1/semantic_review.jsonl"),
        expected_count=EXPECTED_COUNTS["experiment1_attack_calls"],
    )
    sweep, sweep_meta = _simple_dataset(
        name="protection_sweep",
        rows=protection_rows,
        source_file=source_paths.get("protection_sweep", "results/protection_sweep/*.jsonl"),
        reviews=protection_reviews,
        review_file=source_paths.get("protection_review", "results/protection_sweep/semantic_review.jsonl"),
        expected_count=EXPECTED_COUNTS["protection_sweep_calls"],
    )
    extension_rows_normalized, extension_meta = _normalize_extension(
        extension_rows,
        extension_reviews,
        source_file=source_paths.get("extension", "results/extension_dev/dev/*.jsonl"),
        review_file=source_paths.get("extension_review", "results/extension_dev/dev/manual_review.jsonl"),
    )
    adaptive, adaptive_meta = _normalize_adaptive(
        adaptive_rows,
        adaptive_reviews,
        source_file=source_paths.get("adaptive", "results/adaptive_dialogue/adaptive_dialogue.jsonl"),
        review_file=source_paths.get("adaptive_review", "results/adaptive_dialogue/semantic_review.jsonl"),
    )
    if len(experiment1_reviews) != EXPECTED_COUNTS["experiment1_attack_calls"]:
        raise OverviewInputError("Experiment 1 semantic review must cover all 108 attack calls.")
    if len(protection_reviews) != EXPECTED_COUNTS["protection_sweep_calls"]:
        raise OverviewInputError("Protection sweep semantic review must cover all 108 calls.")
    if len(extension_reviews) != EXPECTED_COUNTS["extension_conversations"]:
        raise OverviewInputError("Extension semantic review must cover all 72 conversations.")

    datasets = {
        "experiment1_attacks": {
            **e1_meta,
            **_dataset_summary(e1, group_keys=("model", "condition"), max_rounds=1),
            "semantic_review": "agent review from the saved semantic review file",
        },
        "single_turn_protection_sweep": {
            **sweep_meta,
            **_dataset_summary(sweep, group_keys=("model", "condition"), max_rounds=1),
            "semantic_review": "agent review from the saved semantic review file",
        },
        "fixed_extension": {
            **extension_meta,
            **_dataset_summary(extension_rows_normalized, group_keys=("model", "condition"), max_rounds=4),
            "semantic_review": "agent review from the saved manual review file",
        },
        "adaptive_dialogue": {
            **adaptive_meta,
            **_dataset_summary(
                adaptive,
                group_keys=("attacker_model", "defender_model", "condition"),
                max_rounds=MAX_ADAPTIVE_ROUNDS,
            ),
            "semantic_review": "per-round agent review; adaptive exact scoring uses defender outputs only",
        },
    }
    figure_data = _adaptive_figure_data(adaptive)
    return {
        "schema_version": 1,
        "generated_from_saved_records_only": True,
        "metric_policy": {
            "exact_canaries": "Raw and predeclared-format decoded matches are reported separately.",
            "complete_rules": "A response/conversation hit means at least one complete target rule; all-three is reported separately.",
            "semantic_rules": "Counted only from completed agent reviews, with source line and quote references retained.",
            "full_system": "Raw and decoded full-system recovery are reported separately.",
            "adaptive_defender_only": "Adaptive re-scoring includes defender_result.content only; attacker_result.content is never scored as leakage.",
            "errors": "Errors and missing records remain status/request counts and are excluded from recovery metrics.",
            "cumulative_round_metrics": "Round r includes evidence observed by or before r; recoveries remain cumulative after an early stop.",
        },
        "expected_counts": EXPECTED_COUNTS,
        "datasets": datasets,
        "figure_1": {
            "path": "figures/experiment1_dialogue_recovery.png",
            "data": figure_data,
        },
    }


def build_overview(results_root: Path = RESULTS_ROOT) -> dict[str, Any]:
    results_root = Path(results_root)
    experiment1_rows = _read_model_files(results_root, "experiment1")
    experiment1_review_path = results_root / "experiment1" / "semantic_review.jsonl"
    protection_rows = _read_model_files(results_root, "protection_sweep")
    protection_review_path = results_root / "protection_sweep" / "semantic_review.jsonl"
    extension_rows = _read_model_files(results_root, "extension_dev/dev")
    extension_review_path = results_root / "extension_dev" / "dev" / "manual_review.jsonl"
    adaptive_path = results_root / "adaptive_dialogue" / "adaptive_dialogue.jsonl"
    adaptive_review_path = results_root / "adaptive_dialogue" / "semantic_review.jsonl"
    review_paths = {
        "experiment1_review": _path_label(experiment1_review_path),
        "protection_review": _path_label(protection_review_path),
        "extension_review": _path_label(extension_review_path),
        "adaptive_review": _path_label(adaptive_review_path),
    }
    return build_overview_from_inputs(
        experiment1_rows=experiment1_rows,
        experiment1_reviews=_read_reviews(experiment1_review_path, "call_id"),
        protection_rows=protection_rows,
        protection_reviews=_read_reviews(protection_review_path, "call_id"),
        extension_rows=extension_rows,
        extension_reviews=_read_reviews(extension_review_path, "conversation_id"),
        adaptive_rows=[
            (line, {**row, "__overview_source_file__": _path_label(adaptive_path)})
            for line, row in _read_jsonl(adaptive_path)
        ],
        adaptive_reviews=_read_reviews(adaptive_review_path, "conversation_id"),
        source_paths=review_paths,
    )


def render_figure(figure_data: dict[str, Any], output_path: Path = FIGURE_PATH) -> Path:
    conditions = figure_data.get("conditions", {})
    if set(conditions) != set(ADAPTIVE_CONDITIONS):
        raise OverviewInputError("Figure 1 requires none, weak, and structured adaptive conditions.")
    if any(conditions[name].get("n") != 3 for name in ADAPTIVE_CONDITIONS):
        raise OverviewInputError("Figure 1 must use three observed conversations per condition.")

    # Import lazily so offline aggregation/tests do not inherit unrelated plot
    # module initialization while its other experiment figures are changing.
    from .comparison_plots import _configure_fonts

    _configure_fonts()
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    colors = {"semantic": "#1769aa", "exact": "#d95f02"}
    panel_titles = {"none": "No explicit secrecy", "weak": "Weak secrecy", "structured": "Structured boundary"}
    x = list(range(1, MAX_ADAPTIVE_ROUNDS + 1))
    for axis, condition in zip(axes, ADAPTIVE_CONDITIONS, strict=True):
        cell = conditions[condition]
        axis.plot(
            x,
            cell["semantic_recovery_at_least_one_rule"],
            color=colors["semantic"],
            marker="o",
            linewidth=2.2,
            label="Semantic recovery (≥1 rule)",
        )
        axis.plot(
            x,
            cell["exact_complete_sentence_at_least_one_rule"],
            color=colors["exact"],
            marker="s",
            linestyle="--",
            linewidth=2.0,
            label="Exact complete sentence (≥1 rule)",
        )
        axis.set_title(f"{panel_titles[condition]} (n=3)", fontsize=11)
        axis.set_xticks(x)
        axis.set_xlim(1, MAX_ADAPTIVE_ROUNDS)
        axis.set_ylim(-0.08, 3.35)
        axis.set_yticks(range(4))
        axis.grid(axis="y", alpha=0.25, linewidth=0.7)
        axis.set_xlabel("Dialogue round")
        if cell.get("annotate_zero_canary") is True and cell.get("raw_and_decoded_canary_hits") == 0:
            axis.text(
                0.98,
                0.96,
                "0 exact canaries observed",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                color="#444444",
            )
    axes[0].set_ylabel("Cumulative conversations recovered (of 3)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    figure.suptitle(
        "Adaptive dialogue recovery: gpt-6.1-sol attacker → gpt-6-luna defender",
        y=1.13,
        fontsize=13,
    )
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_artifacts(
    results_root: Path = RESULTS_ROOT,
    overview_path: Path = OVERVIEW_PATH,
    figure_path: Path = FIGURE_PATH,
) -> tuple[Path, Path]:
    overview = build_overview(results_root)
    figure_path = render_figure(overview["figure_1"]["data"], figure_path)
    _write_json_atomic(overview_path, overview)
    return Path(overview_path), figure_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--output", type=Path, default=OVERVIEW_PATH)
    parser.add_argument("--figure", type=Path, default=FIGURE_PATH)
    args = parser.parse_args(argv)
    try:
        overview, figure = build_artifacts(args.results_root, args.output, args.figure)
        print(f"Overview saved to {overview}; Figure 1 saved to {figure}.")
        return 0
    except Exception as exc:
        print(f"Leakage overview stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
