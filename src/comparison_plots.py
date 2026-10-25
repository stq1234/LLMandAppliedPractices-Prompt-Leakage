"""Create the experiment 2 learning and transfer figures."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT2_SUMMARY = ROOT / "results" / "experiment2" / "summary.json"
FIGURES_DIR = ROOT / "figures"
PLOT_DATA_PATH = ROOT / "results" / "comparison_plot_data.json"

MODEL_ORDER = ("gpt-6-luna", "gpt-5.6-luna", "gpt-6.1-sol")
STRATEGY_COLORS = {"active": "#2878B5", "random": "#E07B39"}
POLICY_ORDER = ("A", "B", "C")
SEED_ORDER = (17, 43)
BUDGET_ORDER = (0, 4, 8, 12, 16)


def _configure_fonts() -> str:
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in candidates if name in available), "DejaVu Sans")
    plt.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return selected


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


class IncompleteSummaryError(ValueError):
    """Raised when a summary cannot support the requested complete comparison."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required summary is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Summary root must be a JSON object: {path}")
    return value


def _require_ratio(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IncompleteSummaryError(f"{where} is missing a numeric ratio.")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise IncompleteSummaryError(f"{where} must be a finite ratio in [0, 1].")
    return result


def _expected_episode_keys() -> set[tuple[str, int, str]]:
    return {
        (policy, seed, strategy)
        for policy in POLICY_ORDER
        for seed in SEED_ORDER
        for strategy in STRATEGY_COLORS
    }


def validate_experiment2(summary: Mapping[str, Any]) -> None:
    models = summary.get("models")
    if not isinstance(models, dict) or not set(MODEL_ORDER).issubset(models):
        raise IncompleteSummaryError("Experiment 2 summary must contain all three requested models.")
    budget = summary.get("request_budget", {})
    if not isinstance(budget, dict) or budget.get("planned_total_requests") != 198:
        raise IncompleteSummaryError("Experiment 2 summary does not match the planned 198-request protocol.")
    expected_episode_keys = _expected_episode_keys()
    for model_name in MODEL_ORDER:
        model = models[model_name]
        if not isinstance(model.get("actual_requests"), int) or model["actual_requests"] < 66:
            raise IncompleteSummaryError(f"Experiment 2 is missing planned requests for {model_name}.")
        policies = model.get("policies")
        if not isinstance(policies, dict) or not set(POLICY_ORDER).issubset(policies):
            raise IncompleteSummaryError(f"Experiment 2 policy results are missing for {model_name}.")
        seen: set[tuple[str, int, str]] = set()
        for policy_id in POLICY_ORDER:
            policy = policies[policy_id]
            target = policy.get("target_holdout")
            direct = policy.get("direct_disclosure_probe")
            if not isinstance(target, dict) or target.get("status") != "complete" or target.get("holdout_count") != 28:
                raise IncompleteSummaryError(f"Experiment 2 target holdout is incomplete for {model_name}/{policy_id}.")
            _require_ratio(target.get("truth_accuracy"), f"target holdout accuracy {model_name}/{policy_id}")
            if not isinstance(direct, dict) or direct.get("status") != "complete":
                raise IncompleteSummaryError(f"Experiment 2 direct request is incomplete for {model_name}/{policy_id}.")
            episodes = policy.get("episodes")
            if not isinstance(episodes, list):
                raise IncompleteSummaryError(f"Experiment 2 episodes are missing for {model_name}/{policy_id}.")
            for episode in episodes:
                if not isinstance(episode, dict):
                    continue
                key = (episode.get("policy"), episode.get("seed"), episode.get("strategy"))
                if key[0] != policy_id:
                    raise IncompleteSummaryError(
                        f"Episode is stored under the wrong policy for {model_name}/{policy_id}: {key!r}."
                    )
                if key in seen:
                    raise IncompleteSummaryError(f"Duplicate experiment 2 episode {model_name}/{key!r}.")
                seen.add(key)
                if episode.get("status") != "complete":
                    raise IncompleteSummaryError(f"Experiment 2 episode is incomplete: {model_name}/{key!r}.")
                if episode.get("configured_query_budget") != 16 or episode.get("valid_labels_observed") != 16:
                    raise IncompleteSummaryError(f"Experiment 2 episode has an incomplete query budget: {model_name}/{key!r}.")
                if episode.get("queries_sent") != 16:
                    raise IncompleteSummaryError(f"Experiment 2 episode did not use exactly 16 queries: {model_name}/{key!r}.")
                checkpoints = episode.get("checkpoints")
                if not isinstance(checkpoints, list) or [item.get("budget") for item in checkpoints] != list(BUDGET_ORDER):
                    raise IncompleteSummaryError(f"Experiment 2 checkpoints are incomplete: {model_name}/{key!r}.")
                for checkpoint in checkpoints:
                    _require_ratio(
                        checkpoint.get("full_512_accuracy"),
                        f"full-domain accuracy {model_name}/{key!r}/{checkpoint.get('budget')}",
                    )
                final = checkpoints[-1]
                if final.get("unqueried_count") != 496:
                    raise IncompleteSummaryError(f"Expected 496 unqueried examples in {model_name}/{key!r}.")
                _require_ratio(final.get("unqueried_accuracy"), f"unqueried accuracy {model_name}/{key!r}")
                clone = episode.get("holdout_clone")
                if not isinstance(clone, dict) or clone.get("status") != "complete" or clone.get("holdout_count") != 28:
                    raise IncompleteSummaryError(f"Experiment 2 clone holdout is incomplete: {model_name}/{key!r}.")
                _require_ratio(clone.get("agreement_with_target"), f"clone-target agreement {model_name}/{key!r}")
        if seen != expected_episode_keys:
            missing = sorted(expected_episode_keys - seen)
            raise IncompleteSummaryError(f"Experiment 2 is missing episodes for {model_name}: {missing}.")


def _mean_min_max(values: Sequence[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "values": list(values),
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def aggregate_experiment2(summary: Mapping[str, Any]) -> dict[str, Any]:
    validate_experiment2(summary)
    learning: list[dict[str, Any]] = []
    transfer: list[dict[str, Any]] = []
    target_reference: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        model = summary["models"][model_name]
        episode_index = {
            (episode["policy"], episode["seed"], episode["strategy"]): episode
            for policy in POLICY_ORDER
            for episode in model["policies"][policy]["episodes"]
        }
        target_reference.extend({
            "model": model_name,
            "policy": policy_id,
            "holdout_count": model["policies"][policy_id]["target_holdout"]["holdout_count"],
            "truth_correct": model["policies"][policy_id]["target_holdout"]["truth_correct"],
            "truth_accuracy": model["policies"][policy_id]["target_holdout"]["truth_accuracy"],
        } for policy_id in POLICY_ORDER)
        for strategy in STRATEGY_COLORS:
            episodes = [
                episode_index[(policy, seed, strategy)]
                for policy in POLICY_ORDER
                for seed in SEED_ORDER
            ]
            per_budget = []
            for budget in BUDGET_ORDER:
                values = [
                    next(checkpoint["full_512_accuracy"] for checkpoint in episode["checkpoints"] if checkpoint["budget"] == budget)
                    for episode in episodes
                ]
                stats = _mean_min_max(values)
                per_budget.append({"budget": budget, **stats})
            learning.append({
                "model": model_name,
                "strategy": strategy,
                "episodes": [
                    {
                        "policy": episode["policy"],
                        "seed": episode["seed"],
                        "values": [
                            {"budget": checkpoint["budget"], "full_512_accuracy": checkpoint["full_512_accuracy"]}
                            for checkpoint in episode["checkpoints"]
                        ],
                    }
                    for episode in episodes
                ],
                "mean_curve": per_budget,
            })
            unqueried_values = [float(episode["final_unqueried_accuracy"]) for episode in episodes]
            agreement_values = [float(episode["holdout_clone"]["agreement_with_target"]) for episode in episodes]
            transfer.append({
                "model": model_name,
                "strategy": strategy,
                "episode_count": len(episodes),
                "unqueried_accuracy": {
                    "n_examples_per_episode": 496,
                    **_mean_min_max(unqueried_values),
                },
                "clone_target_agreement": {
                    "n_examples_per_episode": 28,
                    **_mean_min_max(agreement_values),
                },
                "episodes": [
                    {
                        "policy": episode["policy"],
                        "seed": episode["seed"],
                        "unqueried_accuracy": episode["final_unqueried_accuracy"],
                        "clone_target_agreement": episode["holdout_clone"]["agreement_with_target"],
                    }
                    for episode in episodes
                ],
            })
    return {
        "models": list(MODEL_ORDER),
        "strategies": list(STRATEGY_COLORS),
        "policies": list(POLICY_ORDER),
        "seeds": list(SEED_ORDER),
        "budgets": list(BUDGET_ORDER),
        "learning_curves": learning,
        "transfer": transfer,
        "target_holdout_truth_accuracy_reference": target_reference,
        "aggregation": "Mean over three policies and two fixed seeds (six episodes); min/max describe observed episode range, not confidence intervals.",
    }


def _learning_series(aggregate: Mapping[str, Any], model_name: str, strategy: str) -> Mapping[str, Any]:
    return next(
        item for item in aggregate["learning_curves"] if item["model"] == model_name and item["strategy"] == strategy
    )


def _plot_learning(aggregate: Mapping[str, Any], output_path: Path) -> str:
    font_name = _configure_fonts()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.5), sharey=True, constrained_layout=True)
    for axis, model_name in zip(axes, MODEL_ORDER):
        for strategy in STRATEGY_COLORS:
            series = _learning_series(aggregate, model_name, strategy)
            color = STRATEGY_COLORS[strategy]
            for episode in series["episodes"]:
                budgets = [point["budget"] for point in episode["values"]]
                values = [point["full_512_accuracy"] * 100 for point in episode["values"]]
                axis.plot(budgets, values, color=color, alpha=0.2, linewidth=1.0, zorder=1)
            means = series["mean_curve"]
            axis.plot(
                [point["budget"] for point in means],
                [point["mean"] * 100 for point in means],
                color=color,
                marker="o",
                markersize=4.5,
                linewidth=2.5,
                label="主动查询" if strategy == "active" else "随机查询",
                zorder=3,
            )
        axis.set_title(model_name, fontsize=11.5, pad=9)
        axis.set_xticks(BUDGET_ORDER)
        axis.set_xlabel("已用查询标签预算")
        axis.set_ylim(0, 103)
        axis.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
        axis.grid(axis="y", linestyle=":", alpha=0.4)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("推断规则对全部512个未解决输入的一致率")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("主动与随机查询的规则恢复曲线", fontsize=15, y=1.12)
    fig.text(
        0.5,
        -0.045,
        "细线为单次重建结果，粗线为六次结果均值；全域512包含已查询及未查询输入。",
        ha="center",
        fontsize=9,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return font_name


def _transfer_group(aggregate: Mapping[str, Any], model_name: str, strategy: str) -> Mapping[str, Any]:
    return next(item for item in aggregate["transfer"] if item["model"] == model_name and item["strategy"] == strategy)


def _plot_transfer(aggregate: Mapping[str, Any], output_path: Path) -> str:
    font_name = _configure_fonts()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.7), constrained_layout=True, sharey=True)
    panels = (
        ("unqueried_accuracy", "496个未查询输入上的规则一致率"),
        ("clone_target_agreement", "28张测试工单上的模型输出一致率"),
    )
    base_positions = np.arange(len(MODEL_ORDER), dtype=float)
    offsets = {"active": -0.16, "random": 0.16}
    for axis, (field, title) in zip(axes, panels):
        for strategy in STRATEGY_COLORS:
            color = STRATEGY_COLORS[strategy]
            centers = base_positions + offsets[strategy]
            for model_index, model_name in enumerate(MODEL_ORDER):
                metric = _transfer_group(aggregate, model_name, strategy)[field]
                values = np.asarray(metric["values"], dtype=float) * 100
                mean = float(metric["mean"] * 100)
                minimum, maximum = float(metric["min"] * 100), float(metric["max"] * 100)
                center = centers[model_index]
                jitter = np.linspace(-0.045, 0.045, len(values))
                axis.vlines(center, minimum, maximum, color=color, alpha=0.6, linewidth=2.2, zorder=2)
                axis.scatter(center + jitter, values, color=color, alpha=0.45, s=24, edgecolors="none", zorder=3)
                axis.scatter(
                    [center],
                    [mean],
                    color=color,
                    marker="D",
                    edgecolors="white",
                    linewidths=0.7,
                    s=66,
                    label="主动查询均值" if strategy == "active" else "随机查询均值",
                    zorder=4,
                )
        axis.set_title(title, fontsize=12, pad=10)
        axis.set_xticks(base_positions, MODEL_ORDER, rotation=12, ha="right")
        axis.set_ylim(0, 103)
        axis.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
        axis.grid(axis="y", linestyle=":", alpha=0.4)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("一致率")
    handles, labels = axes[0].get_legend_handles_labels()
    unique_handles: list[Any] = []
    unique_labels: list[str] = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_handles.append(handle)
            unique_labels.append(label)
    fig.legend(unique_handles, unique_labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("推断规则与重建服务的检验", fontsize=15, y=1.12)
    fig.text(
        0.5,
        -0.055,
        "细点为单次结果，菱形为六次均值，竖线为最小值至最大值；右图以原提示的实际输出为参照。",
        ha="center",
        fontsize=9,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return font_name


def _write_plot_data(data: Mapping[str, Any], path: Path = PLOT_DATA_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_figures(
    *,
    experiment2_path: Path = EXPERIMENT2_SUMMARY,
    figures_dir: Path = FIGURES_DIR,
    plot_data_path: Path = PLOT_DATA_PATH,
) -> list[Path]:
    # Validate the complete experiment 2 dataset before writing any output files.
    experiment2 = _read_json(experiment2_path)
    aggregate2 = aggregate_experiment2(experiment2)
    source_path = (
        experiment2_path.relative_to(ROOT).as_posix()
        if experiment2_path.is_relative_to(ROOT)
        else str(experiment2_path)
    )
    aggregate_data = {
        "schema_version": 1,
        "sources": {"experiment2": source_path},
        "experiment2": aggregate2,
    }
    outputs = [
        figures_dir / "experiment2_learning.png",
        figures_dir / "experiment2_transfer.png",
    ]
    font_name = _plot_learning(aggregate2, outputs[0])
    _plot_transfer(aggregate2, outputs[1])
    _write_plot_data(aggregate_data, plot_data_path)
    print(f"Chinese font selected: {font_name}")
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot experiment 2 learning and transfer comparisons.")
    parser.add_argument("--experiment2-summary", type=Path, default=EXPERIMENT2_SUMMARY)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--plot-data", type=Path, default=PLOT_DATA_PATH)
    args = parser.parse_args(argv)
    try:
        outputs = build_figures(
            experiment2_path=args.experiment2_summary,
            figures_dir=args.figures_dir,
            plot_data_path=args.plot_data,
        )
    except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
        print(f"Cannot create comparison figures: {exc}", file=sys.stderr)
        return 2
    for output in outputs:
        print(f"Wrote {_display_path(output)}")
    print(f"Wrote {_display_path(args.plot_data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
