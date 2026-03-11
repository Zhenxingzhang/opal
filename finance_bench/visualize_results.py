"""Visualize outputs_judged.jsonl results from a FinanceBench run."""

import argparse
import json
import logging
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def load_records(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def plot_verdict_pie(ax, records: list[dict]):
    verdicts = Counter(r["judge_verdict"] for r in records)
    colors = {"correct": "#4CAF50", "incorrect": "#F44336", "no_answer": "#9E9E9E"}
    labels = list(verdicts.keys())
    sizes = list(verdicts.values())
    cs = [colors.get(l, "#607D8B") for l in labels]
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=[f"{l}\n({v})" for l, v in zip(labels, sizes)],
        colors=cs,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 11},
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight("bold")
    ax.set_title(f"Judge Verdicts (n={len(records)})", fontsize=13, fontweight="bold")


def plot_steps_distribution(ax, records: list[dict]):
    by_verdict = {}
    for r in records:
        by_verdict.setdefault(r["judge_verdict"], []).append(r["steps"])

    verdicts = ["correct", "incorrect", "no_answer"]
    colors = {"correct": "#4CAF50", "incorrect": "#F44336", "no_answer": "#9E9E9E"}

    all_steps = [r["steps"] for r in records]
    bins = np.arange(min(all_steps), max(all_steps) + 2) - 0.5

    for v in verdicts:
        if v in by_verdict:
            ax.hist(
                by_verdict[v],
                bins=bins,
                alpha=0.7,
                label=f"{v} (μ={np.mean(by_verdict[v]):.1f})",
                color=colors[v],
                edgecolor="white",
            )

    ax.set_xlabel("Number of Steps", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Steps Distribution by Verdict", fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_xticks(range(min(all_steps), max(all_steps) + 1))


def plot_tool_calls_distribution(ax, records: list[dict]):
    by_verdict = {}
    for r in records:
        by_verdict.setdefault(r["judge_verdict"], []).append(r["total_tool_calls"])

    verdicts = ["correct", "incorrect", "no_answer"]
    colors = {"correct": "#4CAF50", "incorrect": "#F44336", "no_answer": "#9E9E9E"}

    all_calls = [r["total_tool_calls"] for r in records]
    bins = np.arange(min(all_calls), max(all_calls) + 2) - 0.5

    for v in verdicts:
        if v in by_verdict:
            ax.hist(
                by_verdict[v],
                bins=bins,
                alpha=0.7,
                label=f"{v} (μ={np.mean(by_verdict[v]):.1f})",
                color=colors[v],
                edgecolor="white",
            )

    ax.set_xlabel("Number of Tool Calls", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Tool Calls Distribution by Verdict", fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_xticks(range(min(all_calls), max(all_calls) + 1))


def plot_steps_vs_tools_scatter(ax, records: list[dict]):
    colors = {"correct": "#4CAF50", "incorrect": "#F44336", "no_answer": "#9E9E9E"}
    for verdict in ["correct", "incorrect", "no_answer"]:
        subset = [r for r in records if r["judge_verdict"] == verdict]
        if subset:
            steps = [r["steps"] for r in subset]
            tools = [r["total_tool_calls"] for r in subset]
            ax.scatter(
                steps,
                tools,
                c=colors[verdict],
                label=verdict,
                alpha=0.6,
                edgecolors="white",
                s=50,
            )
    ax.set_xlabel("Steps", fontsize=11)
    ax.set_ylabel("Tool Calls", fontsize=11)
    ax.set_title("Steps vs Tool Calls", fontsize=13, fontweight="bold")
    ax.legend()


def plot_verdict_by_step_bucket(ax, records: list[dict]):
    """Stacked bar: verdict proportions bucketed by step count."""
    buckets = {}
    for r in records:
        s = r["steps"]
        buckets.setdefault(s, Counter())[r["judge_verdict"]] += 1

    step_vals = sorted(buckets.keys())
    verdicts = ["correct", "incorrect", "no_answer"]
    colors = {"correct": "#4CAF50", "incorrect": "#F44336", "no_answer": "#9E9E9E"}

    bottoms = np.zeros(len(step_vals))
    for v in verdicts:
        heights = [buckets[s].get(v, 0) for s in step_vals]
        ax.bar(
            step_vals,
            heights,
            bottom=bottoms,
            color=colors[v],
            label=v,
            edgecolor="white",
        )
        bottoms += np.array(heights)

    ax.set_xlabel("Steps", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Verdict Breakdown by Step Count", fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_xticks(step_vals)


def plot_accuracy_by_step_bucket(ax, records: list[dict]):
    """Line chart: accuracy rate by step count."""
    buckets: dict[int, list[bool]] = {}
    for r in records:
        s = r["steps"]
        buckets.setdefault(s, []).append(r["judge_verdict"] == "correct")

    step_vals = sorted(buckets.keys())
    accuracies = [np.mean(buckets[s]) * 100 for s in step_vals]
    counts = [len(buckets[s]) for s in step_vals]

    ax.bar(step_vals, accuracies, color="#2196F3", alpha=0.4, edgecolor="white")
    ax.plot(step_vals, accuracies, "o-", color="#1565C0", linewidth=2, markersize=6)

    for x, y, n in zip(step_vals, accuracies, counts):
        ax.annotate(
            f"n={n}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color="#555",
        )

    ax.set_xlabel("Steps", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Accuracy by Step Count", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_xticks(step_vals)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize FinanceBench judged outputs"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="results/gpt-4o-agentic-search-default-agent-advanced-react-prompt-v2-bge-large-calculator_20260311_125206/outputs_judged.jsonl",
        help="Path to outputs_judged.jsonl",
    )
    parser.add_argument(
        "--save", type=str, default=None, help="Save figure to file instead of showing"
    )
    args = parser.parse_args()

    records = load_records(args.path)

    verdicts = Counter(r["judge_verdict"] for r in records)
    accuracy = verdicts.get("correct", 0) / len(records) * 100
    logger.info(
        "Loaded %d records from %s | Model: %s | Judge: %s | Verdicts: %s | Accuracy: %.1f%%",
        len(records),
        args.path,
        records[0]["model_name"],
        records[0]["judge_model"],
        dict(verdicts),
        accuracy,
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f"FinanceBench Results — {records[0]['model_name']}  |  Accuracy: {accuracy:.1f}%",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    plot_verdict_pie(axes[0, 0], records)
    plot_steps_distribution(axes[0, 1], records)
    plot_tool_calls_distribution(axes[0, 2], records)
    plot_steps_vs_tools_scatter(axes[1, 0], records)
    plot_verdict_by_step_bucket(axes[1, 1], records)
    plot_accuracy_by_step_bucket(axes[1, 2], records)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        logger.info("Saved to %s", args.save)
    else:
        plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
