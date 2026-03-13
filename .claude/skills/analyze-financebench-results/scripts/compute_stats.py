"""Compute summary statistics from an outputs_judged.jsonl file.

Usage:
    uv run python .claude/skills/analyze-financebench-results/scripts/compute_stats.py <path-to-jsonl>

Outputs JSON to stdout with: metadata, verdict_counts, efficiency_by_verdict, accuracy_by_step.
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load_records(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_stats(records: list[dict]) -> dict:
    total = len(records)
    r0 = records[0]

    # Verdict counts
    by_verdict: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_verdict[r["judge_verdict"]].append(r)

    verdict_counts = {}
    for v in ["correct", "incorrect", "no_answer"]:
        count = len(by_verdict.get(v, []))
        verdict_counts[v] = {"count": count, "pct": round(count / total * 100, 1)}

    accuracy = verdict_counts["correct"]["pct"]

    # Efficiency by verdict
    efficiency = {}
    for v in ["correct", "incorrect", "no_answer"]:
        group = by_verdict.get(v, [])
        if group:
            steps = [r["steps"] for r in group]
            tools = [r["total_tool_calls"] for r in group]
            efficiency[v] = {
                "mean_steps": round(statistics.mean(steps), 1),
                "mean_tool_calls": round(statistics.mean(tools), 1),
            }

    # Accuracy by step count
    step_groups: dict[int, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in records:
        s = r["steps"]
        step_groups[s]["total"] += 1
        if r["judge_verdict"] == "correct":
            step_groups[s]["correct"] += 1

    accuracy_by_step = {}
    for s in sorted(step_groups.keys()):
        g = step_groups[s]
        accuracy_by_step[s] = {
            "correct": g["correct"],
            "total": g["total"],
            "accuracy_pct": round(g["correct"] / g["total"] * 100, 0),
        }

    # Incorrect-to-no-answer ratio
    n_incorrect = verdict_counts["incorrect"]["count"]
    n_no_answer = verdict_counts["no_answer"]["count"]
    inc_to_na_ratio = (
        round(n_incorrect / n_no_answer, 2) if n_no_answer > 0 else float("inf")
    )

    return {
        "metadata": {
            "model_name": r0.get("model_name", "unknown"),
            "judge_model": r0.get("judge_model", "unknown"),
            "tools": r0.get("tools", []),
            "total_questions": total,
        },
        "accuracy": accuracy,
        "verdict_counts": verdict_counts,
        "efficiency_by_verdict": efficiency,
        "accuracy_by_step": accuracy_by_step,
        "incorrect_to_no_answer_ratio": inc_to_na_ratio,
    }


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path-to-outputs_judged.jsonl>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    records = load_records(path)
    stats = compute_stats(records)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
