#!/usr/bin/env python3
"""Compare two FinanceBench evaluation runs and produce a detailed comparison."""

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="Compare two FinanceBench runs")
    parser.add_argument("baseline", help="Path to baseline outputs_judged.jsonl")
    parser.add_argument("experiment", help="Path to experiment outputs_judged.jsonl")
    parser.add_argument("--save", help="Path to save markdown report", default=None)
    args = parser.parse_args()

    baseline = load_jsonl(args.baseline)
    experiment = load_jsonl(args.experiment)

    b_dict = {r["financebench_id"]: r for r in baseline}
    e_dict = {r["financebench_id"]: r for r in experiment}
    common_ids = sorted(set(b_dict.keys()) & set(e_dict.keys()))

    b_only = set(b_dict.keys()) - set(e_dict.keys())
    e_only = set(e_dict.keys()) - set(b_dict.keys())

    # --- High-level stats ---
    def stats(records):
        total = len(records)
        verdicts = Counter(r["judge_verdict"] for r in records)
        correct = verdicts.get("correct", 0)
        incorrect = verdicts.get("incorrect", 0)
        no_answer = verdicts.get("no_answer", 0)
        return {
            "total": total,
            "correct": correct,
            "incorrect": incorrect,
            "no_answer": no_answer,
            "accuracy": correct / total * 100 if total else 0,
            "avg_steps": statistics.mean([r["steps"] for r in records]),
            "avg_tools": statistics.mean([r["total_tool_calls"] for r in records]),
            "model": records[0]["model_name"] if records else "unknown",
            "tools": records[0].get("tools", []) if records else [],
        }

    b_stats = stats(baseline)
    e_stats = stats(experiment)

    # --- Transition matrix ---
    transitions = Counter()
    for qid in common_ids:
        bv = b_dict[qid]["judge_verdict"]
        ev = e_dict[qid]["judge_verdict"]
        transitions[(bv, ev)] += 1

    # --- Improved / Regressed ---
    improved = [
        q
        for q in common_ids
        if b_dict[q]["judge_verdict"] != "correct"
        and e_dict[q]["judge_verdict"] == "correct"
    ]
    regressed = [
        q
        for q in common_ids
        if b_dict[q]["judge_verdict"] == "correct"
        and e_dict[q]["judge_verdict"] != "correct"
    ]

    def group_stats(qids, src):
        if not qids:
            return {"avg_steps": 0, "avg_tools": 0}
        return {
            "avg_steps": statistics.mean([src[q]["steps"] for q in qids]),
            "avg_tools": statistics.mean([src[q]["total_tool_calls"] for q in qids]),
        }

    # --- Print results ---
    print("=" * 70)
    print("FINANCEBENCH RUN COMPARISON")
    print("=" * 70)

    print(f"\nBaseline:   {args.baseline}")
    print(f"Experiment: {args.experiment}")
    print(f"Common questions: {len(common_ids)}")
    if b_only:
        print(f"Baseline-only questions: {len(b_only)}")
    if e_only:
        print(f"Experiment-only questions: {len(e_only)}")

    print(f"\n{'Metric':<20} {'Baseline':>12} {'Experiment':>12} {'Delta':>12}")
    print("-" * 58)
    print(
        f"{'Accuracy':<20} {b_stats['accuracy']:>11.1f}% {e_stats['accuracy']:>11.1f}% {e_stats['accuracy'] - b_stats['accuracy']:>+11.1f}%"
    )
    print(
        f"{'Correct':<20} {b_stats['correct']:>12} {e_stats['correct']:>12} {e_stats['correct'] - b_stats['correct']:>+12}"
    )
    print(
        f"{'Incorrect':<20} {b_stats['incorrect']:>12} {e_stats['incorrect']:>12} {e_stats['incorrect'] - b_stats['incorrect']:>+12}"
    )
    print(
        f"{'No Answer':<20} {b_stats['no_answer']:>12} {e_stats['no_answer']:>12} {e_stats['no_answer'] - b_stats['no_answer']:>+12}"
    )
    print(
        f"{'Avg Steps':<20} {b_stats['avg_steps']:>12.2f} {e_stats['avg_steps']:>12.2f} {e_stats['avg_steps'] - b_stats['avg_steps']:>+12.2f}"
    )
    print(
        f"{'Avg Tool Calls':<20} {b_stats['avg_tools']:>12.2f} {e_stats['avg_tools']:>12.2f} {e_stats['avg_tools'] - b_stats['avg_tools']:>+12.2f}"
    )

    print("\n--- Verdict Transition Matrix (baseline -> experiment) ---")
    for bv in ["correct", "incorrect", "no_answer"]:
        for ev in ["correct", "incorrect", "no_answer"]:
            c = transitions.get((bv, ev), 0)
            if c > 0:
                print(f"  {bv:>12} -> {ev:<12}: {c}")

    print(f"\nImproved:  {len(improved)} questions (baseline wrong/na -> experiment correct)")
    if improved:
        bi = group_stats(improved, b_dict)
        ei = group_stats(improved, e_dict)
        print(
            f"  Baseline avg: {bi['avg_steps']:.1f} steps, {bi['avg_tools']:.1f} tools"
        )
        print(
            f"  Experiment avg: {ei['avg_steps']:.1f} steps, {ei['avg_tools']:.1f} tools"
        )

    print(f"\nRegressed: {len(regressed)} questions (baseline correct -> experiment wrong/na)")
    if regressed:
        br = group_stats(regressed, b_dict)
        er = group_stats(regressed, e_dict)
        print(
            f"  Baseline avg: {br['avg_steps']:.1f} steps, {br['avg_tools']:.1f} tools"
        )
        print(
            f"  Experiment avg: {er['avg_steps']:.1f} steps, {er['avg_tools']:.1f} tools"
        )

    print(f"\nNet change: {len(improved) - len(regressed):+d} correct answers")

    # --- Sample regressions ---
    reg_incorrect = [
        q for q in regressed if e_dict[q]["judge_verdict"] == "incorrect"
    ]
    reg_noanswer = [
        q for q in regressed if e_dict[q]["judge_verdict"] == "no_answer"
    ]

    if reg_incorrect:
        print(f"\n--- Sample Regressions: correct -> incorrect ({len(reg_incorrect)} total) ---")
        for qid in reg_incorrect[:5]:
            b = b_dict[qid]
            e = e_dict[qid]
            print(f"\n  ID: {qid}")
            print(f"  Q: {b['question'][:150]}")
            print(f"  Gold: {b['gold_answer'][:150]}")
            print(
                f"  Baseline ({b['steps']}s/{b['total_tool_calls']}t): {b['model_answer'][:150]}"
            )
            print(
                f"  Experiment ({e['steps']}s/{e['total_tool_calls']}t): {e['model_answer'][:150]}"
            )

    if reg_noanswer:
        print(
            f"\n--- Sample Regressions: correct -> no_answer ({len(reg_noanswer)} total) ---"
        )
        for qid in reg_noanswer[:3]:
            b = b_dict[qid]
            e = e_dict[qid]
            print(f"\n  ID: {qid}")
            print(f"  Q: {b['question'][:150]}")
            print(f"  Gold: {b['gold_answer'][:150]}")
            print(
                f"  Baseline ({b['steps']}s/{b['total_tool_calls']}t): {b['model_answer'][:150]}"
            )
            print(
                f"  Experiment ({e['steps']}s/{e['total_tool_calls']}t): {e['model_answer'][:150]}"
            )

    # --- Sample improvements ---
    if improved:
        print(f"\n--- Sample Improvements ({len(improved)} total) ---")
        for qid in improved[:5]:
            b = b_dict[qid]
            e = e_dict[qid]
            bv = b["judge_verdict"]
            print(f"\n  ID: {qid}")
            print(f"  Q: {b['question'][:150]}")
            print(f"  Gold: {b['gold_answer'][:150]}")
            print(
                f"  Baseline ({bv}, {b['steps']}s/{b['total_tool_calls']}t): {b['model_answer'][:150]}"
            )
            print(
                f"  Experiment (correct, {e['steps']}s/{e['total_tool_calls']}t): {e['model_answer'][:150]}"
            )

    # --- Save report if requested ---
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write("# FinanceBench Run Comparison\n\n")
            f.write(f"**Baseline:** `{args.baseline}`  \n")
            f.write(f"**Experiment:** `{args.experiment}`  \n")
            f.write(f"**Common questions:** {len(common_ids)}\n\n")

            # --- Metrics table ---
            f.write("## Metrics\n\n")
            f.write("| Metric | Baseline | Experiment | Delta |\n")
            f.write("|--------|----------|------------|-------|\n")
            f.write(
                f"| **Accuracy** | {b_stats['accuracy']:.1f}% ({b_stats['correct']}/{b_stats['total']}) "
                f"| {e_stats['accuracy']:.1f}% ({e_stats['correct']}/{e_stats['total']}) "
                f"| {e_stats['accuracy'] - b_stats['accuracy']:+.1f}% |\n"
            )
            f.write(
                f"| Incorrect | {b_stats['incorrect']} ({b_stats['incorrect']/b_stats['total']*100:.1f}%) "
                f"| {e_stats['incorrect']} ({e_stats['incorrect']/e_stats['total']*100:.1f}%) "
                f"| {e_stats['incorrect'] - b_stats['incorrect']:+d} |\n"
            )
            f.write(
                f"| No Answer | {b_stats['no_answer']} ({b_stats['no_answer']/b_stats['total']*100:.1f}%) "
                f"| {e_stats['no_answer']} ({e_stats['no_answer']/e_stats['total']*100:.1f}%) "
                f"| {e_stats['no_answer'] - b_stats['no_answer']:+d} |\n"
            )
            f.write(
                f"| Avg Steps | {b_stats['avg_steps']:.2f} | {e_stats['avg_steps']:.2f} "
                f"| {e_stats['avg_steps'] - b_stats['avg_steps']:+.2f} |\n"
            )
            f.write(
                f"| Avg Tool Calls | {b_stats['avg_tools']:.2f} | {e_stats['avg_tools']:.2f} "
                f"| {e_stats['avg_tools'] - b_stats['avg_tools']:+.2f} |\n"
            )

            # --- Transition matrix ---
            f.write("\n## Verdict Transition Matrix (baseline -> experiment)\n\n")
            f.write("| | -> correct | -> incorrect | -> no_answer |\n")
            f.write("|---|---|---|---|\n")
            for bv in ["correct", "incorrect", "no_answer"]:
                row = f"| **{bv}** |"
                for ev in ["correct", "incorrect", "no_answer"]:
                    row += f" {transitions.get((bv, ev), 0)} |"
                f.write(row + "\n")

            f.write(
                f"\n**{len(improved)} improved**, **{len(regressed)} regressed** "
                f"— net {len(improved) - len(regressed):+d} correct answers.\n"
            )

            # --- Efficiency on improved/regressed ---
            if improved:
                bi = group_stats(improved, b_dict)
                ei = group_stats(improved, e_dict)
                f.write(f"\n**Improved ({len(improved)}):** ")
                f.write(
                    f"baseline avg {bi['avg_steps']:.1f} steps / {bi['avg_tools']:.1f} tools, "
                    f"experiment avg {ei['avg_steps']:.1f} steps / {ei['avg_tools']:.1f} tools\n"
                )
            if regressed:
                br = group_stats(regressed, b_dict)
                er = group_stats(regressed, e_dict)
                f.write(f"\n**Regressed ({len(regressed)}):** ")
                f.write(
                    f"baseline avg {br['avg_steps']:.1f} steps / {br['avg_tools']:.1f} tools, "
                    f"experiment avg {er['avg_steps']:.1f} steps / {er['avg_tools']:.1f} tools\n"
                )

            # --- Sample regressions ---
            if reg_incorrect or reg_noanswer:
                f.write("\n## Sample Regressions\n")

            if reg_incorrect:
                f.write(
                    f"\n### correct -> incorrect ({len(reg_incorrect)} total)\n"
                )
                for qid in reg_incorrect[:5]:
                    b = b_dict[qid]
                    e = e_dict[qid]
                    f.write(f"\n**{qid}**\n")
                    f.write(f"- **Q:** {b['question']}\n")
                    f.write(f"- **Gold:** {b['gold_answer'][:300]}\n")
                    f.write(
                        f"- **Baseline** ({b['steps']}s/{b['total_tool_calls']}t): "
                        f"{b['model_answer'][:300]}\n"
                    )
                    f.write(
                        f"- **Experiment** ({e['steps']}s/{e['total_tool_calls']}t): "
                        f"{e['model_answer'][:300]}\n"
                    )

            if reg_noanswer:
                f.write(
                    f"\n### correct -> no_answer ({len(reg_noanswer)} total)\n"
                )
                for qid in reg_noanswer[:5]:
                    b = b_dict[qid]
                    e = e_dict[qid]
                    f.write(f"\n**{qid}**\n")
                    f.write(f"- **Q:** {b['question']}\n")
                    f.write(f"- **Gold:** {b['gold_answer'][:300]}\n")
                    f.write(
                        f"- **Baseline** ({b['steps']}s/{b['total_tool_calls']}t): "
                        f"{b['model_answer'][:300]}\n"
                    )
                    f.write(
                        f"- **Experiment** ({e['steps']}s/{e['total_tool_calls']}t): "
                        f"{e['model_answer'][:300]}\n"
                    )

            # --- Sample improvements ---
            if improved:
                f.write(f"\n## Sample Improvements ({len(improved)} total)\n")
                for qid in improved[:5]:
                    b = b_dict[qid]
                    e = e_dict[qid]
                    bv = b["judge_verdict"]
                    f.write(f"\n**{qid}**\n")
                    f.write(f"- **Q:** {b['question']}\n")
                    f.write(f"- **Gold:** {b['gold_answer'][:300]}\n")
                    f.write(
                        f"- **Baseline** ({bv}, {b['steps']}s/{b['total_tool_calls']}t): "
                        f"{b['model_answer'][:300]}\n"
                    )
                    f.write(
                        f"- **Experiment** (correct, {e['steps']}s/{e['total_tool_calls']}t): "
                        f"{e['model_answer'][:300]}\n"
                    )

            # --- Interpretation placeholder ---
            f.write("\n## Interpretation\n\n")
            f.write("<!-- Claude fills this section after running the script -->\n\n")
            f.write("### High-level Verdict\n\n")
            f.write("_TODO: Did the change help or hurt? "
                    "State the accuracy delta and net change in correct answers._\n\n")
            f.write("### Transition Analysis\n\n")
            f.write("_TODO: Highlight key flows in the transition matrix. "
                    "Focus on which transitions dominate and what they reveal._\n\n")
            f.write("### Efficiency Trade-off\n\n")
            f.write("_TODO: Compare avg steps/tool calls. "
                    "Was the accuracy gain worth the extra cost?_\n\n")
            f.write("### Failure Pattern Analysis\n\n")
            f.write("_TODO: From the regressions, identify recurring patterns "
                    "(premature confidence, misdirection, reasoning errors, tool failures)._\n\n")
            f.write("### Key Takeaways\n\n")
            f.write("_TODO: 3 numbered insights:_\n\n")
            f.write("1. _Primary mechanism of the change_\n")
            f.write("2. _Net effect (positive/negative/neutral)_\n")
            f.write("3. _Concrete suggestion for improvement_\n")

        print(f"\nReport saved to: {save_path}")


if __name__ == "__main__":
    main()
