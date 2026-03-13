"""Classify failure cases from an outputs_judged.jsonl file.

Usage:
    uv run python .claude/skills/analyze-financebench-results/scripts/classify_failures.py <path-to-jsonl>

Outputs JSON to stdout with: category_summary and failed_cases.
"""

import json
import statistics
import sys
from collections import defaultdict


def load_records(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


CATEGORIES = [
    "Retrieval failure",
    "Calculation error",
    "Reasoning error",
    "Hallucination",
    "Insufficient exploration",
    "Other",
]


def extract_judge_reasoning_text(raw: str) -> str:
    """Extract reasoning string from judge output (may be JSON-wrapped)."""
    raw = raw.strip()
    # Try to parse as JSON (possibly wrapped in ```json ... ```)
    cleaned = raw.strip("`").strip()
    if cleaned.startswith("json"):
        cleaned = cleaned[4:].strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed.get("reasoning", raw)
    except (json.JSONDecodeError, ValueError):
        pass
    return raw


def classify(record: dict) -> str:
    """Classify a failed record into a failure category."""
    reasoning = extract_judge_reasoning_text(
        record.get("judge_reasoning", "")
    ).lower()
    model_ans = str(record.get("model_answer", "")).lower()
    verdict = record["judge_verdict"]
    steps = record["steps"]
    tool_calls = record["total_tool_calls"]

    # No-answer cases: typically retrieval or insufficient exploration
    if verdict == "no_answer":
        gave_up_signals = [
            "could not",
            "did not",
            "unable to",
            "not yield",
            "not provide",
            "reached the tool usage limit",
            "could not locate",
            "could not find",
            "was not able",
        ]
        if any(s in model_ans for s in gave_up_signals):
            if steps <= 3 and tool_calls <= 2:
                return "Insufficient exploration"
            return "Retrieval failure"
        # Default no_answer to retrieval failure
        return "Retrieval failure"

    # Incorrect cases: check for numeric mismatch patterns (calculation error)
    numeric_signals_in_reasoning = [
        "not match",
        "differ",
        "not within",
        "rounding",
        "tolerance",
        "exceeds",
        "numerical",
        "numerically",
    ]
    numeric_context = [
        "number",
        "numeric",
        "value",
        "million",
        "percentage",
        "ratio",
        "rate",
        "%",
        "$",
        "amount",
    ]
    if any(s in reasoning for s in numeric_signals_in_reasoning) and any(
        s in reasoning for s in numeric_context
    ):
        return "Calculation error"

    # Hallucination
    if any(s in reasoning for s in ["fabricat", "not in the", "not present in"]):
        return "Hallucination"

    # Reasoning error
    reasoning_signals = [
        "misinterpret",
        "wrong methodology",
        "incorrect logic",
        "does not address",
        "not aligned",
        "confus",
        "contradict",
        "conflict",
        "not a useful metric",
        "dismissed",
    ]
    if any(s in reasoning for s in reasoning_signals):
        return "Reasoning error"

    # Retrieval failure: missing info, omissions
    retrieval_signals = [
        "not mention",
        "omission",
        "missing",
        "did not",
        "failed to",
        "not specifically",
        "incomplete",
        "does not include",
        "does not provide",
    ]
    if any(s in reasoning for s in retrieval_signals):
        return "Retrieval failure"

    # Fallback numeric mismatch -> calculation error
    if any(s in reasoning for s in ["differ", "not match", "exceeds"]):
        return "Calculation error"

    return "Other"


def classify_all(records: list[dict]) -> dict:
    """Classify all failed records and compute aggregate stats."""
    failed = [r for r in records if r["judge_verdict"] in ("incorrect", "no_answer")]

    # Classify each
    cases = []
    for r in failed:
        category = classify(r)
        question_short = r["question"][:60].rstrip()
        if len(r["question"]) > 60:
            question_short += "..."

        root_cause = extract_judge_reasoning_text(r.get("judge_reasoning", ""))[:150]

        cases.append(
            {
                "financebench_id": r["financebench_id"],
                "question_short": question_short,
                "verdict": r["judge_verdict"],
                "category": category,
                "root_cause": root_cause,
                "steps": r["steps"],
                "total_tool_calls": r["total_tool_calls"],
            }
        )

    # Aggregate stats by category
    cat_counts: dict[str, dict] = defaultdict(
        lambda: {"incorrect": 0, "no_answer": 0, "total": 0}
    )
    cat_steps: dict[str, list] = defaultdict(list)
    cat_tools: dict[str, list] = defaultdict(list)

    for c in cases:
        cat = c["category"]
        cat_counts[cat][c["verdict"]] += 1
        cat_counts[cat]["total"] += 1
        cat_steps[cat].append(c["steps"])
        cat_tools[cat].append(c["total_tool_calls"])

    total_failures = len(cases)
    category_summary = []
    for cat in sorted(cat_counts, key=lambda x: -cat_counts[x]["total"]):
        entry = dict(cat_counts[cat])
        entry["category"] = cat
        entry["pct_of_failures"] = (
            round(entry["total"] / total_failures * 100, 0) if total_failures else 0
        )
        entry["mean_steps"] = round(statistics.mean(cat_steps[cat]), 1)
        entry["mean_tool_calls"] = round(statistics.mean(cat_tools[cat]), 1)
        category_summary.append(entry)

    return {
        "total_failures": total_failures,
        "category_summary": category_summary,
        "failed_cases": cases,
    }


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path-to-outputs_judged.jsonl>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    records = load_records(path)
    result = classify_all(records)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
