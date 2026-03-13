---
name: compare-financebench-runs
description: "This skill should be used when the user asks to compare two FinanceBench evaluation runs, diff results between experiments, or understand what changed between two agent configurations. TRIGGER when the user mentions comparing runs, diffing results, analyzing the impact of a config change (e.g., reranker, prompt, model), or asks 'what difference did X make'. Expects two results directory paths or outputs_judged.jsonl paths as arguments."
---

# Compare FinanceBench Runs

## Overview

Compare two FinanceBench evaluation runs to understand the impact of a configuration
change (e.g., adding a reranker, changing prompt, switching model). Produces a
structured comparison with transition matrix, efficiency analysis, sample
regressions/improvements, and an interpretive summary.

## Workflow

### Step 1: Identify the two JSONL files

Determine the paths to both `outputs_judged.jsonl` files. Typically located at:
```
results/<run-name>/outputs_judged.jsonl
```
If the user provides results directory paths without specifying the file, append
`/outputs_judged.jsonl` automatically.

The first path is the **baseline** (control), the second is the **experiment** (treatment).
Infer which is which from context — the older run or the one without the new feature is
the baseline.

### Step 2: Run the comparison script

Execute the bundled comparison script from the project root:
```bash
uv run python .claude/skills/compare-financebench-runs/scripts/compare_runs.py \
  <baseline-jsonl-path> \
  <experiment-jsonl-path> \
  --save <experiment-dir>/comparison_report.md
```

Save `comparison_report.md` in the experiment's results directory.

### Step 3: Fill in the interpretation section

After the script runs, read the saved `comparison_report.md`. The report contains a
`## Interpretation` section with TODO placeholders. Use the Edit tool to replace each
placeholder with substantive analysis:

1. **High-level verdict**: State the accuracy delta, net change in correct answers, and
   whether the change helped or hurt overall.

2. **Transition analysis**: Present the key flows as a markdown table with columns
   `| Flow | Count | Interpretation |`. Focus on:
   - How many questions flipped from correct to incorrect (and vice versa)
   - Whether the change mostly rescued stuck questions (no_answer -> correct) or caused
     previously correct answers to break
   - The dominant transition pattern

3. **Efficiency trade-off**: Compare avg steps and tool calls between the two runs.
   Note whether accuracy gains justify the extra cost.

4. **Failure pattern analysis**: From the sample regressions in the report, identify
   recurring patterns and categorize them:
   - Over-specificity (answer framing differs from gold)
   - Tool/retrieval failures (errors, timeouts)
   - Retrieval misdirection (wrong passage surfaced)
   - Reasoning errors (right data, wrong conclusion)

5. **Key takeaways**: 3 numbered insights covering:
   - The primary mechanism of the change (how it affects retrieval/reasoning)
   - Whether the net effect is positive, negative, or neutral
   - A concrete suggestion for improving the result

### Step 4: Present results

Display:
- The script's console output (metrics table, transition matrix, samples)
- A brief summary of the interpretation (not the full report — the user can read the file)
- Path to the saved comparison report

## JSONL Record Schema

Each line in `outputs_judged.jsonl` is a JSON object with keys:
`financebench_id`, `model_name`, `system_prompt`, `tools`, `eval_mode`, `temp`,
`question`, `gold_answer`, `model_answer`, `label`, `steps`, `total_tool_calls`,
`judge_model`, `judge_verdict`, `judge_reasoning`.

## Important Notes

- Always match questions by `financebench_id` to ensure apples-to-apples comparison.
- Report any questions that exist in only one run (different question sets).
- The baseline is the control; the experiment is the treatment. Label them clearly.
- Derive the change description from the difference in run directory names
  (e.g., one has `reranker` and the other doesn't -> "impact of adding reranker").
