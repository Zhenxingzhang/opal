---
name: analyze-financebench-results
description: "This skill should be used when the user asks to analyze, report on, or visualize results from a FinanceBench evaluation run. It generates a visualization plot and a markdown report from an outputs_judged.jsonl file. TRIGGER when the user mentions analyzing FinanceBench results, generating a report from judged outputs, or visualizing agent evaluation results."
---

# Analyze FinanceBench Results

## Overview

Generate a visualization plot and a concise markdown report from a FinanceBench
`outputs_judged.jsonl` file. The report covers overall accuracy, verdict breakdown,
efficiency analysis (steps and tool calls by verdict), error analysis with failure
classification, and key takeaways.

## Scripts

Reusable scripts live in `.claude/skills/analyze-financebench-results/scripts/`:

| Script | Purpose | Output |
|--------|---------|--------|
| `compute_stats.py` | Overall stats, verdict counts, efficiency by verdict, accuracy by step | JSON to stdout |
| `classify_failures.py` | Classify each failure, aggregate category stats, failed case details | JSON to stdout |

The existing visualization script is at `finance_bench/visualize_results.py`.

## Workflow

### Step 1: Identify the JSONL file

Determine the path to the `outputs_judged.jsonl` file. It is typically located at:
```
results/<run-name>/outputs_judged.jsonl
```
If the user provides a results directory path without specifying the file, append
`/outputs_judged.jsonl` automatically. Store the directory as `$RESULTS_DIR`.

### Step 2: Run all three scripts in parallel

Run these three commands simultaneously (they are independent):

```bash
# Visualization
uv run python finance_bench/visualize_results.py $RESULTS_DIR/outputs_judged.jsonl --save $RESULTS_DIR/results_viz.png

# Summary statistics
uv run python .claude/skills/analyze-financebench-results/scripts/compute_stats.py $RESULTS_DIR/outputs_judged.jsonl

# Failure classification
uv run python .claude/skills/analyze-financebench-results/scripts/classify_failures.py $RESULTS_DIR/outputs_judged.jsonl
```

### Step 3: Write the report

Using the JSON output from both scripts, write `report.md` in `$RESULTS_DIR`.
Follow this structure:

```markdown
## FinanceBench Results Report — <model_name>

**Config:** <model_name> | <prompt description from run name> | <tools> | <n> questions

![Results Visualization](results_viz.png)

### Overall Performance

<1-2 sentences: accuracy percentage, verdict counts and percentages. Note if the
incorrect rate is a concern relative to no_answer rate.>

### Efficiency Correlates with Correctness

<Bullet points showing mean steps and mean tool calls for each verdict category.>

<1-2 sentences interpreting the accuracy-by-step-count data: what accuracy is at
low step counts vs high step counts, and what that implies.>

### Error Analysis

#### Failure Pattern Summary

| Category | Incorrect | No Answer | Total | % of Failures |
|----------|-----------|-----------|-------|---------------|
| <category> | <n> | <n> | <n> | <pct>% |
| ... | ... | ... | ... | ... |

<1-2 sentences summarizing the dominant failure mode and what it implies about the
agent's weaknesses.>

#### Failed Cases

For each failed case from classify_failures output, include a row:

| ID | Question (short) | Verdict | Category | Root Cause | Steps | Tool Calls |
|----|-----------------|---------|----------|------------|-------|------------|
| <id> | <short question> | incorrect/no_answer | <category> | <one-sentence cause> | <n> | <n> |

<2-3 sentences with actionable observations. For example: "X out of Y incorrect
answers stem from calculation errors — adding a verification step or calculator tool
could address this." or "Most no_answer failures occur at high step counts, suggesting
the agent exhausts its search budget without finding the right passage.">

### Key Takeaways

<3 numbered insights derived from the data. Focus on:
1. What the primary bottleneck is (retrieval vs reasoning vs calculation)
2. Whether the agent uses its step budget efficiently
3. The incorrect-to-no-answer ratio and what it says about calibration>
```

### Step 4: Present results

After writing the report, display:
- The console output from the visualization script (accuracy, verdict counts)
- The path to the saved report and plot
- Read and display the plot image so the user can see it inline
- A brief narrative summary (5-8 lines) of the key findings

## Failure Categories Reference

The `classify_failures.py` script uses these categories:

- **Retrieval failure**: Agent could not find or failed to search for the right
  information. Signs: judge_reasoning mentions missing data, model_answer says it
  couldn't find the information, agent exhausted tool budget without result.
- **Calculation error**: Agent found relevant data but computed the wrong result.
  Signs: judge_reasoning mentions wrong numbers, math errors, values that differ
  significantly from gold.
- **Reasoning error**: Agent found data but drew wrong conclusions, misinterpreted
  the question, or dismissed the metric as "not useful". Signs: judge_reasoning
  mentions misinterpretation, wrong methodology, contradiction.
- **Hallucination**: Agent fabricated numbers or facts not in the source. Signs:
  judge_reasoning mentions figures not in the golden answer.
- **Insufficient exploration**: Agent gave up too early (low step count, few tool
  calls). Signs: model_answer admits uncertainty with minimal search effort.
- **Other**: Doesn't fit above categories, or edge cases like verdict/reasoning
  mismatches.

## JSONL Record Schema

Each line in `outputs_judged.jsonl` is a JSON object with keys:

```
financebench_id, model_name, system_prompt, tools, eval_mode, temp,
question, gold_answer, model_answer, label, steps, total_tool_calls,
judge_model, judge_verdict, judge_reasoning
```

## Important Notes

- The visualization script requires `matplotlib` and `numpy`.
- Always save outputs alongside the source JSONL file.
- Derive the prompt description from the run directory name (e.g.,
  `advanced-react-prompt-v2` -> "ReAct prompt v2").
- Incorrect-to-no_answer ratio: `incorrect_count / no_answer_count`.
- The classify_failures script uses heuristic keyword matching on judge_reasoning
  and model_answer. Results should be reviewed — edge cases (e.g., judge reasoning
  says "correct" but verdict is "incorrect") get classified as "Other".
