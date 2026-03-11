---
name: analyze-financebench-results
description: "This skill should be used when the user asks to analyze, report on, or visualize results from a FinanceBench evaluation run. It generates a visualization plot and a markdown report from an outputs_judged.jsonl file. TRIGGER when the user mentions analyzing FinanceBench results, generating a report from judged outputs, or visualizing agent evaluation results."
---

# Analyze FinanceBench Results

## Overview

Generate a visualization plot and a concise markdown report from a FinanceBench
`outputs_judged.jsonl` file. The report covers overall accuracy, verdict breakdown,
efficiency analysis (steps and tool calls by verdict), and key takeaways.

## Workflow

### Step 1: Identify the JSONL file

Determine the path to the `outputs_judged.jsonl` file. It is typically located at:
```
results/<run-name>/outputs_judged.jsonl
```
If the user provides a results directory path without specifying the file, append
`/outputs_judged.jsonl` automatically.

### Step 2: Generate the visualization

Run the visualization script from the project root:
```bash
uv run python finance_bench/visualize_results.py <path-to-jsonl> --save <output-dir>/results_viz.png
```
Save `results_viz.png` in the same directory as the JSONL file.

### Step 3: Compute statistics for the report

Load the JSONL file and compute:

1. **Overall accuracy**: correct / total as a percentage
2. **Verdict counts**: correct, incorrect, no_answer (count and percentage)
3. **Efficiency by verdict**: mean steps and mean tool calls for each verdict category
4. **Accuracy by step count**: accuracy at the lowest step count vs highest step counts
5. **Config metadata**: model_name, judge_model, tools used, number of questions

Extract these from the JSONL records. Each line is a JSON object with keys:
`financebench_id`, `model_name`, `system_prompt`, `tools`, `eval_mode`, `temp`,
`question`, `gold_answer`, `model_answer`, `label`, `steps`, `total_tool_calls`,
`judge_model`, `judge_verdict`, `judge_reasoning`.

### Step 4: Write the report

Write `report.md` in the same directory as the JSONL file. Follow this structure:

```markdown
## FinanceBench Results Report — <model_name>

**Config:** <model_name> | <prompt description from run name> | <tools> | <n> questions

![Results Visualization](results_viz.png)

### Overall Performance

<1-2 sentences: accuracy percentage, verdict counts and percentages. Note if the
incorrect rate is a concern.>

### Efficiency Correlates with Correctness

<Bullet points showing mean steps and mean tool calls for each verdict category.>

<1-2 sentences interpreting the accuracy-by-step-count chart: what accuracy is at
low step counts vs high step counts, and what that implies.>

### Key Takeaways

<3 numbered insights derived from the data. Focus on:
1. What the primary bottleneck is (retrieval vs reasoning)
2. Whether the agent uses its step budget efficiently
3. The ratio of incorrect to no_answer and what it says about calibration>
```

### Step 5: Present results

After writing the report, display:
- The console output from the visualization script (accuracy, verdict counts)
- The path to the saved report and plot
- Read and display the plot image so the user can see it inline

## Important Notes

- The visualization script requires `matplotlib` and `numpy`.
- Always save outputs alongside the source JSONL file.
- Derive the prompt description from the run directory name (e.g., `advanced-react-prompt-v2` -> "ReAct prompt v2").
- Incorrect-to-no_answer ratio is computed as: incorrect_count / no_answer_count.
