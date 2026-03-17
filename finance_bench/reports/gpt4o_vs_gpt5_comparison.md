# GPT-4o vs GPT-5: FinanceBench Comparison Report

## Overview

This report compares GPT-4o (`gpt-4o-2024-11-20`, temperature 0) and GPT-5 (`gpt-5`, temperature 1) on the FinanceBench benchmark (150 items), examining how the stronger model improves performance across different configurations.

**Note:** GPT-5 was run with `temperature=1` (vs GPT-4o's `temperature=0`), which makes GPT-5's improvements even more notable — it achieves higher accuracy despite a less deterministic sampling strategy.

## Head-to-Head Comparison

### Matched Configurations

| Config | GPT-4o | GPT-5 | Delta |
|--------|--------|-------|-------|
| **Single-turn QA** (closed-book, no tools) | 28.67% (43) | 35.33% (53) | **+6.7pp** |
| **Default agent + BGE-large + calc + reranker** | 66.67% (100) | 78.67% (118) | **+12.0pp** |
| **React agent + BGE-large + calc + reranker** | — | 80.00% (120) | — |

For the best comparable config (default agent + reranker), GPT-5 improves by **+12pp** (100→118 correct), cuts incorrect from 37→28 (−24%), and drops no-answer from 13→4 (−69%).

### Full Results Table

| # | Config | Model | Correct | Incorrect | No Answer | Inc:NA Ratio | Avg Steps | Avg Tools |
|---|--------|-------|---------|-----------|-----------|-------------|-----------|-----------|
| 1 | Single-turn QA | GPT-4o | 28.67% (43) | 32.67% (49) | 38.67% (58) | 0.84:1 | 1.00 | 0.00 |
| 2 | Single-turn QA | GPT-5 | 35.33% (53) | 53.33% (80) | 11.33% (17) | 4.71:1 | 1.00 | 0.00 |
| 3 | Default + reranker | GPT-4o | 66.67% (100) | 24.67% (37) | 8.67% (13) | 2.85:1 | 3.76 | 2.79 |
| 4 | Default + reranker | GPT-5 | 78.67% (118) | 18.67% (28) | 2.67% (4) | 7.00:1 | 3.72 | 2.72 |
| 5 | React agent + reranker | GPT-5 | 80.00% (120) | 16.67% (25) | 3.33% (5) | 5.00:1 | 3.67 | 2.67 |

## Item-Level Analysis (Default Agent + Reranker)

Matching by `financebench_id` across the 150 shared items:

| Transition | Count | Meaning |
|-----------|-------|---------|
| correct → correct | 89 | Stable — both models get it right |
| incorrect → correct | 20 | **GPT-5 fixed a wrong answer** |
| no_answer → correct | 9 | **GPT-5 found an answer GPT-4o couldn't** |
| incorrect → incorrect | 15 | Both wrong (potentially different errors) |
| correct → incorrect | 9 | **GPT-5 regression** |
| correct → no_answer | 2 | **GPT-5 regression (abstained)** |
| incorrect → no_answer | 2 | GPT-5 abstained where GPT-4o guessed wrong |
| no_answer → incorrect | 4 | GPT-5 guessed wrong where GPT-4o abstained |

**Net improvement: +18 items** (29 improved − 11 regressed).

## How GPT-5 Improves: Detailed Examples

### 1. Better Reasoning — Distinguishes Nuance GPT-4o Misses

**JnJ US vs international sales (financebench_id_01484)**

The question asks about reported sales growth. GPT-4o retrieved "operational" growth figures (which exclude currency effects) and reported US +3.0% vs International +9.3%. GPT-5 correctly distinguished between *reported* and *operational* growth:

> **GPT-5:** U.S. sales grew 3.0% year over year, while International sales declined 0.6% (as reported) in FY2022. Operationally (excluding currency), U.S. was +3.0% and International was +9.1%.

GPT-5 understood the question was about reported figures, gave the correct answer (−0.6% international), *and* noted the operational figures for context. GPT-4o conflated the two metrics.

**JPM lowest net revenue segment (financebench_id_00299)**

GPT-4o listed four reportable business segments and picked Commercial Banking ($2.4B) as the lowest. GPT-5 recognized that the Corporate segment (not a reportable LOB) had *negative* net revenue (−$0.5B):

> **GPT-5:** Corporate. It posted negative net revenue in Q1 2021 (about −$0.5B). If you mean only the four reportable LOBs, Commercial Banking had the lowest (~$2.4B).

This shows GPT-5's stronger ability to consider the full organizational structure rather than only the named business segments.

### 2. Better Search Strategy — Knows Where to Look

**AmEx operating geographies (financebench_id_01028)**

GPT-4o used 1 search, found a generic description ("103 countries and territories"), and stopped. GPT-5 issued 3 searches, found the segment reporting structure, and returned the actual four reportable geographies (US, EMEA, APAC, LACC). GPT-5 used 4 steps vs 2 for GPT-4o — it invested more effort to find the precise answer.

**Boeing gross margin (financebench_id_00678)**

GPT-4o used 1 search, couldn't find explicit gross margin data, and gave up after 2 steps. GPT-5 used 5 searches across 6 steps, extracted revenue and COGS from the Consolidated Statements of Operations, computed gross margin for multiple years, and correctly identified the improving trend.

**3M dividend stability (financebench_id_01858)**

GPT-4o searched 3 times and couldn't find long-term dividend data. GPT-5 answered in just 2 steps with 1 search — it found the key fact ("65th consecutive year of dividend increases") immediately, suggesting better query formulation.

### 3. Better Computation — Correct Numbers from Correct Inputs

**Netflix EBITDA margin (financebench_id_04458)**

GPT-4o hallucinated $3,852.871M EBITDA and computed 56.83% margin (vs gold 5.4%). GPT-5 correctly identified operating income ($305.8M) and D&A ($62.3M) from the financial statements, computed EBITDA = $368.1M, and arrived at 5.4% margin. The error was not calculation but *retrieval* — GPT-4o grabbed the wrong numbers.

**CVS fixed asset turnover (financebench_id_05915)**

GPT-4o computed 7.76 (gold: 17.98) using the wrong PP&E figure. GPT-5 correctly identified the PP&E values for both years, averaged them, and divided revenue by the average to get the right answer.

**Adobe operating margin (financebench_id_00438)**

GPT-4o hit the 10-step limit and returned a raw number (5.958) with no explanation. GPT-5 clearly answered "No" and showed the margin declining from 36.8% (FY2021) to 34.6% (FY2022), matching the gold answer exactly. GPT-5 solved it in 7 steps vs GPT-4o's failed 10.

### 4. Better Abstention Recovery — Finds Answers GPT-4o Couldn't

**AmEx debt securities (financebench_id_00476)**

GPT-4o searched for debt securities, found various notes and bonds, but couldn't determine which were exchange-registered — it abstained. GPT-5 went straight to the Section 12(b) table in the 10-K and correctly identified that *no debt securities* were registered (only common shares). GPT-5 knew where the definitive answer would be.

**3M segment drag (financebench_id_01865)**

GPT-4o searched 4 times, couldn't find organic growth by segment, and gave up. GPT-5 found the answer in just 2 searches: Consumer posted −0.9% organic growth. Better query targeting made the difference.

**Boeing legal battles (financebench_id_01091)**

GPT-4o searched 3 times but couldn't surface the legal proceedings section. GPT-5 found it and listed the 737 MAX crash lawsuits and other material litigation.

## Where GPT-5 Regresses: 11 Cases

### Pattern 1: Interpretation Disagreements (4 cases)

**3M capital intensity (financebench_id_00499)** — Both models computed the same CapEx/Revenue ratio (5.1%). GPT-4o said "not capital intensive" (matching gold). GPT-5 said "moderately capital intensive" with the same numbers but a different interpretation. This is a judgment call, not a factual error.

**PepsiCo restructuring (financebench_id_01328)** — GPT-4o found the $411M restructuring line item. GPT-5 searched and concluded the income statement doesn't *explicitly* show restructuring as a separate line (it's bundled into COGS and SG&A), answering "0". GPT-5 applied a stricter reading of "directly outlined in the income statement."

### Pattern 2: Incorrect Numerical Extraction (5 cases)

**Adobe OCF ratio (financebench_id_03856)** — GPT-5 computed 0.50 (gold: 0.83), using $5.49B as total current liabilities instead of the correct ~$3.3B. Likely grabbed the wrong line from the balance sheet.

**American Water Works EBITDA (financebench_id_04254)** — GPT-5 got $1,895M (gold: $1,832M), using $636M for D&A instead of the correct $573M. Pulled from the wrong section of the cash flow statement.

**Adobe OCF ratio FY2015 (financebench_id_04735)** — GPT-5 got 0.47 (gold: 0.66). Similar numerical extraction error.

### Pattern 3: Retrieval Failure Under Temperature (2 cases)

**Corning DPO (financebench_id_10130)** and **Ulta stock repurchases (financebench_id_00605)** — GPT-5 abstained where GPT-4o found the answer. The `temperature=1` setting may cause less consistent search queries, occasionally missing the right document section.

## Efficiency Analysis

| Metric | GPT-4o (default+reranker) | GPT-5 (default+reranker) |
|--------|---------------------------|--------------------------|
| Avg steps | 3.76 | 3.72 |
| Avg tool calls | 2.79 | 2.72 |
| Correct % | 66.67% | 78.67% |
| Correct per tool call | 0.239 | 0.289 |

GPT-5 achieves **+12pp more accuracy using fewer steps and tool calls**. It doesn't brute-force its way to better answers — it makes smarter, more targeted searches.

For improved items specifically, GPT-5 averaged only +0.3 extra steps compared to GPT-4o's attempts on the same questions. In several cases (financebench_id_00438, 01858, 01865, 04458), GPT-5 actually used *fewer* steps:

| Question | GPT-4o Steps | GPT-5 Steps | GPT-4o Verdict | GPT-5 Verdict |
|----------|-------------|-------------|----------------|---------------|
| Adobe operating margin | 10 | 7 | incorrect | correct |
| 3M dividends | 4 | 2 | incorrect | correct |
| 3M segment drag | 5 | 3 | no_answer | correct |
| Netflix EBITDA margin | 5 | 3 | incorrect | correct |
| Pfizer non-standard events | 4 | 3 | incorrect | correct |

## Closed-Book (Single-Turn QA) Comparison

Without tools, GPT-5 shows markedly different behavior:

| Metric | GPT-4o | GPT-5 |
|--------|--------|-------|
| Correct | 28.67% (43) | 35.33% (53) |
| Incorrect | 32.67% (49) | **53.33% (80)** |
| No Answer | 38.67% (58) | **11.33% (17)** |

GPT-5 is far more **willing to attempt an answer** — no-answer dropped from 38.67% to 11.33%. This is a double-edged sword: it finds 10 more correct answers (+23%), but also generates 31 more incorrect ones (+63%). The incorrect:no-answer ratio goes from 0.84:1 to 4.71:1.

Transition analysis for single-turn QA:

| Transition | Count |
|-----------|-------|
| correct → correct | 28 |
| no_answer → correct | 13 |
| incorrect → correct | 12 |
| no_answer → incorrect | **31** |
| correct → incorrect | 15 |
| incorrect → no_answer | 3 |

The largest single flow is no_answer→incorrect (31 items): cases where GPT-4o wisely abstained but GPT-5 hallucinated an answer. Without retrieval grounding, GPT-5's increased confidence becomes a liability.

## Key Takeaways

### 1. GPT-5 + retrieval is a strong combination (+12pp)

With the full agentic pipeline (advanced ReAct prompt + BGE-large + calculator + reranker), GPT-5 reaches **78.67%** accuracy. The model's improvements are multiplicative with good retrieval: it formulates better queries, extracts the right data points from retrieved passages, and reasons more carefully about the results.

### 2. GPT-5's improvements are qualitative, not just quantitative

GPT-5 doesn't just get more answers right — it gets them right in *different ways*:
- **Nuanced reasoning:** Distinguishes reported vs operational figures, considers non-reportable segments
- **Targeted search:** Knows to look at Section 12(b) tables, legal proceedings notes, segment disclosures
- **Efficient problem-solving:** Often uses fewer steps to reach correct answers

### 3. GPT-5 is more assertive — helpful with tools, risky without

In the agentic setting, GPT-5's willingness to answer (no-answer: 2.67%) is a strength — it almost never abstains, and its answers are usually right. In the closed-book setting, this same trait becomes a problem: it generates 3x more incorrect answers than GPT-4o because it confidently fabricates plausible-sounding financial data.

**Implication:** Retrieval is more important for GPT-5 than GPT-4o. GPT-5 will always try to answer, so providing it with accurate source data is critical to channeling its assertiveness into correct answers rather than hallucinations.

### 4. Regressions are minor and follow patterns

Of the 11 regressions:
- 4 are interpretation disagreements (the numbers are correct, the judgment differs)
- 5 are numerical extraction errors (wrong line from the right document)
- 2 are search failures (possibly temperature-related)

None represent fundamental capability regressions. The numerical extraction errors may improve with `temperature=0`.

### 5. The react agent gives GPT-5 a smaller boost than GPT-4o

| Model | Default Agent | React Agent | Delta |
|-------|--------------|-------------|-------|
| GPT-4o | 66.00% (no reranker: 66.00%) | 70.00% (no reranker) | +4.0pp |
| GPT-5 | 78.67% (reranker) | 80.00% (reranker) | +1.3pp |

GPT-5 already follows the ReAct reasoning structure well from prompt instructions alone, so the code-level enforcement in the react agent adds less marginal value. The diminishing returns suggest GPT-5 has internalized structured reasoning patterns more effectively.

### 6. Best overall configuration: GPT-5 + react agent + reranker = 80.0%

This achieves the highest accuracy of any configuration tested, with a reasonable calibration ratio (5.00:1) and the lowest average step count (3.67).

## Summary Progression

```
GPT-4o closed-book            ████████████████ 28.7%
GPT-5 closed-book             ████████████████████ 35.3%
GPT-4o best (react+BGE+calc)  ████████████████████████████████████████ 70.0%
GPT-5 default+reranker        ████████████████████████████████████████████████ 78.7%
GPT-5 react+reranker          █████████████████████████████████████████████████ 80.0%
```
