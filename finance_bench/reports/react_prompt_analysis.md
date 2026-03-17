# Why React Prompts Improve FinanceBench Performance

**Analysis of GPT-4o runs comparing Search vs React prompts (Basic and Advanced)**

Date: 2026-03-15

## Runs Compared

| Run | Prompt | Correct | No Answer | Incorrect | Accuracy |
|-----|--------|---------|-----------|-----------|----------|
| `gpt-4o-agentic-search-default-agent-basic-search-prompt_20260310_154646` | Basic Search | 59 | 61 | 30 | 39.3% |
| `gpt-4o-agentic-search-default-agent-basic-react-prompt_20260310_155614` | Basic React | 68 | 50 | 32 | **45.3%** (+6.0) |
| `gpt-4o-agentic-search-default-agent-advanced-search-prompt_20260310_160619` | Advanced Search | 74 | 41 | 35 | 49.3% |
| `gpt-4o-agentic-search-default-agent-advanced-react-prompt_20260310_161549` | Advanced React | 96 | 24 | 30 | **64.0%** (+14.7) |

The dominant pattern: **React's biggest impact is converting "no answer" into correct answers** (61->50 basic, 41->24 advanced). It makes the agent persist and reason rather than give up.

## Prompt Differences

### Basic Pair

**Naive Search Prompt** (`opal/prompt/naive_search_prompt.txt`):
> You are a helpful assistant that answers questions using retrieved context and reasoning. You will be provided with relevant documents retrieved from a knowledge base. Use the context to answer the user's question accurately. If the retrieved context does not contain enough information to answer, do not prompt user for more information.

**React Prompt** (`opal/prompt/react_prompt.txt`):
> You are a helpful assistant that uses reasoning and actions to solve problems. When given a task, follow this process: 1. Thought: Think about what you need to do and why. 2. Action: If you need to use a tool, call it with the appropriate arguments. 3. Observation: Review the result from the tool. 4. Repeat steps 1-3 as needed until you have enough information. 5. Answer: Provide your final response to the user. Always think step by step.

### Advanced Pair

**Advanced Search Prompt** (`opal/prompt/agentic_search_prompt.txt`): Contains detailed search guidance (crafting queries, reformulation strategies, synonym usage) but no explicit reasoning structure.

**Advanced React Prompt** (`opal/prompt/advanced_react_prompt.txt`): Shares the same search guidance but adds a mandated Thought-Action-Observation cycle, requires reasoning text with every tool call, and imposes a 3-call tool budget.

## Three Mechanisms of Improvement (Ranked by Impact)

### 1. Better Reasoning Over Retrieved Context (Largest Contributor)

The most surprising finding: in many cases **both agents retrieved the same documents**, but React extracted the answer while Search did not.

#### Example: General Mills FCF (financebench_id_04854, advanced pair)

- **Question:** "What is the FY2020 free cash flow (FCF) for General Mills? FCF = cash from operations - capex."
- **Gold answer:** $3,215.00
- **Search:** no_answer | **React:** correct

Both agents retrieved the cash flow statement containing `"Purchases of land, buildings, and equipment (460.8)"`. The search agent said *"specific capital expenditures were not directly provided"* and gave up. The React agent's structured Observation step forced it to actually parse the results, extract the $460.8M figure, and compute FCF = $3,676M - $461M = $3,215M.

#### Example: AMD D&A Margin (financebench_id_03069, basic pair)

- **Question:** "What is the FY2015 D&A margin for AMD?"
- **Gold answer:** 4.2%
- **Search:** no_answer | **React:** correct

Both agents retrieved identical chunks including quarterly revenue data (`$958 $1,061 $942 $1,030`). The search agent looked only for an explicit "FY2015 revenue" line and gave up. The React agent's "think step by step" instruction led it to notice the quarterly figures, sum them ($3,991M), and compute 167/3991 = 4.19%.

#### Example: 3M Capital Intensity (financebench_id_00499, advanced pair)

- **Question:** "Is 3M a capital-intensive business based on FY2022 data?"
- **Gold answer:** No -- CAPEX/Revenue 5.1%, Fixed assets/Total Assets 20%, ROA 12.4%.
- **Search:** incorrect | **React:** correct

The search agent found CapEx = $1.749B and immediately concluded "3M is capital-intensive" (qualitative, wrong). The React agent's Thought step said *"I need to find revenue to compute a ratio"* -- it then fetched revenue ($34.2B), calculated CapEx/Revenue = 5.1%, and correctly concluded "not capital-intensive."

**Why this happens:** The React prompt's Thought->Action->Observation cycle forces the agent to explicitly record what it learned from each search before deciding what to do next. Without this structure, the search agent often skims results superficially and concludes the data isn't there.

### 2. Iterative Query Refinement (Second Largest Contributor)

React agents make **more search calls with progressively refined queries**, eventually hitting the right chunk.

#### Example: 3M FY2018 Capital Expenditure (financebench_id_03029, basic pair)

- **Gold answer:** $1,577 million
- **Search:** no_answer (2 queries) | **React:** correct (3 queries)

Both agents tried `"FY2018 capital expenditure 3M"` and similar queries -- both failed. The search agent stopped after 2 queries. The React agent made a **3rd search**: `"cash flows from investing activities FY2018 3M"` -- this synonym switch from "capital expenditure" to "investing activities" finally retrieved the line `"Purchases of PP&E $(1,577)"`.

#### Example: 3M Segment Growth (financebench_id_01865, advanced pair)

- **Question:** "If we exclude the impact of M&A, which segment has dragged down 3M's overall growth in 2022?"
- **Gold answer:** Consumer segment shrunk by 0.9% organically
- **Search:** no_answer (2 queries) | **React:** correct (4 queries)

Both started with `"3M segment dragging down overall growth 2022 excluding M&A impact"` -- too broad. The search agent tried one similar reformulation and gave up. The React agent iterated through 4 queries, progressively narrowing: `"M&A impact"` -> `"organic growth"` -> `"organic sales growth excluding acquisitions"`. The third query finally retrieved the Consumer segment table showing -0.9% organic sales.

#### Example: 3M FY2018 Net PP&E (financebench_id_04672, basic pair)

- **Gold answer:** $8.70B
- **Search:** no_answer (2 queries) | **React:** correct (4 queries)

The search agent stopped after 2 queries using the abbreviation "PPNE." The React agent made 4 queries, eventually spelling out `"property plant and equipment net"` which matched the balance sheet chunk containing $8,738M.

**Why this happens:** The React prompt explicitly says *"Repeat steps 1-3 as needed until you have enough information."* The basic search prompt says *"If the retrieved context does not contain enough information to answer, do not prompt user for more information"* -- which actively encourages giving up.

### 3. Better Planning and Decomposition (Moderate Contributor)

React agents break multi-part questions into sub-tasks, fetching each data point separately.

#### Example: Activision Blizzard Fixed Asset Turnover (financebench_id_02987, basic pair)

- **Gold answer:** 24.26
- **Search:** no_answer (5 steps) | **React:** correct (7 steps)

Computing this ratio requires: FY2019 revenue, FY2018 PP&E, FY2019 PP&E. The search agent made 4 unfocused queries and never found PP&E. The React agent planned a decomposition -- first find revenue (4 queries to nail down $6,489M), then find PP&E (1 targeted query found Note 6 with $253M/$282M). Computed 24.26.

#### Example: AMCOR Gross Margin (financebench_id_00684, basic pair)

- **Gold answer:** slight decline of 0.8%
- **Search:** incorrect | **React:** correct

Both found the same income statement. The search agent described the trend qualitatively ("gross profit declined"). The React agent explicitly computed: 2725/14694 = 18.55% vs 2820/14544 = 19.39%, correctly identifying the -0.84% decline.

#### Example: 3M Operating Margin Drivers (financebench_id_01226, advanced pair)

- **Gold answer:** Decreased by 1.7% due to gross margin decline, Combat Arms litigation, PFAS exit, Russia exit costs.
- **Search:** incorrect (2 queries) | **React:** correct (1 query)

Paradoxically, the React version used *fewer* searches (1 vs 2) but produced a correct answer. The key was the Observation phase -- the React prompt forced the agent to carefully review search results before proceeding. The search version's results contained similar content but the agent failed to extract the relevant information. The React version's explicit reasoning structure forced it to synthesize what it found rather than give up.

## React Failure Modes

React isn't all upside. Two patterns caused regressions:

### 1. `top_k=1` Over-Focus (basic pair)

The React agent sometimes used `top_k=1` to be "efficient," which backfired when the answer was in lower-ranked chunks.

**Example -- Amazon FY2019 Net Income (financebench_id_08286):** The answer was result #5 at score 0.549. The search agent's `top_k=5` caught it, but React's `top_k=1` missed it across 3 different query attempts.

### 2. Query Reformulations That Don't Improve Retrieval

Sometimes the React agent burns its search budget on incremental rewording that retrieves the same chunks, while the search agent gets lucky with a different initial phrasing.

**Example -- Adobe FY2017 Operating Cash Flow Ratio (financebench_id_03856):** The search agent found both data points needed. The React agent found current liabilities but could not find cash from operations despite multiple queries.

## Why is the Advanced React Improvement (+14.7%) So Much Larger Than Basic (+6.0%)?

The advanced search prompt already contains detailed guidance on *how* to formulate good queries (synonym usage, breaking questions into parts, reformulation strategies). But **it provides the WHAT without the WHEN**.

The React framework provides the WHEN -- explicit reasoning checkpoints that *activate* the search guidance:

| | Search Guidance (WHAT) | Reasoning Structure (WHEN) | Result |
|---|---|---|---|
| Basic Search | No | No | 39.3% |
| Basic React | No | Yes | 45.3% |
| Advanced Search | Yes | No | 49.3% |
| Advanced React | Yes | Yes | **64.0%** |

The interaction is **super-additive**: search guidance alone adds +10%, React structure alone adds +6%, but together they add +24.7%. The Thought step before each search naturally triggers the agent to apply the synonym and decomposition strategies from the advanced prompt. Without React's structure, the agent *knows* these strategies but doesn't consistently *use* them.

Additionally, the advanced React prompt's **3-call tool budget** paradoxically helps by forcing upfront planning rather than speculative searches. The agent writes better first queries when it knows it has limited attempts.

## Summary

React helps through **three mechanisms in order of impact**:

| Mechanism | Contribution | Description |
|-----------|-------------|-------------|
| Doc comprehension | Largest | Forces the agent to actually read and extract from retrieved results via the Observation step, rather than superficially scanning and giving up |
| Iterative refinement | Second largest | Encourages persistence through multiple search rounds with synonym/reformulation strategies |
| Task decomposition | Moderate | Plans multi-step calculations (find revenue, find CapEx, compute ratio) rather than hoping one search returns everything |

The +14.7% advanced improvement >> +6% basic improvement because React's reasoning structure *activates* the search strategies that the advanced prompt teaches but the agent otherwise doesn't consistently apply.
