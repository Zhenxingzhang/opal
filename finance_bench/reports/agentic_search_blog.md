# Let the Agent Search: How Iterative Retrieval and Structured Reasoning Transform Knowledge-Intensive QA

## Abstract

Retrieval-augmented generation (RAG) has become the standard approach for grounding LLMs in external knowledge. But standard RAG treats retrieval as a one-shot, passive step -- the model gets one chance to fetch documents before generating an answer. For complex financial questions that require multi-step reasoning, precise numerical extraction, and domain-specific terminology, this single-pass design falls short.

In this work, we introduce **agentic search** -- a paradigm where the LLM actively controls its own retrieval process through iterative, tool-based interactions. We implement this approach in **OPAL** (Open Platform for Agentic Learning), a modular framework for building and evaluating LLM agents with multi-step reasoning and tool use. Through systematic experiments on FinanceBench -- a benchmark of 150 expert-authored questions over real SEC filings -- we improve accuracy from **36.67% with a traditional single-pass RAG baseline to 80.00%**, a +43pp gain.

Our experiments reveal three key findings:

1. **Agentic search substantially outperforms traditional RAG**, adding +2.7pp from making retrieval agentic alone, and +27.3pp when combined with structured reasoning -- reaching 64% where single-pass RAG plateaus at 37%.
2. **ReAct-style structured reasoning is the highest-leverage intervention**, contributing over half of total improvement through better document comprehension, iterative query refinement, and task decomposition.
3. **Stronger models amplify agentic gains** -- GPT-5 improves +12pp over GPT-4o with the same pipeline, achieving 80% accuracy through qualitatively better reasoning rather than brute-force search.

## The OPAL Framework

OPAL is a general-purpose agentic learning system designed for research on LLM agents with tool use and multi-step reasoning. Rather than being tied to a specific model or task, OPAL provides composable abstractions that let researchers rapidly iterate on agent configurations.

The framework follows a three-layer architecture:

```
┌──────────────────────────────────────────────────────┐
│                   Orchestration                       │
│   SessionRunner  ·  SessionLogger  ·  SessionConfig   │
├──────────────────────────────────────────────────────┤
│                    Environment                        │
│      Tools  ·  Step Tracking  ·  Trajectory State     │
├──────────────────────────────────────────────────────┤
│                     Agentic                           │
│   LLM Models (OpenAI, Anthropic)  ·  Agent Registry   │
│   DefaultAgent  ·  ReActAgent  ·  Prompt Templates    │
└──────────────────────────────────────────────────────┘
```

- **Agentic layer**: Abstracts LLM providers and agent implementations. An `AGENT_REGISTRY` maps names (`"default"`, `"react"`) to agent classes and their associated prompts, making it trivial to swap agent types.
- **Environment layer**: Manages tools (e.g., `search_pdf`, `calculator`), tracks per-step state, and records full execution trajectories for analysis.
- **Orchestration layer**: The `SessionRunner` coordinates agent execution asynchronously, while `SessionLogger` handles structured logging of every LLM call and tool invocation.

This separation means we can change one variable at a time -- swap a prompt, add a tool, switch the agent type, or upgrade the model -- while keeping everything else constant. Every experiment in this blog was produced by changing a single YAML configuration file.

## Experiments on FinanceBench

### Why FinanceBench?

FinanceBench contains 150 expert-authored questions over real 10-K and 10-Q SEC filings from public companies. We chose it for three reasons: (1) questions require precise numerical extraction and computation, not just factual recall; (2) source documents are long, structured financial filings where naive keyword search often fails; and (3) gold answers with explanations enable automated evaluation.

Each question is evaluated as **correct**, **incorrect**, or **no answer** (the model abstains). This three-way classification lets us measure not just accuracy but also *calibration* -- whether the model knows when it doesn't know.

### Results

We ran 12 GPT-4o configurations and 3 GPT-5 configurations. The table below shows the full progression:

| # | Configuration | Model | Correct | Incorrect | No Answer |
|---|--------------|-------|---------|-----------|-----------|
| 0 | Closed-book (no tools) | GPT-4o | 28.67% | 32.67% | 38.67% |
| 1 | Naive search, single-turn (traditional RAG) | GPT-4o | 36.67% | 21.33% | 42.00% |
| 2 | Agentic search, basic prompt | GPT-4o | 39.33% | 20.00% | 40.67% |
| 3 | + ReAct prompt | GPT-4o | 45.33% | 21.33% | 33.33% |
| 4 | + Advanced search guidance | GPT-4o | 49.33% | 23.33% | 27.33% |
| 5 | + Advanced ReAct prompt | GPT-4o | 64.00% | 20.00% | 16.00% |
| 6 | + BGE-large retrieval | GPT-4o | 65.33% | 18.67% | 16.00% |
| 7 | + BGE-large + calculator | GPT-4o | 66.00% | 21.33% | 12.67% |
| 8 | + Reranker | GPT-4o | 66.67% | 24.67% | 8.67% |
| 9 | ReAct agent + BGE-large + calc | GPT-4o | 70.00% | 18.00% | 12.00% |
| 10 | Same as #8, GPT-5 | GPT-5 | 78.67% | 18.67% | 2.67% |
| 11 | ReAct agent + reranker, GPT-5 | GPT-5 | **80.00%** | 16.67% | 3.33% |

```
Closed-book (GPT-4o)                    ████████████████ 28.7%
Traditional RAG (GPT-4o)                ████████████████████ 36.7%
+ Agentic search                        ██████████████████████ 39.3%
+ ReAct prompt                          █████████████████████████ 45.3%
+ Advanced ReAct prompt                 ███████████████████████████████████ 64.0%
+ Retrieval upgrades                    █████████████████████████████████████ 66.7%
+ ReAct agent (GPT-4o best)            ████████████████████████████████████████ 70.0%
+ GPT-5 (default agent)                ████████████████████████████████████████████ 78.7%
+ GPT-5 ReAct agent (best overall)     █████████████████████████████████████████████ 80.0%
```

### What drove the improvement?

The +43pp gain from the traditional RAG baseline decomposes into four compounding factors:

| Factor | Contribution | Experiments |
|--------|-------------|-------------|
| Agentic retrieval (vs. single-pass RAG) | +2.7pp | #1 -> #2 |
| Structured reasoning (ReAct prompts) | +24.7pp | #2 -> #5 |
| Retrieval infrastructure (BGE-large, reranker, calculator) | +2.7pp | #5 -> #8 |
| Stronger model (GPT-5) | +13.3pp | #8 -> #11 |

Prompt engineering and agent architecture account for the bulk of improvement. Infrastructure upgrades (better embeddings, reranker) provide meaningful but smaller gains. The model upgrade from GPT-4o to GPT-5 delivers a large, complementary boost.

## Discussion

### Finding 1: Agentic search outperforms traditional RAG

Traditional RAG -- a single retrieval pass before generation -- already improves over closed-book answering (28.67% -> 36.67%, +8pp). But its gains plateau quickly. Making retrieval agentic -- giving the model a `search_pdf` tool it can call on demand -- adds another +2.7pp (36.67% -> 39.33%) even with a basic prompt. The real payoff comes when agentic search is combined with structured reasoning: the full agentic pipeline reaches 64.00% with GPT-4o, a +27.3pp gain over traditional RAG that single-pass retrieval cannot match.

Why does traditional RAG plateau? It creates two failure modes: (1) the initial query may not retrieve the relevant passage, and (2) the model has no way to follow up when the first retrieval is insufficient. Agentic search eliminates both. We observed agents issuing 2-4 search queries per question, progressively refining their approach -- switching from "capital expenditure" to "purchases of PP&E" when the first query fails, or decomposing "fixed asset turnover" into separate searches for revenue and property values.

The key insight is that retrieval and reasoning are not sequential stages but an interleaved loop. The agent reads initial results, identifies gaps, formulates targeted follow-up queries, and synthesizes across multiple retrievals. This loop is especially valuable for financial questions that span multiple sections of a filing (e.g., computing ratios from figures on different pages).

### Finding 2: ReAct prompts are the highest-leverage intervention

The most striking result is the **super-additive interaction** between ReAct-style reasoning and search guidance. Each alone produces modest gains, but together they account for +24.7pp -- more than half the total improvement.

| | No Search Guidance | Search Guidance |
|---|---|---|
| **No ReAct structure** | 39.3% (agentic baseline) | 49.3% (+10.0) |
| **ReAct structure** | 45.3% (+6.0) | **64.0% (+24.7)** |

This interaction arises because the ReAct Thought-Action-Observation cycle *activates* search strategies that the model otherwise knows but doesn't consistently apply. Our detailed analysis identified three mechanisms, ranked by impact:

**1. Better document comprehension.** In many cases, both the basic and ReAct agents retrieved the *same documents*, but only the ReAct agent extracted the answer. The Observation step forces the agent to explicitly process what it retrieved rather than skimming superficially. For example, when computing General Mills' free cash flow, both agents found the cash flow statement. The basic agent concluded "capital expenditures were not directly provided" and gave up. The ReAct agent's Observation step forced it to parse the retrieved text, find "Purchases of land, buildings, and equipment (460.8)," and compute the answer.

**2. Iterative query refinement.** The ReAct structure encourages persistence. When searching for 3M's capital expenditure, both agents' initial queries failed. The basic agent stopped after 2 attempts. The ReAct agent made a third search using "cash flows from investing activities" -- a synonym switch that finally retrieved the right passage. The explicit "Repeat steps 1-3 as needed" instruction makes the difference between giving up and trying again.

**3. Task decomposition.** For multi-step computations, the ReAct agent naturally breaks the problem into sub-tasks. Computing Activision Blizzard's fixed asset turnover (requiring revenue, and PP&E for two years), the basic agent made unfocused queries and never found PP&E. The ReAct agent planned: first find revenue, then find PP&E, then compute the ratio -- each with a targeted query.

We also found that enforcing ReAct at the code level (the `react` agent type) adds another +4pp over prompt-only ReAct. The agent implementation parses each response into explicit Thought/Action/Observation segments, preventing the model from skipping reasoning steps. This is especially valuable for weaker models that don't always follow prompt instructions faithfully.

### Finding 3: Stronger models amplify agentic gains

Upgrading from GPT-4o to GPT-5 with the same agentic pipeline adds +12pp (66.67% -> 78.67%). Crucially, GPT-5 achieves this **using fewer steps and tool calls** -- it doesn't brute-force its way to better answers but makes smarter decisions at each step.

| Metric | GPT-4o | GPT-5 | Delta |
|--------|--------|-------|-------|
| Correct | 66.67% | 78.67% | +12.0pp |
| Incorrect | 24.67% | 18.67% | -6.0pp |
| No Answer | 8.67% | 2.67% | -6.0pp |
| Avg steps | 3.76 | 3.72 | -0.04 |
| Avg tool calls | 2.79 | 2.72 | -0.07 |

Item-level analysis across the 150 shared questions reveals the nature of GPT-5's improvement:

- **20 items** where GPT-4o was wrong, GPT-5 is correct (fixed reasoning errors)
- **9 items** where GPT-4o abstained, GPT-5 finds the answer
- **11 items** where GPT-5 regresses (mostly interpretation differences and numerical extraction errors)
- **Net: +18 items improved**

The improvements are qualitative, not just quantitative. GPT-5 exhibits three distinct capabilities:

**Better reasoning.** On a Johnson & Johnson question about US vs. international sales growth, GPT-4o conflated "reported" and "operational" growth figures. GPT-5 correctly distinguished between the two, provided the reported figure as the primary answer, and noted the operational figure for context.

**Smarter search strategy.** On a Boeing gross margin question, GPT-4o used 1 search, couldn't find explicit gross margin data, and gave up after 2 steps. GPT-5 used 5 searches across 6 steps, extracted revenue and COGS from the income statement, and computed the margin. Conversely, on a 3M dividend question, GPT-5 answered in 2 steps with 1 search where GPT-4o needed 4 steps and still failed -- GPT-5 formulated a better initial query.

**Assertiveness with retrieval grounding.** Without agentic search, GPT-5's increased willingness to answer is actually a liability -- it generates 3x more incorrect answers than GPT-4o because it confidently fabricates financial data. But in the agentic setting, this same assertiveness becomes a strength: GPT-5 almost never abstains (no-answer: 2.67%), and its answers are usually correct because they're grounded in retrieved documents. This highlights that **retrieval is more important for stronger models** -- their confidence needs to be channeled through verified sources.

The best overall configuration -- GPT-5 with the ReAct agent, BGE-large retrieval, calculator, and reranker -- reaches **80.00%** accuracy with only 16.67% incorrect and 3.33% no-answer. The progression from 36.67% (traditional RAG) to 80.00% demonstrates that agentic search, structured reasoning, and model capability are complementary factors that compound rather than substitute for each other.
