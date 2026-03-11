# GPT-4o FinanceBench Experiment Report

## Summary Table

| # | Experiment | Prompt | Tools | Correct | Incorrect | No Answer |
|---|-----------|--------|-------|---------|-----------|-----------|
| 1 | Single-turn QA | `default_prompt` (minimal) | None | **20.67%** (31) | 41.33% (62) | 38.00% (57) |
| 2 | Naive search single-turn | `naive_search_prompt` | env-level search | *(empty results file)* | — | — |
| 3 | Agentic search, basic search prompt | `naive_search_prompt` | `search_pdf` tool | **31.33%** (47) | 30.67% (46) | 38.00% (57) |
| 4 | Agentic search, basic ReAct prompt | `react_prompt` | `search_pdf` tool | **38.00%** (57) | 29.33% (44) | 32.67% (49) |
| 5 | Agentic search, advanced search prompt | `agentic_search_prompt` | `search_pdf` tool | **38.67%** (58) | 34.67% (52) | 26.67% (40) |
| 6 | Agentic search, advanced ReAct prompt | `advanced_react_prompt` | `search_pdf` tool | **46.67%** (70) | 38.00% (57) | 15.33% (23) |

All experiments use `gpt-4o-2024-11-20` with temperature 0, max_steps 10, and 150 FinanceBench items.

## Key Changes That Drove Improvement

### 1. Adding agentic tool use (+10.7pp, Exp 1 -> 3)

The biggest single jump came from giving the agent a `search_pdf` tool instead of relying on closed-book knowledge or environment-level retrieval. This let the model actively query the source documents rather than guessing from parametric knowledge. Correct answers went from 20.67% to 31.33%.

### 2. Switching to a ReAct-style prompt (+6.7pp, Exp 3 -> 4)

Replacing the simple `naive_search_prompt` (which just says "use retrieved context") with the `react_prompt` (explicit Thought -> Action -> Observation cycle) improved accuracy from 31.33% to 38.00%. The structured reasoning loop helped the model plan searches more effectively and synthesize results before answering. It also reduced no-answer rate from 38% to 32.67%.

### 3. Adding detailed search guidance in the prompt (+8.7pp, Exp 4 -> 6)

The `advanced_react_prompt` combines the ReAct reasoning structure with detailed instructions on how to use `search_pdf` effectively -- including guidance on crafting good queries, using synonyms/alternate terminology, broadening/narrowing scope, and breaking complex questions into parts. This was the single most impactful change, pushing accuracy to 46.67%.

### 4. Advanced search prompt vs. advanced ReAct prompt (Exp 5 vs 6)

Comparing experiments 5 and 6 isolates the effect of the reasoning structure: both include the same search guidance, but Exp 6 adds the explicit Thought-Action-Observation cycle with stronger instructions ("ALWAYS include your reasoning"). The ReAct framing adds +8pp to accuracy (38.67% -> 46.67%) and dramatically cuts no-answer rate (26.67% -> 15.33%).

## Main Takeaway

The progression from 20.67% to 46.67% (+26pp) was driven by three compounding factors:

1. **Tool access** -- letting the agent search documents instead of closed-book answering
2. **Structured reasoning** -- the ReAct Thought -> Action -> Observation loop for disciplined multi-step problem solving
3. **Domain-specific search guidance** -- teaching the agent how to craft effective queries and recover from failed searches

The combination of structured reasoning and search guidance (Exp 6) was critical -- each alone (Exp 4 or 5) plateaued around 38%, but together they reached 46.67%. The no-answer rate dropped from 38% to 15.33%, indicating the agent became much more persistent at finding information rather than giving up.
