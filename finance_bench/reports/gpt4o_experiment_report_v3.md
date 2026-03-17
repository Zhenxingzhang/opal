# GPT-4o FinanceBench Experiment Report (v3)

## Summary Table

| # | Experiment | Prompt | Tools | Retrieval Model | Correct | Incorrect | No Answer |
|---|-----------|--------|-------|-----------------|---------|-----------|-----------|
| 1 | Single-turn QA | `default_prompt` | None | default | **20.67%** (31) | 41.33% (62) | 38.00% (57) |
| 2 | Naive search single-turn | `naive_search_prompt` | env-level search | default | **28.67%** (43) | 29.33% (44) | 42.00% (63) |
| 3 | Agentic, basic search prompt | `naive_search_prompt` | `search_pdf` | default | **31.33%** (47) | 30.67% (46) | 38.00% (57) |
| 4 | Agentic, basic ReAct prompt | `react_prompt` | `search_pdf` | default | **38.00%** (57) | 29.33% (44) | 32.67% (49) |
| 5 | Agentic, advanced search prompt | `agentic_search_prompt` | `search_pdf` | default | **38.67%** (58) | 34.67% (52) | 26.67% (40) |
| 6 | Agentic, advanced ReAct prompt | `advanced_react_prompt` | `search_pdf` | default | **46.67%** (70) | 38.00% (57) | 15.33% (23) |
| 7 | Agentic, advanced ReAct + calculator | `advanced_react_prompt` | `search_pdf` + `calculator` | default | **45.33%** (68) | 40.00% (60) | 14.67% (22) |
| 8 | Agentic, advanced ReAct + BGE-large | `advanced_react_prompt` | `search_pdf` | BGE-large-en-v1.5 | **50.00%** (75) | 30.00% (45) | 20.00% (30) |
| 9 | Agentic, advanced ReAct + BGE-large + calculator | `advanced_react_prompt` | `search_pdf` + `calculator` | BGE-large-en-v1.5 | **54.00%** (81) | 33.33% (50) | 12.67% (19) |

All experiments use `gpt-4o-2024-11-20` with temperature 0, max_steps 10, and 150 FinanceBench items.

## Key Changes That Drove Improvement

### 1. Adding retrieval (+8pp, Exp 1 -> 2)

The biggest single jump came from adding any form of document retrieval. Even environment-level (non-agentic) search in Exp 2 boosted accuracy from 20.67% to 28.67%, showing the model benefits greatly from access to source documents rather than relying on parametric knowledge alone.

### 1b. Making retrieval agentic (+2.7pp, Exp 2 -> 3)

Switching from environment-level search to an agentic `search_pdf` tool gave a smaller but meaningful lift (28.67% -> 31.33%), letting the model actively decide what and when to query.

### 2. Switching to a ReAct-style prompt (+6.7pp, Exp 3 -> 4)

Replacing the simple `naive_search_prompt` (which just says "use retrieved context") with the `react_prompt` (explicit Thought -> Action -> Observation cycle) improved accuracy from 31.33% to 38.00%. The structured reasoning loop helped the model plan searches more effectively and synthesize results before answering. It also reduced no-answer rate from 38% to 32.67%.

### 3. Adding detailed search guidance in the prompt (+8.7pp, Exp 4 -> 6)

The `advanced_react_prompt` combines the ReAct reasoning structure with detailed instructions on how to use `search_pdf` effectively -- including guidance on crafting good queries, using synonyms/alternate terminology, broadening/narrowing scope, and breaking complex questions into parts. This was the single most impactful change, pushing accuracy to 46.67%.

### 4. Advanced search prompt vs. advanced ReAct prompt (Exp 5 vs 6)

Comparing experiments 5 and 6 isolates the effect of the reasoning structure: both include the same search guidance, but Exp 6 adds the explicit Thought-Action-Observation cycle with stronger instructions ("ALWAYS include your reasoning"). The ReAct framing adds +8pp to accuracy (38.67% -> 46.67%) and dramatically cuts no-answer rate (26.67% -> 15.33%).

### 5. Adding a calculator tool (-1.3pp, Exp 6 -> 7)

Adding a `calculator` tool alongside `search_pdf` did not improve accuracy -- it actually dropped slightly from 46.67% to 45.33%, with the incorrect rate climbing from 38.00% to 40.00%. The calculator tool may be introducing noise by encouraging the model to attempt computations that it could already handle inline, or the tool-switching overhead may distract from the core search task.

### 6. Upgrading the retrieval model (+3.3pp, Exp 6 -> 8)

Replacing the default embedding model with `BAAI/bge-large-en-v1.5` for semantic retrieval was a significant improvement on top of the best prompt configuration, reaching 50.00% correct -- a +3.3pp gain over Exp 6. Crucially, the incorrect rate *dropped* from 38.00% to 30.00%, meaning the better retrieval model didn't just make the agent more aggressive but actually improved answer quality.

### 7. BGE-large + calculator combined (+4pp, Exp 8 -> 9)

While adding a calculator tool alone hurt accuracy (Exp 7), combining it with the BGE-large retrieval model yielded the best result: 54.00% correct (+4pp over Exp 8, +7.3pp over Exp 6). With better retrieval providing higher-quality context, the calculator tool becomes useful for verifying and computing financial figures rather than adding noise. The no-answer rate also dropped to 12.67%, the lowest across all experiments (avg 4.33 steps and 3.37 tool calls per item).

## Progression

```
Single-turn QA                    ████████████ 20.7%
+ Naive search                    ████████████████ 28.7%
+ Agentic search                  ██████████████████ 31.3%
+ ReAct prompt                    ██████████████████████ 38.0%
+ Advanced ReAct prompt           ███████████████████████████ 46.7%
+ BGE-large retrieval             █████████████████████████████ 50.0%
+ BGE-large + calculator           ███████████████████████████████ 54.0%
```

## Main Takeaway

The progression from 20.67% to 54.00% (+33pp) was driven by four compounding factors:

1. **Document retrieval** -- giving the model access to source documents instead of closed-book answering (+8pp)
2. **Agentic tool use** -- letting the agent control its own search queries (+2.7pp)
3. **Structured reasoning + search guidance** -- the ReAct prompt with domain-specific search instructions (+15.3pp)
4. **Better retrieval model** -- upgrading to BGE-large-en-v1.5 for higher-quality search results (+3.3pp)
5. **Calculator + better retrieval** -- the calculator tool only helps when paired with high-quality retrieval (+4pp on top of BGE-large)

The combination of structured reasoning and search guidance (Exp 6) was critical -- each alone (Exp 4 or 5) plateaued around 38%, but together they reached 46.67%. Upgrading the retrieval embedding model then pushed this to 50%, and adding the calculator on top reached 54%. Notably, the calculator tool *hurt* with the default retrieval model (Exp 7) but *helped* with BGE-large (Exp 9), suggesting that tool utility depends on retrieval quality.

### Things not working

- **Thinking tool:** "I integrated the thinking tool via the tool and system prompts, but it didn't function as expected. My goal was for the model to use the thinking block to generate more informative search queries; however, the model only triggered the tool at the end of the process, resulting in a refusal to answer."
- **Calculator tool (without better retrieval):** Adding `calculator` alongside `search_pdf` with the default retrieval model slightly hurt accuracy (46.67% -> 45.33%), suggesting the extra tool adds noise when retrieval quality is low. However, it helps when combined with BGE-large (Exp 9: 54.00%).
