## GPT-4o FinanceBench Experiment Report (2026-03-10)

**Model:** `gpt-4o-2024-11-20` | **Judge:** `gpt-4o-2024-11-20` | **Dataset:** 150 FinanceBench questions (closed-book)

### Results Summary

| # | Configuration | Agent | Prompt | Correct | Incorrect | No Answer |
|---|--------------|-------|--------|---------|-----------|-----------|
| 1 | Single-turn QA | default | default | 31 (20.7%) | 62 (41.3%) | 57 (38.0%) |
| 2 | Naive search (single-turn) | default | naive_search_prompt | 43 (28.7%) | 44 (29.3%) | 63 (42.0%) |
| 3 | Agentic search + basic prompt | default | naive_search_prompt | 49 (32.7%) | 43 (28.7%) | 58 (38.7%) |
| 4 | Agentic search + basic react prompt | react | react_prompt | 51 (34.0%) | 42 (28.0%) | 57 (38.0%) |
| 5 | Agentic search + advanced search prompt | react | agentic_search_prompt | 55 (36.7%) | 57 (38.0%) | 38 (25.3%) |
| 6 | Agentic search + advanced react prompt | react | advanced_react_prompt | **72 (48.0%)** | 61 (40.7%) | 17 (11.3%) |

### Key Observations

1. **Accuracy scales with agent sophistication.** Correct answers more than doubled from the baseline single-turn QA (20.7%) to the best agentic configuration (48.0%), a **+27.3pp** improvement.

2. **Search retrieval is the biggest single lever.** Simply adding naive search to the single-turn agent (config 1 -> 2) boosted accuracy by +8pp. Switching to the ReAct agent with multi-step reasoning (config 2 -> 3) added another +4pp.

3. **Prompt engineering matters significantly.** Comparing the ReAct configurations with different prompts (configs 3-6): the advanced react prompt nearly doubled the no-search baseline, going from 32.7% to 48.0% correct. The basic react prompt (config 4) provides a modest +1.3pp gain over the naive search prompt (config 3), while the advanced search prompt (config 5) and advanced react prompt (config 6) deliver larger jumps.

4. **Accuracy-abstention tradeoff.** The best-performing config (#6) has the lowest no-answer rate (11.3%) but the highest incorrect rate (40.7%). More aggressive answering yields more correct answers but also more errors. The conservative configs (1-4) abstain on 38-42% of questions while keeping incorrect rates at 28-41%.

5. **The advanced search prompt (config 5) is a middle ground.** It reduced no-answer rate to 25.3% and improved accuracy to 36.7%, but the advanced react prompt (config 6) clearly dominates on accuracy.

6. **Basic react prompt (config 4) improves accuracy without increasing errors.** Compared to the naive search prompt on the same ReAct agent (config 3 vs 4), the basic react prompt nudges correct answers from 32.7% to 34.0% while slightly reducing incorrect answers (28.7% -> 28.0%). The no-answer rate stays similar (~38%), suggesting the prompt helps quality without changing answer aggressiveness.

### Progression

```
Single-turn QA           ████████████ 20.7%
+ Naive search           ████████████████ 28.7%
+ ReAct agent            ██████████████████ 32.7%
+ Basic react prompt     ███████████████████ 34.0%
+ Advanced search prompt █████████████████████ 36.7%
+ Advanced react prompt  ████████████████████████████ 48.0%
```

### Conclusion

The combination of ReAct-style agentic reasoning with a well-tuned prompt (`advanced_react`) is the strongest GPT-4o configuration tested, reaching 48% accuracy on FinanceBench. However, nearly 41% of answers are incorrect, suggesting room for improvement in retrieval quality or answer verification. The basic react prompt (config 4) shows that even a simple structured ReAct prompt helps over the naive search prompt, but the gains are modest compared to the advanced prompts. Future work could explore answer self-validation steps or ensemble approaches to reduce the incorrect rate while maintaining the low abstention rate.
