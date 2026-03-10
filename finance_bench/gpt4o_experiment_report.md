## GPT-4o FinanceBench Experiment Report (2026-03-09)

**Model:** `gpt-4o-2024-11-20` | **Judge:** `gpt-4o-2024-11-20` | **Dataset:** 150 FinanceBench questions (closed-book)

### Results Summary

| # | Configuration | Agent | Prompt | Correct | Incorrect | No Answer |
|---|--------------|-------|--------|---------|-----------|-----------|
| 1 | Single-turn QA | default | default | 31 (20.7%) | 62 (41.3%) | 57 (38.0%) |
| 2 | Naive search (single-turn) | default | naive_search | 43 (28.7%) | 44 (29.3%) | 63 (42.0%) |
| 3 | Agentic search | react | naive_search | 49 (32.7%) | 43 (28.7%) | 58 (38.7%) |
| 4 | Agentic search + advanced search prompt | react | agentic_search | 55 (36.7%) | 57 (38.0%) | 38 (25.3%) |
| 5 | Agentic search + advanced react prompt | advanced_react | advanced_react | **72 (48.0%)** | 61 (40.7%) | 17 (11.3%) |

### Key Observations

1. **Accuracy scales with agent sophistication.** Correct answers more than doubled from the baseline single-turn QA (20.7%) to the best agentic configuration (48.0%), a **+27.3pp** improvement.

2. **Search retrieval is the biggest single lever.** Simply adding naive search to the single-turn agent (config 1 -> 2) boosted accuracy by +8pp. Switching to the ReAct agent with multi-step reasoning (config 2 -> 3) added another +4pp.

3. **Prompt engineering matters significantly.** Comparing the two ReAct configurations with different prompts (configs 3 vs 5): the advanced react prompt nearly doubled the no-search baseline, going from 32.7% to 48.0% correct.

4. **Accuracy-abstention tradeoff.** The best-performing config (#5) has the lowest no-answer rate (11.3%) but the highest incorrect rate (40.7%). More aggressive answering yields more correct answers but also more errors. The conservative configs (1-3) abstain on 38-42% of questions.

5. **The advanced search prompt (config 4) is a middle ground.** It reduced no-answer rate to 25.3% and improved accuracy to 36.7%, but the advanced react prompt (config 5) clearly dominates on accuracy.

### Progression

```
Single-turn QA          ████████████ 20.7%
+ Naive search          ████████████████ 28.7%
+ ReAct agent           ██████████████████ 32.7%
+ Advanced search prompt █████████████████████ 36.7%
+ Advanced react prompt  ████████████████████████████ 48.0%
```

### Conclusion

The combination of ReAct-style agentic reasoning with a well-tuned prompt (`advanced_react`) is the strongest GPT-4o configuration tested, reaching 48% accuracy on FinanceBench. However, nearly 41% of answers are incorrect, suggesting room for improvement in retrieval quality or answer verification. Future work could explore answer self-validation steps or ensemble approaches to reduce the incorrect rate while maintaining the low abstention rate.
