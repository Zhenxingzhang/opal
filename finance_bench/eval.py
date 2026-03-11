import os
import hashlib
import asyncio
import json
import logging
from collections import Counter
from pathlib import Path
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

CACHE_DIR = Path(__file__).parent / ".eval_cache"
MAX_CONCURRENCY = 10

_caching_enabled = True

# Lazy singleton clients
_openai_client: AsyncOpenAI | None = None
_deepseek_client: AsyncOpenAI | None = None


def _get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def _get_deepseek_client() -> AsyncOpenAI:
    global _deepseek_client
    if _deepseek_client is None:
        _deepseek_client = AsyncOpenAI(
            api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL
        )
    return _deepseek_client


def _cache_key(model: str, prompt: str) -> str:
    raw = json.dumps({"model": model, "prompt": prompt}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def _sanitize_model_name(model: str) -> str:
    return model.replace("/", "_").replace("\\", "_") if model else "unknown"


def _cache_path(model: str, prompt: str) -> Path:
    return CACHE_DIR / _sanitize_model_name(model) / f"{_cache_key(model, prompt)}.json"


def _load_cache(model: str, prompt: str) -> str | None:
    if not _caching_enabled:
        return None
    cache_file = _cache_path(model, prompt)
    if cache_file.exists():
        entry = json.loads(cache_file.read_text())
        return entry["response"]
    return None


def _save_cache(model: str, prompt: str, response: str):
    if not _caching_enabled:
        return
    path = _cache_path(model, prompt)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"model": model, "prompt": prompt, "response": response}
    path.write_text(json.dumps(entry))


async def get_completion_async(prompt, model=None, semaphore=None):
    cached = _load_cache(model, prompt)
    if cached is not None:
        return cached

    if model == DEEPSEEK_MODEL:
        openai_client = _get_deepseek_client()
    else:
        openai_client = _get_openai_client()

    sem = semaphore or asyncio.Semaphore(1)
    async with sem:
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(3):
            try:
                response = await openai_client.chat.completions.create(
                    model=model, messages=messages
                )
                text = response.choices[0].message.content
                _save_cache(model, prompt, text)
                return text
            except Exception as e:
                logger.warning(
                    "Attempt %d failed for model %s: %s", attempt + 1, model, e
                )
                await asyncio.sleep(1)
        return None


VALID_VERDICTS = {"correct", "incorrect", "no_answer"}


def _parse_judge_response(response: str | None) -> dict:
    """Parse a structured JSON judge response with 3-way verdict."""
    if response is None:
        return {"verdict": "no_answer", "reasoning": "No response from judge model."}

    # Try parsing as JSON first
    try:
        parsed = json.loads(response.strip())
        if isinstance(parsed, dict) and "verdict" in parsed:
            verdict = parsed["verdict"]
            # Handle legacy boolean verdicts from cache
            if isinstance(verdict, bool):
                verdict = "correct" if verdict else "incorrect"
            if verdict not in VALID_VERDICTS:
                verdict = "incorrect"
            return {
                "verdict": verdict,
                "reasoning": parsed.get("reasoning", ""),
            }
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: check for verdict strings or legacy true/false in plain text
    lower = response.lower()
    if "no_answer" in lower:
        verdict = "no_answer"
    elif "correct" in lower and "incorrect" not in lower:
        verdict = "correct"
    elif "true" in lower:
        verdict = "correct"
    else:
        verdict = "incorrect"
    return {"verdict": verdict, "reasoning": response.strip()}


async def check_answer_equivalence(
    answer, gold_answer, query=None, model="gpt-4o-2024-11-20", semaphore=None
):
    query_prompt = f"- Query: {query}" if query else ""

    prompt = f"""You are an expert evaluator for AI-generated responses to queries. Your task is to classify the AI-generated answer into one of three categories based on the golden answer provided by a human expert.

Categories:
- "correct": The AI-generated answer correctly answers the query (matches the golden answer).
- "incorrect": The AI attempted to answer but the response is wrong or inaccurate.
- "no_answer": The AI failed to produce an answer — it refused, said it couldn't answer, gave up, or produced no meaningful response.

Numerical Accuracy (STRICT):
- Numeric values must match EXACTLY between the AI-generated answer and the golden answer. Any difference in numbers, percentages, fractions, or figures means the answer is "incorrect".
- No rounding tolerance is allowed. For example, if the golden answer is 1.23, the AI answer of 1.2 is "incorrect".
- Equivalent representations of the same exact value are acceptable: "50%" and "0.50" and "1/2" all represent the same value and are considered matching.
- If the golden answer contains a specific number and the AI-generated answer contains a different number, the verdict is "incorrect" — even if the difference is small.

Evaluation Criteria:
- For answers involving numeric values: the numbers must match exactly (see Numerical Accuracy above). Any numeric discrepancy makes the answer "incorrect".
- For non-numeric answers: the AI-generated answer is "correct" if it conveys the same meaning, conclusion, or rationale as the golden answer.
- If the AI-generated answer is a superset of the golden answer (contains the correct answer plus additional information), it is "correct".
- If the AI-generated answer is empty, states it cannot answer, or does not attempt the question, it is "no_answer".
- Otherwise, the AI-generated answer is "incorrect".

Inputs:
{query_prompt}
- AI-Generated Answer: {answer}
- Golden Answer: {gold_answer}

Your output should be ONLY a JSON object with the following format, nothing else:
{{"verdict": "correct", "reasoning": "brief explanation"}}

Set "verdict" to one of: "correct", "incorrect", or "no_answer". Provide a brief reasoning for your judgment."""

    response = await get_completion_async(prompt, model=model, semaphore=semaphore)

    return _parse_judge_response(response)


async def judge_benchmark_results(
    benchmark_results,
    model="gpt-4o-2024-11-20",
    concurrency=MAX_CONCURRENCY,
    gold_answer_col="gold_answer",
    model_answer_col="model_answer",
):
    semaphore = asyncio.Semaphore(concurrency)
    tasks = []
    for result in benchmark_results:
        query = result["question"]
        gold_answer = result[gold_answer_col]
        answer = result[model_answer_col]
        tasks.append(
            check_answer_equivalence(
                answer, gold_answer, query=query, model=model, semaphore=semaphore
            )
        )
    results = await tqdm.gather(*tasks, desc="Evaluating answers")
    return results


def judge_benchmark_results_from_file(
    json_file_path,
    model="gpt-4o-2024-11-20",
    concurrency=MAX_CONCURRENCY,
    gold_answer_col="gold_answer",
    model_answer_col="model_answer",
):
    with open(json_file_path, "r") as f:
        if json_file_path.endswith(".jsonl"):
            benchmark_results = [json.loads(line) for line in f if line.strip()]
        else:
            benchmark_results = json.load(f)
    results = asyncio.run(
        judge_benchmark_results(
            benchmark_results,
            model=model,
            concurrency=concurrency,
            gold_answer_col=gold_answer_col,
            model_answer_col=model_answer_col,
        )
    )
    incorrect_indexes = [
        i for i, result in enumerate(results) if result["verdict"] == "incorrect"
    ]
    no_answer_indexes = [
        i for i, result in enumerate(results) if result["verdict"] == "no_answer"
    ]
    logger.info("Incorrect indexes: %s", incorrect_indexes)
    logger.info("No-answer indexes: %s", no_answer_indexes)
    return results, benchmark_results


def judge_benchmark_results_from_file_hybrid(
    json_file_path,
    models=["gpt-4o-2024-11-20", "o1-mini", "o3-mini"],
    concurrency=MAX_CONCURRENCY,
    gold_answer_col="gold_answer",
    model_answer_col="model_answer",
):
    if not models:
        raise ValueError("No models provided for hybrid evaluation.")

    hybrid_results = {}
    all_benchmark_results = None
    for model in models:
        logger.info("Judging results using model '%s'...", model)
        results, benchmark_results = judge_benchmark_results_from_file(
            json_file_path,
            model=model,
            concurrency=concurrency,
            gold_answer_col=gold_answer_col,
            model_answer_col=model_answer_col,
        )
        hybrid_results[model] = results
        if all_benchmark_results is None:
            all_benchmark_results = benchmark_results

    combined_results = []
    for i in range(len(hybrid_results[models[0]])):
        # OR logic: correct if any judge says correct
        any_correct = any(
            hybrid_results[model][i]["verdict"] == "correct" for model in models
        )
        any_no_answer = all(
            hybrid_results[model][i]["verdict"] == "no_answer" for model in models
        )
        # Collect reasoning from all models
        reasoning_parts = []
        for model in models:
            r = hybrid_results[model][i]
            reasoning_parts.append(f"{model}: {r['reasoning']}")
        if any_correct:
            verdict = "correct"
        elif any_no_answer:
            verdict = "no_answer"
        else:
            verdict = "incorrect"
        combined_results.append(
            {
                "verdict": verdict,
                "reasoning": " | ".join(reasoning_parts),
            }
        )

    return combined_results, all_benchmark_results


def calculate_accuracy(results: list[dict]) -> dict[str, float]:
    """Calculate accuracy breakdown from a list of judge result dicts.

    Args:
        results: A list of dicts with "verdict" keys ("correct", "incorrect", "no_answer").

    Returns:
        Dict with "correct", "incorrect", "no_answer" rates (each 0.0-1.0).

    Raises:
        ValueError: If the results list is empty.
    """
    if not results:
        raise ValueError("Results list is empty.")
    n = len(results)
    correct = sum(1 for r in results if r["verdict"] == "correct") / n
    incorrect = sum(1 for r in results if r["verdict"] == "incorrect") / n
    no_answer = sum(1 for r in results if r["verdict"] == "no_answer") / n
    return {"correct": correct, "incorrect": incorrect, "no_answer": no_answer}


def save_judged_results(
    benchmark_results: list[dict],
    judge_results: list[dict],
    model: str,
    output_path: str,
):
    """Write per-item judged results to a JSONL file."""
    with open(output_path, "w") as f:
        for record, judge in zip(benchmark_results, judge_results):
            enriched = {
                **record,
                "judge_model": model,
                "judge_verdict": judge["verdict"],
                "judge_reasoning": judge["reasoning"],
            }
            f.write(json.dumps(enriched) + "\n")
    logger.info("Judged results saved to %s", output_path)


def _aggregate_tool_usage(benchmark_results: list[dict]) -> dict[str, float]:
    """Compute average per-tool call count across benchmark results.

    Returns a dict mapping tool name to average calls per item (only items
    that have ``tool_usage`` data are counted).
    """
    total: Counter[str] = Counter()
    count = 0
    for r in benchmark_results:
        usage = r.get("tool_usage")
        if usage and isinstance(usage, dict):
            total.update(usage)
            count += 1
    if count == 0:
        return {}
    return {tool: total[tool] / count for tool in sorted(total)}


def write_summary(
    input_file: str,
    judge_model: str,
    accuracy: dict[str, float],
    results: list[dict],
    benchmark_results: list[dict],
    output_path: str,
):
    """Write a short summary of the eval results to a text file next to the judged output."""
    summary_path = Path(output_path).with_suffix(".summary.txt")
    total = len(results)
    correct = sum(1 for r in results if r["verdict"] == "correct")
    incorrect = sum(1 for r in results if r["verdict"] == "incorrect")
    no_answer = sum(1 for r in results if r["verdict"] == "no_answer")

    lines = [
        "Eval Summary",
        "============",
        f"Input file:    {input_file}",
        f"Judge model:   {judge_model}",
        f"Total items:   {total}",
        "",
        "Results:",
        f"  Correct:     {correct}/{total} ({accuracy['correct']:.2%})",
        f"  Incorrect:   {incorrect}/{total} ({accuracy['incorrect']:.2%})",
        f"  No answer:   {no_answer}/{total} ({accuracy['no_answer']:.2%})",
    ]

    # Agent cost metrics (steps and tool usage)
    steps_values = [
        r.get("steps") for r in benchmark_results if r.get("steps") is not None
    ]
    tool_values = [
        r.get("total_tool_calls")
        for r in benchmark_results
        if r.get("total_tool_calls") is not None
    ]
    if steps_values or tool_values:
        lines += ["", "Agent Cost:"]
        if steps_values:
            avg_steps = sum(steps_values) / len(steps_values)
            lines.append(
                f"  Avg steps:        {avg_steps:.2f} (over {len(steps_values)} items)"
            )
        if tool_values:
            avg_tools = sum(tool_values) / len(tool_values)
            lines.append(
                f"  Avg tool calls:   {avg_tools:.2f} (over {len(tool_values)} items)"
            )

    avg_tool_usage = _aggregate_tool_usage(benchmark_results)
    if avg_tool_usage:
        lines += ["", "Avg Tool Usage (per item):"]
        for tool_name, avg in avg_tool_usage.items():
            lines.append(f"  {tool_name:20s} {avg:.2f}")

    incorrect_ids = [i for i, r in enumerate(results) if r["verdict"] == "incorrect"]
    no_answer_ids = [i for i, r in enumerate(results) if r["verdict"] == "no_answer"]
    if incorrect_ids:
        lines += ["", f"Incorrect indexes: {incorrect_ids}"]
    if no_answer_ids:
        lines += ["", f"No-answer indexes: {no_answer_ids}"]

    summary_path.write_text("\n".join(lines) + "\n")
    logger.info("Summary written to %s", summary_path)


def clear_cache():
    """Remove all cached LLM responses."""
    if CACHE_DIR.exists():
        import shutil

        for child in CACHE_DIR.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        logger.info("Cache cleared: %s", CACHE_DIR)
    else:
        logger.info("No cache directory found.")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Evaluate finance benchmark results using LLM judges."
    )
    parser.add_argument(
        "file",
        nargs="?",
        default="gpt-4_singleStore.jsonl",
        help="Path to the benchmark results file (.json or .jsonl)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-2024-11-20",
        help="Judge model to use (default: gpt-4o-2024-11-20)",
    )
    parser.add_argument(
        "--hybrid",
        nargs="+",
        metavar="MODEL",
        help="Use multiple judge models and combine results (OR logic)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the LLM response cache and exit",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Disable cache for this run"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=MAX_CONCURRENCY,
        help=f"Max concurrent API calls (default: {MAX_CONCURRENCY})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for judged results JSONL (default: <input_stem>_judged.jsonl)",
    )
    parser.add_argument(
        "--gold-answer-col",
        default="gold_answer",
        help="Column name for the gold/reference answer (default: gold_answer)",
    )
    parser.add_argument(
        "--model-answer-col",
        default="model_answer",
        help="Column name for the model-generated answer (default: model_answer)",
    )
    args = parser.parse_args()

    if args.clear_cache:
        clear_cache()
        exit(0)

    if args.no_cache:
        _caching_enabled = False

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.file)
        output_path = str(input_path.parent / f"{input_path.stem}_judged.jsonl")

    if args.hybrid:
        results, benchmark_results = judge_benchmark_results_from_file_hybrid(
            args.file,
            models=args.hybrid,
            concurrency=args.concurrency,
            gold_answer_col=args.gold_answer_col,
            model_answer_col=args.model_answer_col,
        )
        judge_model_label = "+".join(args.hybrid)
    else:
        results, benchmark_results = judge_benchmark_results_from_file(
            args.file,
            model=args.model,
            concurrency=args.concurrency,
            gold_answer_col=args.gold_answer_col,
            model_answer_col=args.model_answer_col,
        )
        judge_model_label = args.model

    accuracy = calculate_accuracy(results)
    steps_values = [
        r.get("steps") for r in benchmark_results if r.get("steps") is not None
    ]
    tool_values = [
        r.get("total_tool_calls")
        for r in benchmark_results
        if r.get("total_tool_calls") is not None
    ]
    cost_parts = []
    if steps_values:
        cost_parts.append(f"avg_steps: {sum(steps_values) / len(steps_values):.2f}")
    if tool_values:
        cost_parts.append(f"avg_tool_calls: {sum(tool_values) / len(tool_values):.2f}")
    cost_str = f", {', '.join(cost_parts)}" if cost_parts else ""
    logger.info(
        "input: %s, correct: %.2f%%, incorrect: %.2f%%, no_answer: %.2f%%%s",
        args.file,
        accuracy["correct"] * 100,
        accuracy["incorrect"] * 100,
        accuracy["no_answer"] * 100,
        cost_str,
    )

    avg_tool_usage = _aggregate_tool_usage(benchmark_results)
    if avg_tool_usage:
        lines = ["Avg tool usage per item:"]
        for tool_name, avg in avg_tool_usage.items():
            lines.append(f"  {tool_name:20s} {avg:.2f}")
        logger.info("\n".join(lines))

    save_judged_results(benchmark_results, results, judge_model_label, output_path)
    write_summary(
        args.file, judge_model_label, accuracy, results, benchmark_results, output_path
    )
