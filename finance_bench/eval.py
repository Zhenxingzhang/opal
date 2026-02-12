import os
import hashlib
import asyncio
import json
from pathlib import Path
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

DEEPSEEK_MODEL = ""
DEEPSEEK_BASE_URL = ""
DEEPSEEK_API_KEY = ""

CACHE_DIR = Path(__file__).parent / ".eval_cache"
MAX_CONCURRENCY = 10


def _cache_key(model: str, prompt: str) -> str:
    raw = json.dumps({"model": model, "prompt": prompt}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def _sanitize_model_name(model: str) -> str:
    return model.replace("/", "_").replace("\\", "_") if model else "unknown"


def _cache_path(model: str, prompt: str) -> Path:
    return CACHE_DIR / _sanitize_model_name(model) / f"{_cache_key(model, prompt)}.json"


def _load_cache(model: str, prompt: str) -> str | None:
    cache_file = _cache_path(model, prompt)
    if cache_file.exists():
        entry = json.loads(cache_file.read_text())
        return entry["response"]
    return None


def _save_cache(model: str, prompt: str, response: str):
    path = _cache_path(model, prompt)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"model": model, "prompt": prompt, "response": response}
    path.write_text(json.dumps(entry))


async def get_completion_async(prompt, model=None, semaphore=None):
    cached = _load_cache(model, prompt)
    if cached is not None:
        return cached

    if model == DEEPSEEK_MODEL:
        openai_client = AsyncOpenAI(
            api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL
        )
    else:
        openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    sem = semaphore or asyncio.Semaphore(1)
    async with sem:
        try:
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
                    print(f"Attempt {attempt + 1} failed for model {model}: {e}")
                    await asyncio.sleep(1)
            return None
        finally:
            await openai_client.close()


async def check_answer_equivalence(
    answer, gold_answer, query=None, model="gpt-4o-2024-11-20", semaphore=None
):
    query_prompt = f"- Query: {query}" if query else ""

    prompt = f"""
    You are an expert evaluator for AI-generated responses to queries. Your task is to determine whether the AI-generated answer correctly answers the query based on the golden answer provided by a human expert.

    Numerical Accuracy:
    - Rounding differences should be **ignored** if they do not meaningfully change the conclusion.
    - You can allow some flexibility in accuracy. For example, 1.2 is considered similar to 1.23. Two numbers are considered similar if one can be rounded to the other.
    - Fractions, percentage, and numerics could be considered similar, for example: "11 of 14" is considered equivalent to "79%" and "0.79".

    Evaluation Criteria:
    - If the golden answer or any of its equivalence can be inferred or generated from the AI-generated answer, then the AI-generated answer is considered correct.
    - If any number, percentage, fraction, or figure in the golden answer is not present in the AI-generated answer, but can be inferred or generated from the AI-generated answer or implicitly exist in the AI-generated answer, then the AI-generated answer is considered correct.
    - The AI-generated answer is considered correct if it conveys the same or similar meaning, conclusion, or rationale as the golden answer.
    - If the AI-generated answer is a superset of the golden answer, it is also considered correct.
    - If the AI-generated answer provides a valid answer or reasonable interpretation compared to the golden answer, it is considered correct.
    - If the AI-generated answer contains subjective judgments or opinions, it is considered correct as long as they are reasonable and justifiable compared to the golden answer.

    - Otherwise, the AI-generated answer is incorrect.

    Inputs:
    {query_prompt}
    - AI-Generated Answer: {answer}
    - Golden Answer: {gold_answer}

    Your output should be ONLY a boolean value: `True` or `False`, nothing else.
    """

    response = await get_completion_async(prompt, model=model, semaphore=semaphore)

    return response is not None and "true" in response.lower()


async def judge_benchmark_results(
    benchmark_results, model="gpt-4o-2024-11-20", concurrency=MAX_CONCURRENCY
):
    semaphore = asyncio.Semaphore(concurrency)
    tasks = []
    for result in benchmark_results:
        query = result["question"]
        gold_answer = result["gold_answer"]
        answer = result["model_answer"]
        tasks.append(
            check_answer_equivalence(
                answer, gold_answer, query=query, model=model, semaphore=semaphore
            )
        )
    results = await tqdm.gather(*tasks, desc="Evaluating answers")
    return results


def judge_benchmark_results_from_file(
    json_file_path, model="gpt-4o-2024-11-20", concurrency=MAX_CONCURRENCY
):
    with open(json_file_path, "r") as f:
        if json_file_path.endswith(".jsonl"):
            benchmark_results = [json.loads(line) for line in f if line.strip()]
        else:
            benchmark_results = json.load(f)
    results = asyncio.run(
        judge_benchmark_results(benchmark_results, model=model, concurrency=concurrency)
    )
    wrong_indexes = [
        i
        for i, result in enumerate(results)
        if not result and benchmark_results[i]["label"] == "AL"
    ]
    print(f"Wrong indexes: {wrong_indexes}")
    return results


def judge_benchmark_results_from_file_hybrid(
    json_file_path,
    models=["gpt-4o-2024-11-20", "o1-mini", "o3-mini"],
    concurrency=MAX_CONCURRENCY,
):
    if not models:
        raise ValueError("No models provided for hybrid evaluation.")

    hybrid_results = {}
    for model in models:
        print(f"\nJudging results using model '{model}'...")
        results = judge_benchmark_results_from_file(
            json_file_path, model=model, concurrency=concurrency
        )
        hybrid_results[model] = results

    combined_results = [
        any(hybrid_results[model][i] for model in models)
        for i in range(len(hybrid_results[models[0]]))
    ]

    return combined_results


def calculate_accuracy(results: list[bool]) -> float:
    """Calculate accuracy from a list of boolean values.

    Args:
        results: A list of True/False values where True indicates a correct answer.

    Returns:
        Accuracy as a float between 0.0 and 1.0.

    Raises:
        ValueError: If the results list is empty.
    """
    if not results:
        raise ValueError("Results list is empty.")
    return sum(results) / len(results)


def clear_cache():
    """Remove all cached LLM responses."""
    if CACHE_DIR.exists():
        import shutil

        for child in CACHE_DIR.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        print(f"Cache cleared: {CACHE_DIR}")
    else:
        print("No cache directory found.")


if __name__ == "__main__":
    import argparse

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
    args = parser.parse_args()

    if args.clear_cache:
        clear_cache()
        exit(0)

    if args.no_cache:
        # Point cache dir to a temp location that won't persist
        import tempfile

        CACHE_DIR = Path(tempfile.mkdtemp())

    if args.hybrid:
        results = judge_benchmark_results_from_file_hybrid(
            args.file, models=args.hybrid, concurrency=args.concurrency
        )
    else:
        results = judge_benchmark_results_from_file(
            args.file, model=args.model, concurrency=args.concurrency
        )

    accuracy = calculate_accuracy(results)
    print(f"input: {args.file}, results: {accuracy}")
