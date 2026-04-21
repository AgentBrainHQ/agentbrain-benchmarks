"""
Evaluate LongMemEval results using LLM judge.
Compares Brain (System A) vs Baseline (System B).

Output: results/eval_report.json

Usage:
  python evaluate.py                          # Evaluate both systems
  python evaluate.py --system brain           # Brain only
  python evaluate.py --system baseline        # Baseline only
"""

import argparse
import asyncio
import json
import os
import time
from collections import defaultdict

import aiohttp

from config import (
    OPENAI_URL, OPENAI_KEY, JUDGE_MODEL,
    DATA_DIR, RESULTS_DIR, MAX_CONCURRENT_LLM,
)

QUERIES_FILE = os.path.join(DATA_DIR, "queries.jsonl")
BRAIN_OUTPUT = os.path.join(RESULTS_DIR, "brain_output.jsonl")
BASELINE_OUTPUT = os.path.join(RESULTS_DIR, "baseline_output.jsonl")
EVAL_REPORT = os.path.join(RESULTS_DIR, "eval_report.json")


def load_queries() -> dict[str, dict]:
    """Load queries indexed by question_id."""
    queries = {}
    with open(QUERIES_FILE) as f:
        for line in f:
            q = json.loads(line)
            queries[q["question_id"]] = q
    return queries


def load_outputs(path: str) -> dict[str, dict]:
    outputs = {}
    if not os.path.exists(path):
        return outputs
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            outputs[r["question_id"]] = r
    return outputs


JUDGE_PROMPT = """You are evaluating whether a system correctly answered a question about past conversations.

## Ground Truth Answer
{ground_truth}

## System's Answer
{hypothesis}

## Question
{question}

Rate the system's answer on correctness:
- CORRECT: The answer contains the key information from the ground truth
- PARTIAL: The answer is partially correct or contains some relevant info but misses key details
- WRONG: The answer is incorrect, irrelevant, or says it doesn't know when the info was available
- ABSTAIN_CORRECT: The system correctly abstained (said it doesn't know) when the ground truth is also "not available" or similar

Respond with ONLY one of: CORRECT, PARTIAL, WRONG, ABSTAIN_CORRECT"""


async def judge_answer(
    session: aiohttp.ClientSession,
    question: str,
    ground_truth: str,
    hypothesis: str,
    semaphore: asyncio.Semaphore,
) -> str:
    """Use LLM judge to score an answer."""
    async with semaphore:
        prompt = JUDGE_PROMPT.format(
            ground_truth=ground_truth,
            hypothesis=hypothesis,
            question=question,
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_KEY}",
        }
        payload = {
            "model": JUDGE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 20,
            "temperature": 0,
        }

        try:
            async with session.post(OPENAI_URL, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    verdict = data["choices"][0]["message"]["content"].strip().upper()
                    # Normalize
                    for v in ["CORRECT", "PARTIAL", "WRONG", "ABSTAIN_CORRECT"]:
                        if v in verdict:
                            return v
                    return "WRONG"  # default if unclear
                return "ERROR"
        except Exception as e:
            print(f"  Judge error: {e}")
            return "ERROR"


async def evaluate_system(
    session: aiohttp.ClientSession,
    system_name: str,
    outputs: dict[str, dict],
    queries: dict[str, dict],
    semaphore: asyncio.Semaphore,
) -> dict:
    """Evaluate all outputs for a system."""
    print(f"\nEvaluating {system_name}: {len(outputs)} answers...")

    tasks = []
    question_ids = []
    for qid, output in outputs.items():
        if qid not in queries:
            continue
        query = queries[qid]
        tasks.append(judge_answer(
            session, query["question"], query["answer"], output["hypothesis"], semaphore
        ))
        question_ids.append(qid)

    verdicts = await asyncio.gather(*tasks)

    # Aggregate
    counts = defaultdict(int)
    per_query = {}
    for qid, verdict in zip(question_ids, verdicts):
        counts[verdict] += 1
        per_query[qid] = verdict

    total = len(verdicts)
    correct = counts.get("CORRECT", 0) + counts.get("ABSTAIN_CORRECT", 0)
    partial = counts.get("PARTIAL", 0)

    accuracy = correct / max(total, 1)
    accuracy_with_partial = (correct + partial * 0.5) / max(total, 1)

    return {
        "system": system_name,
        "total": total,
        "counts": dict(counts),
        "accuracy": round(accuracy * 100, 1),
        "accuracy_with_partial": round(accuracy_with_partial * 100, 1),
        "per_query": per_query,
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["brain", "baseline", "both"], default="both")
    args = parser.parse_args()

    queries = load_queries()
    print(f"Loaded {len(queries)} ground-truth queries")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM)
    start = time.time()

    report = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "systems": []}

    async with aiohttp.ClientSession() as session:
        if args.system in ("brain", "both"):
            brain_outputs = load_outputs(BRAIN_OUTPUT)
            if brain_outputs:
                result = await evaluate_system(session, "brain", brain_outputs, queries, semaphore)
                report["systems"].append(result)
            else:
                print(f"WARNING: {BRAIN_OUTPUT} not found or empty")

        if args.system in ("baseline", "both"):
            baseline_outputs = load_outputs(BASELINE_OUTPUT)
            if baseline_outputs:
                result = await evaluate_system(session, "baseline", baseline_outputs, queries, semaphore)
                report["systems"].append(result)
            else:
                print(f"WARNING: {BASELINE_OUTPUT} not found or empty")

    # Print comparison
    print("\n" + "=" * 60)
    print("LONGMEMEVAL BENCHMARK RESULTS")
    print("=" * 60)

    for sys_result in report["systems"]:
        name = sys_result["system"].upper()
        print(f"\n{name}:")
        print(f"  Total queries:       {sys_result['total']}")
        print(f"  Correct:             {sys_result['counts'].get('CORRECT', 0)}")
        print(f"  Abstain Correct:     {sys_result['counts'].get('ABSTAIN_CORRECT', 0)}")
        print(f"  Partial:             {sys_result['counts'].get('PARTIAL', 0)}")
        print(f"  Wrong:               {sys_result['counts'].get('WRONG', 0)}")
        print(f"  Errors:              {sys_result['counts'].get('ERROR', 0)}")
        print(f"  Accuracy:            {sys_result['accuracy']}%")
        print(f"  Accuracy (w/partial): {sys_result['accuracy_with_partial']}%")

    if len(report["systems"]) == 2:
        brain = report["systems"][0]
        base = report["systems"][1]
        delta = brain["accuracy"] - base["accuracy"]
        print(f"\n  DELTA (Brain - Baseline): {delta:+.1f}%")
        if delta > 0:
            print(f"  Brain wins by {delta:.1f} percentage points")
        elif delta < 0:
            print(f"  Baseline wins by {abs(delta):.1f} percentage points")
        else:
            print(f"  TIE")

    # Save report
    with open(EVAL_REPORT, "w") as f:
        json.dump(report, f, indent=2)

    elapsed = time.time() - start
    print(f"\nEvaluation time: {elapsed:.1f}s")
    print(f"Full report: {EVAL_REPORT}")


if __name__ == "__main__":
    asyncio.run(main())
