"""
Run LongMemEval queries against Brain API.
For each question: brain_recall → build context → LLM answer.

Output: results/brain_output.jsonl (question_id + hypothesis)

Usage:
  python query.py                      # All 500 queries
  python query.py --limit 10           # First 10 queries
  python query.py --tenant 7161e7e2    # Single tenant only
"""

import argparse
import asyncio
import json
import os
import time

import aiohttp

from config import (
    BRAIN_API_URL, OPENROUTER_URL, OPENROUTER_KEY,
    DATA_DIR, RESULTS_DIR, RECALL_LIMIT, ANSWER_MODEL,
    MAX_CONCURRENT_RECALLS, MAX_CONCURRENT_LLM,
)

QUERIES_FILE = os.path.join(DATA_DIR, "queries.jsonl")
WORKSPACES_FILE = os.path.join(DATA_DIR, "workspaces.json")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "brain_output.jsonl")


def load_queries(limit: int = 0, tenant_filter: str = "") -> list[dict]:
    queries = []
    with open(QUERIES_FILE) as f:
        for line in f:
            q = json.loads(line)
            if tenant_filter and q["tenant_id"] != tenant_filter:
                continue
            queries.append(q)
            if limit and len(queries) >= limit:
                break
    return queries


def load_workspaces() -> dict:
    with open(WORKSPACES_FILE) as f:
        return json.load(f)


async def brain_recall(
    session: aiohttp.ClientSession,
    api_key: str,
    workspace_id: str,
    query: str,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """Recall memories from Brain API."""
    async with semaphore:
        url = f"{BRAIN_API_URL}/memory/recall"
        headers = {"Content-Type": "application/json", "X-API-Key": api_key}
        payload = {"workspace_id": workspace_id, "query": query, "limit": RECALL_LIMIT}

        try:
            async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("memories", [])
                return []
        except Exception as e:
            print(f"  Recall error: {e}")
            return []


async def generate_answer(
    session: aiohttp.ClientSession,
    question: str,
    memories: list[dict],
    semaphore: asyncio.Semaphore,
) -> str:
    """Generate answer using LLM with recalled memories as context."""
    async with semaphore:
        # Build context from memories
        context_parts = []
        for i, mem in enumerate(memories, 1):
            content = mem.get("content", "")
            relevance = mem.get("relevance", 0)
            context_parts.append(f"[Memory {i} (relevance: {relevance:.2f})]:\n{content}")

        context = "\n\n".join(context_parts) if context_parts else "(No relevant memories found)"

        system_prompt = (
            "You are answering questions based on your memory of past conversations. "
            "Use ONLY the provided memories to answer. Be specific and concise. "
            "If the memories don't contain the answer, say 'I don't have that information in my memory.'"
        )

        user_prompt = f"## Recalled Memories\n\n{context}\n\n## Question\n\n{question}\n\n## Answer"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENROUTER_KEY}",
        }
        payload = {
            "model": ANSWER_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 500,
            "temperature": 0,
        }

        try:
            async with session.post(OPENROUTER_URL, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    text = await resp.text()
                    print(f"  LLM error: {resp.status} {text[:200]}")
                    return "ERROR: Failed to generate answer"
        except Exception as e:
            print(f"  LLM error: {e}")
            return "ERROR: Failed to generate answer"


async def process_query(
    session: aiohttp.ClientSession,
    query: dict,
    workspace: dict,
    recall_sem: asyncio.Semaphore,
    llm_sem: asyncio.Semaphore,
) -> dict:
    """Process a single query: recall → generate answer."""
    api_key = workspace["api_key"]
    ws_id = workspace["workspace_id"]

    # Step 1: Recall
    memories = await brain_recall(session, api_key, ws_id, query["question"], recall_sem)

    # Step 2: Generate answer
    answer = await generate_answer(session, query["question"], memories, llm_sem)

    return {
        "question_id": query["question_id"],
        "hypothesis": answer,
        "num_memories_recalled": len(memories),
        "tenant_id": query["tenant_id"],
    }


async def main():
    parser = argparse.ArgumentParser(description="Run LongMemEval queries against Brain")
    parser.add_argument("--limit", type=int, default=0, help="Max queries to run")
    parser.add_argument("--tenant", type=str, help="Single tenant_id")
    args = parser.parse_args()

    if not os.path.exists(QUERIES_FILE):
        print(f"ERROR: {QUERIES_FILE} not found. Run download_data.py first.")
        return
    if not os.path.exists(WORKSPACES_FILE):
        print(f"ERROR: {WORKSPACES_FILE} not found. Run ingest.py first.")
        return

    queries = load_queries(limit=args.limit, tenant_filter=args.tenant or "")
    workspaces = load_workspaces()

    # Filter queries to tenants we have workspaces for
    valid_queries = [q for q in queries if q["tenant_id"] in workspaces]
    print(f"Running {len(valid_queries)} queries ({len(queries) - len(valid_queries)} skipped, no workspace)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    recall_sem = asyncio.Semaphore(MAX_CONCURRENT_RECALLS)
    llm_sem = asyncio.Semaphore(MAX_CONCURRENT_LLM)

    start = time.time()

    async with aiohttp.ClientSession() as session:
        # Process in batches of 20
        batch_size = 20
        results = []

        for i in range(0, len(valid_queries), batch_size):
            batch = valid_queries[i:i + batch_size]
            tasks = [
                process_query(session, q, workspaces[q["tenant_id"]], recall_sem, llm_sem)
                for q in batch
            ]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            print(f"  Processed {len(results)}/{len(valid_queries)} queries", end="\r")

    # Write output
    with open(OUTPUT_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    elapsed = time.time() - start
    avg_memories = sum(r["num_memories_recalled"] for r in results) / max(len(results), 1)
    errors = sum(1 for r in results if r["hypothesis"].startswith("ERROR"))

    print(f"\n=== QUERY COMPLETE ===")
    print(f"Queries: {len(results)}")
    print(f"Avg memories recalled: {avg_memories:.1f}")
    print(f"Errors: {errors}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
