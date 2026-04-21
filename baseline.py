"""
Baseline: Direct pgvector cosine similarity (no Dream Cycle).
Queries the Brain DB directly with embedding similarity search.

Output: results/baseline_output.jsonl

Usage:
  python baseline.py                   # All 500 queries
  python baseline.py --limit 10        # First 10
"""

import argparse
import asyncio
import json
import os
import time

import aiohttp

from config import (
    BRAIN_DB_URL, BRAIN_DB_SERVICE_KEY,
    OPENROUTER_URL, OPENROUTER_KEY,
    DATA_DIR, RESULTS_DIR, BASELINE_LIMIT, ANSWER_MODEL,
    MAX_CONCURRENT_RECALLS, MAX_CONCURRENT_LLM,
)

QUERIES_FILE = os.path.join(DATA_DIR, "queries.jsonl")
WORKSPACES_FILE = os.path.join(DATA_DIR, "workspaces.json")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "baseline_output.jsonl")

# We need to generate embeddings for queries to do cosine search
# Use Brain DB's built-in embedding function via RPC, or generate externally
EMBED_MODEL = "text-embedding-3-small"  # OpenAI via OpenRouter doesn't support embeddings
# We'll use Supabase's built-in pgvector match function instead


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


async def vector_search(
    session: aiohttp.ClientSession,
    workspace_id: str,
    query: str,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """
    Direct pgvector search via Supabase RPC.
    Uses the Brain DB's match_memories function (cosine similarity).
    Falls back to basic text search if RPC not available.
    """
    async with semaphore:
        # Try RPC match_memories first
        url = f"{BRAIN_DB_URL}/rest/v1/rpc/match_memories"
        headers = {
            "apikey": BRAIN_DB_SERVICE_KEY,
            "Authorization": f"Bearer {BRAIN_DB_SERVICE_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "p_workspace_id": workspace_id,
            "p_query": query,
            "p_limit": BASELINE_LIMIT,
        }

        try:
            async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data if isinstance(data, list) else []

                # Fallback: direct table query with text search (no vector, just ILIKE)
                url2 = f"{BRAIN_DB_URL}/rest/v1/memories?workspace_id=eq.{workspace_id}&content=ilike.*{query[:50]}*&limit={BASELINE_LIMIT}&order=created_at.desc"
                async with session.get(url2, headers=headers) as resp2:
                    if resp2.status == 200:
                        return await resp2.json()
                    return []
        except Exception as e:
            print(f"  Vector search error: {e}")
            return []


async def generate_answer(
    session: aiohttp.ClientSession,
    question: str,
    memories: list[dict],
    semaphore: asyncio.Semaphore,
) -> str:
    """Generate answer using LLM with retrieved memories."""
    async with semaphore:
        context_parts = []
        for i, mem in enumerate(memories, 1):
            content = mem.get("content", mem.get("text", ""))
            score = mem.get("similarity", mem.get("relevance", 0))
            context_parts.append(f"[Result {i} (score: {score:.2f})]:\n{content}")

        context = "\n\n".join(context_parts) if context_parts else "(No results found)"

        system_prompt = (
            "You are answering questions based on your memory of past conversations. "
            "Use ONLY the provided context to answer. Be specific and concise. "
            "If the context doesn't contain the answer, say 'I don't have that information in my memory.'"
        )

        user_prompt = f"## Retrieved Context\n\n{context}\n\n## Question\n\n{question}\n\n## Answer"

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
                return "ERROR: Failed to generate answer"
        except Exception as e:
            return f"ERROR: {e}"


async def process_query(session, query, workspace, recall_sem, llm_sem) -> dict:
    ws_id = workspace["workspace_id"]

    memories = await vector_search(session, ws_id, query["question"], recall_sem)
    answer = await generate_answer(session, query["question"], memories, llm_sem)

    return {
        "question_id": query["question_id"],
        "hypothesis": answer,
        "num_results": len(memories),
        "tenant_id": query["tenant_id"],
    }


async def main():
    parser = argparse.ArgumentParser(description="Baseline pgvector search")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tenant", type=str)
    args = parser.parse_args()

    queries = load_queries(limit=args.limit, tenant_filter=args.tenant or "")
    workspaces = load_workspaces()

    valid = [q for q in queries if q["tenant_id"] in workspaces]
    print(f"Running baseline for {len(valid)} queries")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    recall_sem = asyncio.Semaphore(MAX_CONCURRENT_RECALLS)
    llm_sem = asyncio.Semaphore(MAX_CONCURRENT_LLM)

    start = time.time()

    async with aiohttp.ClientSession() as session:
        batch_size = 20
        results = []
        for i in range(0, len(valid), batch_size):
            batch = valid[i:i + batch_size]
            tasks = [process_query(session, q, workspaces[q["tenant_id"]], recall_sem, llm_sem) for q in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            print(f"  {len(results)}/{len(valid)}", end="\r")

    with open(OUTPUT_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    elapsed = time.time() - start
    print(f"\n=== BASELINE COMPLETE ===")
    print(f"Queries: {len(results)}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
