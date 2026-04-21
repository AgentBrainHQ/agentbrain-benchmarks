"""
Full Benchmark Run — Direct Ingest + Query + Evaluate for all 500 tenants.
Bypasses Brain API for ingest (local embeddings + Supabase REST).
Uses Brain API for recall (Dream Cycle) and direct pgvector for baseline.

Usage:
  python run_full.py                    # Full 500-tenant run
  python run_full.py --limit 10         # First 10 tenants only
  python run_full.py --skip-ingest      # Skip ingest, only query+eval
"""

import argparse
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime

import requests
from sentence_transformers import SentenceTransformer

from config import (
    BRAIN_API_URL, BRAIN_DB_URL, BRAIN_DB_SERVICE_KEY,
    OPENROUTER_URL, OPENROUTER_KEY,
    DATA_DIR, RESULTS_DIR, RECALL_LIMIT, ANSWER_MODEL,
    OPENAI_KEY, OPENAI_URL, JUDGE_MODEL,
)

DOCS_FILE = os.path.join(DATA_DIR, "docs.jsonl")
QUERIES_FILE = os.path.join(DATA_DIR, "queries.jsonl")
WORKSPACES_FILE = os.path.join(DATA_DIR, "workspaces.json")
BATCH_SIZE = 10

# Supabase REST headers
SB_HEADERS = {
    "apikey": BRAIN_DB_SERVICE_KEY,
    "Authorization": f"Bearer {BRAIN_DB_SERVICE_KEY}",
    "Content-Type": "application/json",
}

print("Loading embedding model...", flush=True)
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("Model loaded.", flush=True)


# ============================================================
# DATA LOADING
# ============================================================

def load_docs_by_tenant(limit_tenants: int = 0) -> dict:
    """Load docs grouped by tenant_id."""
    tenants = defaultdict(list)
    with open(DOCS_FILE) as f:
        for line in f:
            doc = json.loads(line)
            tenants[doc["tenant_id"]].append(doc)
    # Sort by tenant_id for reproducibility
    result = dict(sorted(tenants.items()))
    if limit_tenants:
        result = dict(list(result.items())[:limit_tenants])
    return result


def load_queries() -> list[dict]:
    queries = []
    with open(QUERIES_FILE) as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def load_workspaces() -> dict:
    if os.path.exists(WORKSPACES_FILE):
        with open(WORKSPACES_FILE) as f:
            return json.load(f)
    return {}


def save_workspaces(ws: dict):
    with open(WORKSPACES_FILE, "w") as f:
        json.dump(ws, f, indent=2)


# ============================================================
# PHASE 1: CREATE WORKSPACES
# ============================================================

def create_workspaces(tenant_ids: list[str], existing: dict) -> dict:
    """Create Brain workspaces for tenants that don't have one yet."""
    new_tenants = [t for t in tenant_ids if t not in existing]
    if not new_tenants:
        print(f"All {len(tenant_ids)} workspaces exist.", flush=True)
        return existing

    print(f"Creating {len(new_tenants)} new workspaces...", flush=True)
    url = f"{BRAIN_DB_URL}/rest/v1/workspaces"

    for i, tid in enumerate(new_tenants):
        workspace_id = str(uuid.uuid4())
        api_key = f"brain_bench_{tid}_{uuid.uuid4().hex[:16]}"

        payload = {
            "id": workspace_id,
            "name": f"bench-{tid}",
            "api_key": api_key,
        }
        headers = {**SB_HEADERS, "Prefer": "return=representation"}

        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        if resp.status_code in (200, 201):
            existing[tid] = {"workspace_id": workspace_id, "api_key": api_key}
        else:
            print(f"  FAILED {tid}: {resp.status_code} {resp.text[:100]}", flush=True)

        if (i + 1) % 50 == 0:
            print(f"  Created {i+1}/{len(new_tenants)} workspaces", flush=True)

    save_workspaces(existing)
    print(f"Workspaces ready: {len(existing)}", flush=True)
    return existing


# ============================================================
# PHASE 2: DIRECT INGEST
# ============================================================

def ingest_tenant(tenant_id: str, docs: list[dict], workspace_id: str) -> dict:
    """Ingest all docs for a tenant via direct Supabase REST insert."""
    url = f"{BRAIN_DB_URL}/rest/v1/memories"
    now = datetime.utcnow().isoformat()
    total_inserted = 0

    for i in range(0, len(docs), BATCH_SIZE):
        batch_docs = docs[i:i + BATCH_SIZE]

        # Build content strings and embed locally
        contents = []
        for doc in batch_docs:
            content = f"[Session: {doc['session_id']} | Date: {doc['session_date']}]\n{doc['session_text']}"
            contents.append(content)

        embeddings = [e.tolist() for e in model.encode(contents, show_progress_bar=False)]

        # Build rows
        memories = []
        for doc, content, embedding in zip(batch_docs, contents, embeddings):
            memories.append({
                "id": str(uuid.uuid4()),
                "workspace_id": workspace_id,
                "content": content,
                "type": "episodic",
                "embedding": str(embedding),
                "emotion_score": 0.0,
                "novelty_score": 0.5,
                "urgency_score": 0.0,
                "source_trust": 0.9,
                "weight": 0.6,
                "decay_rate": 0.03,
                "access_count": 0,
                "stability": 1.0,
                "difficulty": 0.3,
                "reps": 0,
                "created_at": now,
            })

        headers = {**SB_HEADERS, "Prefer": "return=minimal"}

        # Retry with backoff on timeout
        for attempt in range(3):
            try:
                resp = requests.post(url, json=memories, headers=headers, timeout=60)
                if resp.status_code in (200, 201):
                    total_inserted += len(memories)
                    break
                elif resp.status_code == 500 and "57014" in resp.text:
                    # Statement timeout — wait and retry
                    wait = 2 ** attempt * 2
                    print(f"    Timeout, retry {attempt+1}/3 in {wait}s...", flush=True)
                    time.sleep(wait)
                else:
                    print(f"    Insert error: {resp.status_code} {resp.text[:100]}", flush=True)
                    break
            except requests.exceptions.Timeout:
                wait = 2 ** attempt * 2
                print(f"    Request timeout, retry {attempt+1}/3 in {wait}s...", flush=True)
                time.sleep(wait)

    return {"total": len(docs), "inserted": total_inserted}


def run_ingest(tenants: dict, workspaces: dict):
    """Ingest all tenants."""
    print(f"\n{'='*60}", flush=True)
    print(f"PHASE 2: INGEST ({len(tenants)} tenants, {sum(len(d) for d in tenants.values()):,} docs)", flush=True)
    print(f"{'='*60}", flush=True)

    start = time.time()
    total_inserted = 0
    total_docs = 0
    skipped = 0

    for i, (tid, docs) in enumerate(tenants.items()):
        if tid not in workspaces:
            continue

        ws_id = workspaces[tid]["workspace_id"]

        # Check if already ingested (skip if memories exist)
        count_url = f"{BRAIN_DB_URL}/rest/v1/memories?workspace_id=eq.{ws_id}&select=id&limit=1"
        count_headers = {**SB_HEADERS, "Prefer": "count=estimated", "Range": "0-0"}
        try:
            cr = requests.get(count_url, headers=count_headers, timeout=5)
            cr_range = cr.headers.get("content-range", "")
            if "/" in cr_range:
                existing = int(cr_range.split("/")[1])
            else:
                existing = 0
            if existing >= len(docs) * 0.8:  # 80% threshold
                skipped += 1
                total_docs += len(docs)
                total_inserted += existing
                if (i + 1) % 50 == 0 or (i + 1) == 10:
                    print(f"  [{i+1}/{len(tenants)}] skipped {skipped} already-ingested tenants", flush=True)
                continue
        except Exception:
            pass

        print(f"  Ingesting tenant {i+1} ({tid}): {len(docs)} docs...", flush=True)
        time.sleep(0.5)  # Brief pause before ingest to let connection pool settle
        result = ingest_tenant(tid, docs, ws_id)
        total_inserted += result["inserted"]
        total_docs += result["total"]

        elapsed = time.time() - start
        rate = total_inserted / elapsed if elapsed > 0 else 0

        if (i + 1) % 10 == 0 or i == len(tenants) - 1:
            print(f"  [{i+1}/{len(tenants)}] {total_inserted:,}/{total_docs:,} docs "
                  f"({rate:.0f}/s, {elapsed:.0f}s elapsed)", flush=True)

    elapsed = time.time() - start
    print(f"\nIngest complete: {total_inserted:,}/{total_docs:,} in {elapsed:.0f}s "
          f"({total_inserted/elapsed:.0f} docs/s)", flush=True)


# ============================================================
# PHASE 3: QUERY (Brain + Baseline)
# ============================================================

def vector_recall_local(workspace_id: str, query: str) -> list[dict]:
    """Direct pgvector cosine similarity — local fallback."""
    import numpy as np

    query_embedding = model.encode(query).tolist()

    url = f"{BRAIN_DB_URL}/rest/v1/memories?workspace_id=eq.{workspace_id}&select=id,content,embedding"
    resp = requests.get(url, headers=SB_HEADERS, timeout=30)
    if resp.status_code != 200:
        return []

    memories = resp.json()
    if not memories:
        return []

    query_vec = np.array(query_embedding)
    results = []
    for mem in memories:
        if not mem.get("embedding"):
            continue
        mem_vec = np.array(json.loads(mem["embedding"]))
        sim = float(np.dot(query_vec, mem_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(mem_vec)))
        results.append({"content": mem["content"], "similarity": sim})

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:RECALL_LIMIT]


def brain_recall(api_key: str, workspace_id: str, query: str) -> list[dict]:
    """Recall via Brain API (Dream Cycle enriched)."""
    url = f"{BRAIN_API_URL}/memory/recall"
    headers = {"Content-Type": "application/json", "X-API-Key": api_key}
    payload = {"workspace_id": workspace_id, "query": query, "limit": RECALL_LIMIT}

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("memories", [])
    except Exception:
        pass
    return []


def generate_answer(question: str, memories: list[dict]) -> str:
    """Generate answer using LLM."""
    context_parts = []
    for i, mem in enumerate(memories, 1):
        content = mem.get("content", "")
        score = mem.get("similarity", mem.get("relevance", 0))
        context_parts.append(f"[Memory {i} (score: {score:.2f})]:\n{content}")

    context = "\n\n".join(context_parts) if context_parts else "(No relevant memories found)"

    system_prompt = (
        "You are answering questions based on your memory of past conversations. "
        "Use ONLY the provided memories to answer. Be specific and concise. "
        "If the memories don't contain the answer, say 'I don't have that information in my memory.'"
    )
    user_prompt = f"## Recalled Memories\n\n{context}\n\n## Question\n\n{question}\n\n## Answer"

    # Use OpenAI direct for GPT models, OpenRouter for others
    if ANSWER_MODEL.startswith("gpt") or ANSWER_MODEL.startswith("openai/"):
        model_name = ANSWER_MODEL.replace("openai/", "")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_KEY}"}
        url = OPENAI_URL
    else:
        model_name = ANSWER_MODEL
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {OPENROUTER_KEY}"}
        url = OPENROUTER_URL

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 500,
        "temperature": 0,
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        else:
            return f"ERROR: LLM failed ({resp.status_code})"
    except Exception as e:
        return f"ERROR: {e}"


def run_queries(queries: list[dict], workspaces: dict):
    """Run both Brain and Baseline queries."""
    valid = [q for q in queries if q["tenant_id"] in workspaces]
    print(f"\n{'='*60}", flush=True)
    print(f"PHASE 3: QUERY ({len(valid)} queries)", flush=True)
    print(f"{'='*60}", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    brain_results = []
    baseline_results = []

    start = time.time()

    for i, q in enumerate(valid):
        ws = workspaces[q["tenant_id"]]
        question = q["question"]

        # Brain recall
        brain_mems = brain_recall(ws["api_key"], ws["workspace_id"], question)
        brain_answer = generate_answer(question, brain_mems)
        brain_results.append({
            "question_id": q["question_id"],
            "hypothesis": brain_answer,
            "num_memories_recalled": len(brain_mems),
            "tenant_id": q["tenant_id"],
        })

        # Baseline recall
        baseline_mems = vector_recall_local(ws["workspace_id"], question)
        baseline_answer = generate_answer(question, baseline_mems)
        baseline_results.append({
            "question_id": q["question_id"],
            "hypothesis": baseline_answer,
            "num_memories_recalled": len(baseline_mems),
            "tenant_id": q["tenant_id"],
        })

        if (i + 1) % 10 == 0 or i == len(valid) - 1:
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(valid)}] Brain: {len(brain_mems)} mems, "
                  f"Baseline: {len(baseline_mems)} mems ({elapsed:.0f}s)", flush=True)

    # Save results
    brain_file = os.path.join(RESULTS_DIR, "brain_output.jsonl")
    baseline_file = os.path.join(RESULTS_DIR, "baseline_output.jsonl")

    with open(brain_file, "w") as f:
        for r in brain_results:
            f.write(json.dumps(r) + "\n")
    with open(baseline_file, "w") as f:
        for r in baseline_results:
            f.write(json.dumps(r) + "\n")

    elapsed = time.time() - start
    print(f"\nQueries complete in {elapsed:.0f}s", flush=True)
    print(f"  Brain avg memories: {sum(r['num_memories_recalled'] for r in brain_results)/max(len(brain_results),1):.1f}", flush=True)
    print(f"  Baseline avg memories: {sum(r['num_memories_recalled'] for r in baseline_results)/max(len(baseline_results),1):.1f}", flush=True)


# ============================================================
# PHASE 4: EVALUATE
# ============================================================

def judge_answer(question: str, ground_truth: str, hypothesis: str) -> str:
    """Use GPT-4o to judge answer quality."""
    system = (
        "You are a judge evaluating whether a hypothesis answer is correct given a ground-truth answer. "
        "Respond with exactly one word: CORRECT, PARTIAL, WRONG, or ABSTAIN_CORRECT.\n"
        "- CORRECT: hypothesis conveys the same information as ground truth\n"
        "- PARTIAL: hypothesis is partially correct but missing key details\n"
        "- WRONG: hypothesis is incorrect or irrelevant\n"
        "- ABSTAIN_CORRECT: hypothesis correctly states it doesn't know (and the question is unanswerable)"
    )
    user = f"Question: {question}\nGround Truth: {ground_truth}\nHypothesis: {hypothesis}\n\nVerdict:"

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_KEY}"}
    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": 10,
        "temperature": 0,
    }

    try:
        resp = requests.post(OPENAI_URL, json=payload, headers=headers, timeout=30)
        if resp.status_code == 200:
            verdict = resp.json()["choices"][0]["message"]["content"].strip().upper()
            for valid in ["CORRECT", "PARTIAL", "WRONG", "ABSTAIN_CORRECT"]:
                if valid in verdict:
                    return valid
            return "WRONG"
        return "ERROR"
    except Exception:
        return "ERROR"


def run_evaluate(queries: list[dict]):
    """Evaluate Brain vs Baseline results."""
    print(f"\n{'='*60}", flush=True)
    print(f"PHASE 4: EVALUATE (GPT-4o Judge)", flush=True)
    print(f"{'='*60}", flush=True)

    # Load results
    brain_file = os.path.join(RESULTS_DIR, "brain_output.jsonl")
    baseline_file = os.path.join(RESULTS_DIR, "baseline_output.jsonl")

    brain_answers = {}
    with open(brain_file) as f:
        for line in f:
            r = json.loads(line)
            brain_answers[r["question_id"]] = r["hypothesis"]

    baseline_answers = {}
    with open(baseline_file) as f:
        for line in f:
            r = json.loads(line)
            baseline_answers[r["question_id"]] = r["hypothesis"]

    # Build ground truth lookup
    gt = {q["question_id"]: q for q in queries}

    start = time.time()
    brain_verdicts = {}
    baseline_verdicts = {}

    evaluated = 0
    for qid in brain_answers:
        if qid not in gt:
            continue

        question = gt[qid]["question"]
        ground_truth = gt[qid]["answer"]

        brain_verdicts[qid] = judge_answer(question, ground_truth, brain_answers[qid])
        baseline_verdicts[qid] = judge_answer(question, ground_truth, baseline_answers.get(qid, "NO ANSWER"))

        evaluated += 1
        if evaluated % 20 == 0:
            print(f"  Evaluated {evaluated}/{len(brain_answers)}", flush=True)

    # Compute metrics
    def metrics(verdicts):
        total = len(verdicts)
        correct = sum(1 for v in verdicts.values() if v == "CORRECT")
        partial = sum(1 for v in verdicts.values() if v == "PARTIAL")
        wrong = sum(1 for v in verdicts.values() if v == "WRONG")
        abstain = sum(1 for v in verdicts.values() if v == "ABSTAIN_CORRECT")
        errors = sum(1 for v in verdicts.values() if v == "ERROR")
        acc = (correct + abstain) / total * 100 if total else 0
        acc_partial = (correct + abstain + 0.5 * partial) / total * 100 if total else 0
        return {
            "total": total, "correct": correct, "partial": partial,
            "wrong": wrong, "abstain_correct": abstain, "errors": errors,
            "accuracy": round(acc, 1), "accuracy_with_partial": round(acc_partial, 1),
        }

    brain_m = metrics(brain_verdicts)
    baseline_m = metrics(baseline_verdicts)
    delta = brain_m["accuracy"] - baseline_m["accuracy"]

    elapsed = time.time() - start

    # Print results
    print(f"\n{'='*60}", flush=True)
    print(f"LONGMEMEVAL BENCHMARK RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"\nBRAIN (Dream Cycle):", flush=True)
    for k, v in brain_m.items():
        print(f"  {k:25s}: {v}", flush=True)
    print(f"\nBASELINE (pgvector only):", flush=True)
    for k, v in baseline_m.items():
        print(f"  {k:25s}: {v}", flush=True)
    print(f"\n  DELTA (Brain - Baseline): {delta:+.1f}%", flush=True)
    if delta > 0:
        print(f"  >>> BRAIN WINS <<<", flush=True)
    elif delta < 0:
        print(f"  >>> BASELINE WINS <<<", flush=True)
    else:
        print(f"  >>> TIE <<<", flush=True)

    # Save report
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "brain": brain_m,
        "baseline": baseline_m,
        "delta": delta,
        "brain_verdicts": brain_verdicts,
        "baseline_verdicts": baseline_verdicts,
        "eval_time_s": round(elapsed, 1),
    }
    report_file = os.path.join(RESULTS_DIR, "eval_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report: {report_file}", flush=True)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Full LongMemEval Benchmark")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of tenants")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingest phase")
    args = parser.parse_args()

    # Load data
    queries = load_queries()
    query_tenant_ids = set(q["tenant_id"] for q in queries)

    if not args.skip_ingest:
        tenants = load_docs_by_tenant(limit_tenants=args.limit)
        # Only process tenants that have queries
        tenants = {tid: docs for tid, docs in tenants.items() if tid in query_tenant_ids}
        print(f"Dataset: {len(tenants)} tenants, {sum(len(d) for d in tenants.values()):,} docs, "
              f"{len(queries)} queries", flush=True)
    else:
        tenants = {}
        print(f"Skipping ingest. {len(queries)} queries.", flush=True)

    workspaces = load_workspaces()

    # Phase 1: Create workspaces
    if not args.skip_ingest:
        tenant_ids = list(tenants.keys())
        workspaces = create_workspaces(tenant_ids, workspaces)

    # Phase 2: Ingest
    if not args.skip_ingest:
        run_ingest(tenants, workspaces)

    # Phase 3: Query (Brain + Baseline)
    run_queries(queries, workspaces)

    # Phase 4: Evaluate
    run_evaluate(queries)

    print(f"\n{'='*60}", flush=True)
    print(f"BENCHMARK COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
