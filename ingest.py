"""
Ingest LongMemEval sessions into Brain API.
Creates one workspace per tenant_id, stores all their sessions.

Usage:
  python ingest.py                    # Full ingestion
  python ingest.py --tenant 7161e7e2  # Single tenant only
  python ingest.py --limit 100        # First 100 docs only
  python ingest.py --dry-run          # Count tenants/sessions without storing
"""

import argparse
import asyncio
import json
import os
import time
from collections import defaultdict

import aiohttp

from config import (
    BRAIN_API_URL, BRAIN_DB_URL, BRAIN_DB_SERVICE_KEY,
    DATA_DIR, MAX_CONCURRENT_STORES, MEMORY_TYPE, SOURCE_TRUST,
)

DOCS_FILE = os.path.join(DATA_DIR, "docs.jsonl")
WORKSPACES_FILE = os.path.join(DATA_DIR, "workspaces.json")


def load_docs(limit: int = 0, tenant_filter: str = "") -> dict[str, list[dict]]:
    """Load docs grouped by tenant_id."""
    tenants: dict[str, list[dict]] = defaultdict(list)
    count = 0
    with open(DOCS_FILE) as f:
        for line in f:
            doc = json.loads(line)
            tid = doc["tenant_id"]
            if tenant_filter and tid != tenant_filter:
                continue
            tenants[tid].append(doc)
            count += 1
            if limit and count >= limit:
                break
    return dict(tenants)


def load_workspaces() -> dict:
    """Load existing workspace mappings."""
    if os.path.exists(WORKSPACES_FILE):
        with open(WORKSPACES_FILE) as f:
            return json.load(f)
    return {}


def save_workspaces(ws: dict):
    with open(WORKSPACES_FILE, "w") as f:
        json.dump(ws, f, indent=2)


async def create_workspace(session: aiohttp.ClientSession, tenant_id: str) -> dict:
    """Create a Brain workspace for a benchmark tenant via Supabase direct insert."""
    import uuid
    workspace_id = str(uuid.uuid4())
    api_key = f"brain_bench_{tenant_id}_{uuid.uuid4().hex[:16]}"

    # Insert workspace directly into Brain DB
    url = f"{BRAIN_DB_URL}/rest/v1/workspaces"
    headers = {
        "apikey": BRAIN_DB_SERVICE_KEY,
        "Authorization": f"Bearer {BRAIN_DB_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    payload = {
        "id": workspace_id,
        "name": f"bench-{tenant_id}",
        "api_key": api_key,
    }

    async with session.post(url, json=payload, headers=headers) as resp:
        if resp.status in (200, 201):
            data = await resp.json()
            return {"workspace_id": workspace_id, "api_key": api_key}
        else:
            text = await resp.text()
            raise Exception(f"Failed to create workspace for {tenant_id}: {resp.status} {text}")


async def store_memory(
    session: aiohttp.ClientSession,
    api_key: str,
    workspace_id: str,
    content: str,
    semaphore: asyncio.Semaphore,
) -> bool:
    """Store a single memory via Brain API."""
    async with semaphore:
        url = f"{BRAIN_API_URL}/memory/store"
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key,
        }
        payload = {
            "workspace_id": workspace_id,
            "content": content,
            "memory_type": MEMORY_TYPE,
            "source_trust": SOURCE_TRUST,
        }

        for attempt in range(3):
            try:
                async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    if resp.status == 200:
                        return True
                    elif resp.status == 502:
                        # Server overloaded, wait and retry
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        text = await resp.text()
                        if attempt == 0:
                            print(f"  Store error {resp.status}: {text[:100]}")
                        return False
            except Exception as e:
                if attempt == 2:
                    print(f"  Store error: {e}")
                await asyncio.sleep(2 ** attempt)
        return False


async def ingest_tenant(
    session: aiohttp.ClientSession,
    tenant_id: str,
    docs: list[dict],
    workspace: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Ingest all sessions for a tenant in small batches."""
    api_key = workspace["api_key"]
    ws_id = workspace["workspace_id"]
    batch_size = MAX_CONCURRENT_STORES  # process this many at a time

    success = 0
    failed = 0

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        tasks = []
        for doc in batch:
            content = f"[Session: {doc['session_id']} | Date: {doc['session_date']}]\n{doc['session_text']}"
            tasks.append(store_memory(session, api_key, ws_id, content, semaphore))

        results = await asyncio.gather(*tasks)
        success += sum(1 for r in results if r)
        failed += sum(1 for r in results if not r)

        if (i + batch_size) % 50 < batch_size:
            print(f"    Progress: {min(i + batch_size, len(docs))}/{len(docs)} ({success} ok, {failed} err)", flush=True)

    return {"tenant_id": tenant_id, "total": len(docs), "success": success, "failed": failed}


async def main():
    parser = argparse.ArgumentParser(description="Ingest LongMemEval into Brain")
    parser.add_argument("--tenant", type=str, help="Single tenant_id to process")
    parser.add_argument("--limit", type=int, default=0, help="Max docs to load")
    parser.add_argument("--dry-run", action="store_true", help="Just count, don't store")
    args = parser.parse_args()

    if not os.path.exists(DOCS_FILE):
        print(f"ERROR: {DOCS_FILE} not found. Run download_data.py first.")
        return

    print("Loading docs...")
    tenants = load_docs(limit=args.limit, tenant_filter=args.tenant or "")

    total_docs = sum(len(d) for d in tenants.values())
    print(f"Loaded {total_docs:,} docs across {len(tenants)} tenants")

    if args.dry_run:
        for tid, docs in sorted(tenants.items(), key=lambda x: -len(x[1]))[:20]:
            print(f"  {tid}: {len(docs):,} sessions")
        return

    # Load or create workspaces
    workspaces = load_workspaces()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_STORES)

    start = time.time()

    async with aiohttp.ClientSession() as session:
        # Create missing workspaces
        new_tenants = [tid for tid in tenants if tid not in workspaces]
        if new_tenants:
            print(f"Creating {len(new_tenants)} new workspaces...")
            for tid in new_tenants:
                try:
                    ws = await create_workspace(session, tid)
                    workspaces[tid] = ws
                    print(f"  Created workspace for {tid}: {ws['workspace_id'][:8]}...")
                except Exception as e:
                    print(f"  FAILED for {tid}: {e}")
            save_workspaces(workspaces)

        # Ingest each tenant
        results = []
        for i, (tid, docs) in enumerate(tenants.items()):
            if tid not in workspaces:
                print(f"  Skipping {tid} (no workspace)")
                continue

            print(f"[{i+1}/{len(tenants)}] Ingesting {tid}: {len(docs):,} sessions...")
            result = await ingest_tenant(session, tid, docs, workspaces[tid], semaphore)
            results.append(result)
            print(f"  Done: {result['success']}/{result['total']} stored ({result['failed']} failed)")

    elapsed = time.time() - start
    total_stored = sum(r["success"] for r in results)
    total_failed = sum(r["failed"] for r in results)
    print(f"\n=== INGESTION COMPLETE ===")
    print(f"Tenants: {len(results)}")
    print(f"Stored: {total_stored:,}")
    print(f"Failed: {total_failed:,}")
    print(f"Time: {elapsed:.1f}s")

    # Save final workspace mapping
    save_workspaces(workspaces)
    print(f"Workspaces saved to {WORKSPACES_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
