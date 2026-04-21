"""
Download LongMemEval-M dataset from HuggingFace API.
Saves docs.jsonl and queries.jsonl to data/ directory.
Supports resume (appends to existing file from last offset).
"""

import asyncio
import json
import os
import sys
import time

import aiohttp

from config import HF_ROWS_URL, HF_PAGE_SIZE, DATA_DIR

DOCS_FILE = os.path.join(DATA_DIR, "docs.jsonl")
QUERIES_FILE = os.path.join(DATA_DIR, "queries.jsonl")


async def fetch_page(session: aiohttp.ClientSession, config: str, offset: int, retries: int = 3) -> list[dict]:
    url = f"{HF_ROWS_URL}&config={config}&split=train&offset={offset}&length={HF_PAGE_SIZE}"
    for attempt in range(retries):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [r["row"] for r in data.get("rows", [])]
                elif resp.status == 429:
                    wait = 2 ** (attempt + 1)  # 2, 4, 8 seconds
                    await asyncio.sleep(wait)
                    continue
                else:
                    print(f"  ERROR {config} offset={offset}: {resp.status}")
                    return []
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(2)
            else:
                print(f"  TIMEOUT {config} offset={offset}: {e}")
                return []
    return []


async def download_config(config: str, output_file: str, resume_from: int = 0):
    async with aiohttp.ClientSession() as session:
        # Get total count
        url = f"{HF_ROWS_URL}&config={config}&split=train&offset=0&length=1"
        async with session.get(url) as resp:
            data = await resp.json()
            total = data["num_rows_total"]

        if resume_from >= total:
            print(f"[{config}] Already complete ({resume_from:,}/{total:,})")
            return

        print(f"[{config}] {total:,} rows, resuming from {resume_from:,}")

        count = resume_from
        mode = "a" if resume_from > 0 else "w"
        with open(output_file, mode) as f:
            # Sequential batches of 3 concurrent requests (conservative for rate limits)
            batch_size = 3
            offset = resume_from
            while offset < total:
                tasks = []
                for i in range(batch_size):
                    o = offset + i * HF_PAGE_SIZE
                    if o >= total:
                        break
                    tasks.append(fetch_page(session, config, o))

                results = await asyncio.gather(*tasks)
                for rows in results:
                    for row in rows:
                        f.write(json.dumps(row) + "\n")
                        count += 1

                offset += batch_size * HF_PAGE_SIZE
                pct = min(100, count * 100 // total)
                print(f"  [{config}] {count:,}/{total:,} ({pct}%)", end="\r")

                # Rate limit pause between batches
                await asyncio.sleep(0.5)

        print(f"\n[{config}] Done: {count:,} rows saved")


async def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    start = time.time()

    # Download queries (small, 500 rows)
    if os.path.exists(QUERIES_FILE) and "--force" not in sys.argv:
        lines = sum(1 for _ in open(QUERIES_FILE))
        if lines >= 500:
            print(f"[queries] Complete ({lines:,} rows)")
        else:
            await download_config("queries", QUERIES_FILE, resume_from=lines)
    else:
        await download_config("queries", QUERIES_FILE)

    # Download docs (large, 237K rows) — supports resume
    if os.path.exists(DOCS_FILE) and "--force" not in sys.argv:
        lines = sum(1 for _ in open(DOCS_FILE))
        if lines >= 237655:
            print(f"[docs] Complete ({lines:,} rows)")
        else:
            await download_config("docs", DOCS_FILE, resume_from=lines)
    else:
        await download_config("docs", DOCS_FILE)

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
