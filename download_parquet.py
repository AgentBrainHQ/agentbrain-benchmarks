"""
Download LongMemEval-M via Parquet files (much faster than Row API).
HuggingFace serves parquet files directly — no rate limits.
"""

import json
import os
import sys
import time

import pyarrow.parquet as pq

from config import DATA_DIR

DOCS_FILE = os.path.join(DATA_DIR, "docs.jsonl")
QUERIES_FILE = os.path.join(DATA_DIR, "queries.jsonl")

# HuggingFace parquet URLs (discovered via API)
PARQUET_BASE = "https://huggingface.co/api/datasets/weaviate/longmemeval-m-cleaned/parquet"


def download_parquet_to_jsonl(config: str, output_file: str):
    """Download parquet file and convert to JSONL."""
    import urllib.request

    parquet_url = f"{PARQUET_BASE}/{config}/train/0.parquet"
    parquet_file = os.path.join(DATA_DIR, f"{config}.parquet")

    # Download parquet
    if not os.path.exists(parquet_file):
        print(f"[{config}] Downloading parquet...")
        urllib.request.urlretrieve(parquet_url, parquet_file)
        size_mb = os.path.getsize(parquet_file) / 1024 / 1024
        print(f"[{config}] Downloaded: {size_mb:.1f} MB")
    else:
        size_mb = os.path.getsize(parquet_file) / 1024 / 1024
        print(f"[{config}] Parquet exists: {size_mb:.1f} MB")

    # Convert to JSONL
    print(f"[{config}] Converting to JSONL...")
    table = pq.read_table(parquet_file)
    df = table.to_pandas()

    import numpy as np

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    count = 0
    with open(output_file, "w") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), default=convert) + "\n")
            count += 1
            if count % 10000 == 0:
                print(f"  [{config}] {count:,} rows...", end="\r")

    print(f"\n[{config}] Done: {count:,} rows saved to {output_file}")

    # Cleanup parquet
    os.remove(parquet_file)
    return count


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    start = time.time()

    # Check if docs might have multiple parquet shards
    # For large datasets HF may split into multiple files
    for config, output, expected_min in [("queries", QUERIES_FILE, 500), ("docs", DOCS_FILE, 237000)]:
        if os.path.exists(output) and "--force" not in sys.argv:
            lines = sum(1 for _ in open(output))
            if lines >= expected_min:
                print(f"[{config}] Complete ({lines:,} rows)")
                continue

        download_parquet_to_jsonl(config, output)

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
