#!/bin/bash
# LongMemEval Benchmark — Full Pipeline
# Usage: ./run_benchmark.sh [--small]  (--small = 1 tenant only for testing)

set -e
cd "$(dirname "$0")"

echo "=== LongMemEval Benchmark ==="
echo ""

# Step 1: Download dataset
echo "[1/5] Downloading dataset..."
python download_data.py

# Step 2: Analyze tenants
echo ""
echo "[2/5] Analyzing tenants..."
python ingest.py --dry-run

# Step 3: Ingest into Brain
echo ""
if [ "$1" = "--small" ]; then
    echo "[3/5] Ingesting (small test — limit 100)..."
    python ingest.py --limit 100
else
    echo "[3/5] Ingesting all sessions into Brain..."
    python ingest.py
fi

# Step 4: Wait for Dream Cycle (Brain needs time to process)
echo ""
echo "[4/5] Waiting 60s for Dream Cycle to process..."
echo "  (In production, wait longer for full consolidation)"
sleep 60

# Step 5: Run queries
echo ""
echo "[5a/5] Running Brain queries..."
if [ "$1" = "--small" ]; then
    python query.py --limit 10
else
    python query.py
fi

echo ""
echo "[5b/5] Running Baseline queries..."
if [ "$1" = "--small" ]; then
    python baseline.py --limit 10
else
    python baseline.py
fi

# Step 6: Evaluate
echo ""
echo "[6/6] Evaluating results..."
python evaluate.py

echo ""
echo "=== BENCHMARK COMPLETE ==="
echo "Results in: results/"
