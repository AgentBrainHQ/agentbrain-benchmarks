# Reproducibility Guide

Goal: take someone from `git clone` to reproducing our 71.7% / 69.8% /
72.2–73.9% numbers in one afternoon.

## Prerequisites

| Requirement | Version tested |
|---|---|
| Python | 3.10, 3.11, 3.12 |
| pip | 23+ |
| Postgres | 15+ with `pgvector` + `tsvector` + `pg_cron` |
| Supabase REST API | current (tested April 2026) |
| OpenAI API access | GPT-4o (answer + judge) |

You need access to a **Brain instance**. Two options:

1. **Self-host Brain** from the main Agent Brain codebase (not in this repo).
   Spin up a Supabase project, apply the Brain schema, deploy the Brain API
   container, and point `BRAIN_API_URL` / `BRAIN_DB_URL` at it.
2. **Evaluation endpoint**. Contact the authors (see README) for read/write
   access to the hosted eval deployment at `bench.agentbrain.ch`. Reserved
   for academic reproduction; do not use it for production load.

## Step-by-step

### 1. Clone + install

```bash
git clone https://github.com/AgentBrainHQ/agentbrain-benchmarks
cd agentbrain-benchmarks
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On first run, `sentence-transformers` will download
`paraphrase-multilingual-MiniLM-L12-v2` (~ 450 MB) to `~/.cache/huggingface`.

### 2. Configure

```bash
cp .env.example .env
# edit .env — fill in BRAIN_DB_URL, BRAIN_DB_SERVICE_KEY, OPENAI_API_KEY
set -a; source .env; set +a
```

### 3. Download dataset

Preferred (fast, single parquet file):

```bash
python download_parquet.py
```

Fallback (resumable via HF Rows API):

```bash
python download_data.py
```

Verify:

```bash
wc -l data/docs.jsonl data/queries.jsonl
# Expected: 237655 data/docs.jsonl
#           500 data/queries.jsonl
```

### 4. Smoke test (10 tenants, ~10 min)

```bash
python run_full.py --limit 10
```

Check `results/eval_report.json` — you should see ~10 queries evaluated for
both `brain` and `baseline`. If this fails, fix the error before running the
full benchmark — a full run costs USD 18+.

### 5. Full run (500 queries, ~3.5 h, ~USD 18–22)

**Brain Test 1 (with Dream Cycle):**

```bash
python run_full.py
```

Results:

- `results/brain_output.jsonl` — 500 Brain hypotheses
- `results/baseline_output.jsonl` — 500 pgvector-baseline hypotheses
- `results/eval_report.json` — aggregated verdicts + per-query verdicts

Expected headline numbers (within ±1 pp from GPT-4o variance):

| System | Accuracy | Accuracy with partial |
|---|:---:|:---:|
| Brain (Test 1, Dream Cycle) | ~69.8% | ~72.7% |
| Baseline (pgvector only) | ~72.2–73.9% | ~74.2% |

### 6. Brain Test 0 (no Dream Cycle)

To reproduce the 71.7% Test 0 number:

1. Disable the Dream Cycle scheduler on your Brain deployment
   (`DREAM_CYCLE_ENABLED=false` or analogous).
2. Run `python run_full.py --skip-ingest` against a freshly-ingested corpus
   that has never seen a consolidation pass.

Alternatively, ingest fresh workspaces, query immediately (before 02:00 UTC),
then stop. Dream Cycle will not yet have modified memories.

## Known failure modes

| Symptom | Cause | Fix |
|---|---|---|
| 500 error "57014 statement timeout" on insert | Brain DB statement timeout too low | Raise `statement_timeout` to 60s, or insert in smaller batches |
| `content-range` header parse error | Supabase REST returned `*/0` for empty table | Ignore, treated as 0 existing memories |
| Judge returns "ERROR" | OpenAI rate limit | Lower `MAX_CONCURRENT_LLM` to 3 |
| Cosine similarity all zero | Embedding column stored as string without cast | Ensure pgvector extension is enabled and embedding cast `::vector(384)` works |

## Verifying integrity

Our reference `results/eval_report_aggregated.json` contains aggregated
counts. Your run should produce verdict counts within these ranges:

| Verdict (Brain Test 1) | Our count | Acceptable range |
|---|---:|---|
| CORRECT | 349 | 330–370 |
| PARTIAL | 29 | 20–40 |
| WRONG | 122 | 100–140 |
| ABSTAIN_CORRECT | 0 | 0–10 |
| ERROR | 0 | 0 |

If you fall outside the acceptable range, please open a GitHub issue with
your full `eval_report.json` and the commit hash you ran against.

## Determinism notes

- MiniLM embeddings are deterministic on CPU (`model.encode(..., show_progress_bar=False)`).
- GPT-4o judge is **not** bitwise-deterministic even at `temperature=0` —
  expect ±1 pp run-to-run variance.
- Pipeline ordering is deterministic by `sorted(tenant_ids)`.
- Random workspace UUIDs and API keys differ per run but do not affect metrics.

## Contact

Reproduction issues: open a GitHub issue with `[repro]` in the title.
