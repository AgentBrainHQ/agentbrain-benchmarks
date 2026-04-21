# Agent Brain Benchmarks

Public, reproducible benchmarks for [Agent Brain](https://agentbrain.ch) and
comparisons to peer memory systems. Companion code to the paper
[**Agent Brain: A Biologically Inspired Memory System for Autonomous AI Agents,
with Head-to-Head Evaluation on LongMemEval**](https://doi.org/10.5281/zenodo.19673133)
(Sritharan, 2026).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19673133.svg)](https://doi.org/10.5281/zenodo.19673133)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Results: LongMemEval-m-cleaned

500 QA pairs across 510 multi-turn workspaces, GPT-4o judge.

| Configuration | Accuracy | Reproducible here |
|---|:---:|:---:|
| **Agent Brain — Test 0 (no consolidation)** | **71.7%** | Yes |
| Agent Brain — Test 1 (with Dream Cycle) | 69.8% | Yes |
| Baseline pgvector (our control) | 72.2% – 73.9% | Yes |

We report transparently a 1.9 pp regression when the Dream Cycle is
enabled, and a 2.2 pp gap versus our own pgvector-only control. See
[§15.4 of the paper](https://doi.org/10.5281/zenodo.19673133) for discussion.

Aggregated per-run numbers: [`results/SUMMARY.md`](results/SUMMARY.md).

> **Note on cross-system comparisons (April 2026).** Published numbers from
> Zep, Mem0, LangMem, and OpenAI Memory are mostly reported on the
> `LongMemEval-S` (Small) variant, while this work uses `LongMemEval-M`
> (Medium, `m-cleaned`). We initially compared numerically across variants in
> earlier preprint versions; a corrected discussion is in preparation (paper
> v3). For an apples-to-apples comparison we recommend re-evaluating each
> system on `weaviate/longmemeval-m-cleaned` using the same judging rubric.
> We also note that Mem0 released a substantially improved v2 around
> 17 April 2026 (~92% reported, variant unclear) which post-dates the
> writing of v1 but should inform any future comparison.

---

## Quickstart

### Requirements

- Python 3.10+
- A Brain instance (self-hosted or the hosted evaluation endpoint)
- OpenAI API key (for GPT-4o answer generation and judge)
- ~8 h wall clock for a full 500-tenant run
- ~USD 22 in API cost (USD 18 judge + USD 4 embeddings, or USD 0 with self-hosted MiniLM)

### Setup

```bash
git clone https://github.com/AgentBrainHQ/agentbrain-benchmarks
cd agentbrain-benchmarks

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env with your Brain DB + OpenAI keys
set -a; source .env; set +a
```

### Download the dataset

```bash
# Faster: parquet-based loader (single file, ~1 min)
python download_parquet.py

# Or: HuggingFace rows API (resumable, slower)
python download_data.py
```

Both variants write `data/docs.jsonl` (237,655 sessions) and `data/queries.jsonl`
(500 QA pairs).

### Run the full benchmark

```bash
# One-shot orchestrator — ingest + query both systems + judge
python run_full.py

# Smoke test with 10 tenants
python run_full.py --limit 10

# Or step-by-step (async variants)
python ingest.py          # Ingest via Brain API
python query.py           # Brain recall + GPT-4o answer
python baseline.py        # pgvector-only control
python evaluate.py        # GPT-4o judge

# Or via shell wrapper
./run_benchmark.sh        # full run
./run_benchmark.sh --small   # 100-doc smoke run
```

Results land in `results/`.

---

## Repository Layout

```
agentbrain-benchmarks/
├── config.py              # env-based config (no secrets committed)
├── .env.example           # template for your environment
│
├── download_data.py       # HuggingFace Rows API loader (resumable)
├── download_parquet.py    # HuggingFace Parquet loader (faster)
│
├── ingest.py              # async ingest via Brain API /memory/store
├── query.py               # async Brain recall + GPT-4o answer
├── baseline.py            # async pgvector-only control (RPC match_memories)
├── evaluate.py            # async GPT-4o judge (rubric-graded)
├── run_full.py            # one-shot sync orchestrator (ingest+query+eval)
├── run_benchmark.sh       # shell wrapper for the modular flow
│
├── prompts/               # exact prompts used in the paper
│   ├── answer_prompt.md
│   └── judge_prompt.md
│
├── docs/
│   ├── METHODOLOGY.md     # ingestion, retrieval, judging — details
│   ├── REPRODUCIBILITY.md # step-by-step reproduction guide
│   └── LIMITATIONS.md     # what LongMemEval does and does not measure
│
├── results/
│   ├── SUMMARY.md                      # aggregated verdict table
│   └── eval_report_aggregated.json     # per-run counts (no raw LongMemEval content)
│
├── CITATION.cff           # cite this work
├── LICENSE                # MIT
├── requirements.txt       # pinned versions
└── README.md              # you are here
```

---

## Methodology (at a glance)

- **Dataset:** `weaviate/longmemeval-m-cleaned` (500 queries × 510 workspaces, ~115k tokens each).
- **Ingestion:** one user × one session per workspace. Turn-level messages
  written to `memories` table via the Brain `/memory/store` endpoint.
- **Embedding:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  (384 dimensions, normalized).
- **Retrieval:** hybrid vector (0.7) + PostgreSQL tsvector (0.3) via Reciprocal
  Rank Fusion (k = 60), top-5 returned. For the baseline: pure pgvector cosine
  top-10 via Supabase RPC.
- **Answer generation:** GPT-4o with top-5 memories as context, single forward
  pass, no chain-of-thought, temperature 0.
- **Judge:** GPT-4o with the rubric in [`prompts/judge_prompt.md`]. Verdicts:
  `CORRECT`, `PARTIAL`, `WRONG`, `ABSTAIN_CORRECT`. Accuracy =
  (correct + abstain) / total.

Full details: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

---

## Reproducing our numbers

A clean Brain deployment + the steps in the Quickstart above reproduces the
paper's Test 1 (69.8%) run end-to-end in ~8 hours for ~USD 22. Test 0 requires
disabling the Dream Cycle between ingest and query (see
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)).

If you can't reproduce a number within ±1 pp, please open an issue with your
`results/eval_report.json`. We take reproducibility seriously.

---

## What this benchmark does **not** measure

LongMemEval measures quiz-style factual recall from long conversational context.
It does **not** measure cross-session continuity, relational reasoning across
workspaces, temporal reasoning, or creative-connection synthesis — workloads
Agent Brain is primarily designed for. See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md)
and §15.5 of the paper.

---

## Citation

```bibtex
@techreport{sritharan2026agentbrain,
  title   = {Agent Brain: A Biologically Inspired Memory System for Autonomous
             AI Agents, with Head-to-Head Evaluation on LongMemEval},
  author  = {Sritharan, Theshoth},
  year    = {2026},
  month   = {4},
  address = {Sachseln OW, Switzerland},
  institution = {Valtis},
  doi     = {10.5281/zenodo.19673133},
  url     = {https://doi.org/10.5281/zenodo.19673133},
  note    = {Version 2}
}
```

Or use the [`CITATION.cff`](CITATION.cff) file directly (GitHub renders a
"Cite this repository" button).

---

## License

MIT — see [`LICENSE`](LICENSE). The LongMemEval dataset itself is distributed
under its own license (see
[weaviate/longmemeval-m-cleaned](https://huggingface.co/datasets/weaviate/longmemeval-m-cleaned)).
This repository contains only evaluation code and aggregated scores; no raw
LongMemEval content is committed.

---

## Contact

Paper and benchmark: **Theshoth Sritharan** · [t.sritharan@valtis.ch](mailto:t.sritharan@valtis.ch) · [ORCID 0009-0006-4400-3352](https://orcid.org/0009-0006-4400-3352)

Issues, bug reports, reproduction problems: please open a GitHub issue.
