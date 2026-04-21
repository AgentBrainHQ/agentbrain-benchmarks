# Methodology

This document details the ingestion, retrieval, and judging procedures used
for the LongMemEval benchmark reported in the paper
([DOI: 10.5281/zenodo.19673132](https://doi.org/10.5281/zenodo.19673132), Concept DOI resolving to latest version).

## 1. Dataset

- **Source:** [`weaviate/longmemeval-m-cleaned`](https://huggingface.co/datasets/weaviate/longmemeval-m-cleaned)
  on Hugging Face.
- **Queries:** 500 question–answer pairs with human-annotated ground truth.
- **Docs:** 237,655 conversation sessions across 510 tenants
  (~465 sessions per tenant on average).
- **Context length:** approximately 115k tokens per workspace.
- **License:** Distributed under the dataset's license on Hugging Face. This
  repository does **not** redistribute raw LongMemEval content; only aggregated
  verdict counts are committed.

## 2. Ingestion

Each tenant in the dataset is mapped to one **Brain workspace**. Within that
workspace, the 500+ sessions belonging to the tenant are each stored as one
episodic memory.

Memory content is assembled as:

```
[Session: <session_id> | Date: <session_date>]
<session_text>
```

Two ingestion paths exist in this repository:

- **`ingest.py`** (async, via Brain API). Sends each memory through the full
  Brain pipeline: Perception Gate, Deduplication Guard, NER extraction,
  Knowledge Graph linking. This path produces the **Brain Test 1** corpus
  (entity-extracted, consolidation-eligible).

- **`run_full.py::ingest_tenant()`** (sync, direct Supabase insert). Bypasses
  the Brain API and writes memories straight to the `memories` table with
  locally computed embeddings. This path produces the **Brain Test 0** and
  **Baseline** corpus (no entity extraction, no consolidation side-effects).

### Embedding model

```
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

- 384-dimensional embeddings.
- Output is normalized.
- Runs locally (no API cost for embeddings).

### Memory row schema (benchmark-relevant fields)

| Column | Value |
|---|---|
| `content` | `[Session: <id> \| Date: <d>]\n<session_text>` |
| `type` | `episodic` |
| `embedding` | 384-dim vector (MiniLM-L12-v2) |
| `source_trust` | 0.9 |
| `weight` | 0.6 (initial) |
| `emotion_score` | 0.0 |
| `novelty_score` | 0.5 |
| `urgency_score` | 0.0 |
| `decay_rate` | 0.03 |
| `stability` | 1.0 |
| `difficulty` | 0.3 |
| `reps` | 0 |

### Dream Cycle (Test 1 only)

After ingestion, the Brain Dream Cycle is allowed to run (nightly job, 02:00
UTC). It executes five phases: FSRS decay, consolidation of near-duplicates
(cosine > 0.92), creative-connection synthesis via random walks in the
knowledge graph, pattern recognition, and predictive alerts. See §12 of the
paper.

For **Test 0** runs, Dream Cycle scheduling is disabled between ingestion and
query so that memories remain in their ingested state.

## 3. Retrieval

### Agent Brain (Test 0 and Test 1)

Each query goes through the Brain `/memory/recall` endpoint:

```
Query → LLM Query Expansion (Claude Haiku, 3–5 technical-term expansions)
      → Hybrid Search:
         - Vector: cosine similarity on MiniLM embeddings (weight 0.7)
         - Keyword: PostgreSQL tsvector with German stemmer (weight 0.3)
         - Fusion: Reciprocal Rank Fusion (k = 60)
      → Weight filter (> 0.05)
      → NER on query → Graph traversal (depth 1)
      → top-k memories returned
```

`k = RECALL_LIMIT = 10` by default.

### Baseline (pgvector-only control)

The `baseline.py` (async) and `run_full.py::vector_recall_local()` (sync) paths
compute cosine similarity on the raw MiniLM embeddings via Supabase RPC
`match_memories` (or local fallback with numpy) and return top-10. **No**
query expansion, **no** hybrid search, **no** entity graph, **no** Dream
Cycle.

## 4. Answer generation

Retrieval top-5 is passed as context to GPT-4o with the prompt in
[`../prompts/answer_prompt.md`]. Single forward pass, no chain-of-thought,
temperature 0, max 500 tokens.

## 5. Judging

Each hypothesis is evaluated against the LongMemEval ground truth by GPT-4o
using the rubric in [`../prompts/judge_prompt.md`]. The judge emits exactly
one of four verdicts:

- `CORRECT` — hypothesis contains the key ground-truth information.
- `PARTIAL` — partially correct, missing key details.
- `WRONG` — incorrect, irrelevant, or incorrectly abstained.
- `ABSTAIN_CORRECT` — correctly abstained (ground truth is unanswerable).

Temperature 0, max 20 tokens. Verdicts are normalized via substring matching
to handle minor formatting drift.

## 6. Metrics

For a set of verdicts `V`:

```
accuracy = (|CORRECT| + |ABSTAIN_CORRECT|) / |V|
accuracy_with_partial = (|CORRECT| + |ABSTAIN_CORRECT| + 0.5 × |PARTIAL|) / |V|
```

The paper reports **`accuracy`** (strict) as the headline number. The
`accuracy_with_partial` metric is provided as supplementary information and
tends to be 2–3 pp higher.

## 7. Configurations

| Configuration | Ingest path | Dream Cycle | Retrieval | Reported in paper |
|---|---|---|---|---|
| Brain Test 0 | direct Supabase | disabled | Brain Recall | §15.3, 71.7% |
| Brain Test 1 | Brain API | enabled (5 phases) | Brain Recall | §15.3, 69.8% |
| Baseline run 1 | direct Supabase | disabled | pgvector RPC | §15.3, 73.9% |
| Baseline run 2 | direct Supabase | disabled | pgvector RPC | §15.3, 72.2% |

The two Baseline runs use the same ingested corpus as Test 0; the run-to-run
variance reflects stochasticity in the GPT-4o judge (temperature 0 but not
bitwise-deterministic across API calls).

## 8. Answer and judge model

Answer model: `gpt-4o`. Judge model: `gpt-4o`. Same model family, no cross-
model contamination. Alternative models can be set via
`ANSWER_MODEL` / `JUDGE_MODEL` environment variables, but published numbers
were produced with the defaults.

## 9. Cost and wall clock

- Ingestion: ~30 min for the direct-Supabase path (500 tenants × ~465 sessions,
  10 sessions per batch, ~150 batches/min throughput).
- Query: ~2 h for 500 queries × 2 systems (Brain + Baseline) with
  MAX_CONCURRENT_LLM = 5.
- Judge: ~30 min for 1000 verdicts at 5 concurrent GPT-4o calls.
- **Total wall clock:** ~3.5 h (sync `run_full.py`); the async modular flow
  is faster but harder to debug end-to-end.
- **API cost:** ~USD 18 for GPT-4o (answer + judge), USD 0 for MiniLM
  embeddings (self-hosted).

Full cost for one reproducible run: **~USD 18–22**.

## 10. What we do **not** claim

- We do **not** claim Agent Brain is universally better than Zep or Mem0 for
  all memory tasks. On quiz-style benchmarks our own un-consolidated control
  can outperform the full Brain pipeline (see [`LIMITATIONS.md`] and §15.4).
- We do **not** claim the numbers are bitwise reproducible. GPT-4o judging
  introduces ±1 pp variance run-to-run.
- We do **not** claim the MiniLM embedding is optimal. Larger embeddings
  (bge-base-en-v1.5, mxbai-embed-large) are under evaluation; §18 of the
  paper.

---

**See also:** [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md),
[`LIMITATIONS.md`](LIMITATIONS.md).
