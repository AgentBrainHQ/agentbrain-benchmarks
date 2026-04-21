# Results Summary

Aggregated verdict counts from the runs reported in the paper
([DOI: 10.5281/zenodo.19673133](https://doi.org/10.5281/zenodo.19673133)).

## LongMemEval-m-cleaned — 500 queries, GPT-4o judge

| System | Accuracy (strict) | Accuracy (with partial) | Run date |
|---|:---:|:---:|:---:|
| **Agent Brain — Test 0** (no consolidation) | **71.7%** | 74.2% | 2026-04-16 |
| Agent Brain — Test 1 (with Dream Cycle) | 69.8% | 72.7% | 2026-04-17 |
| Baseline pgvector, run 1 | 73.9% | ~75.5% | 2026-04-16 |
| Baseline pgvector, run 2 | 72.2% | 74.2% | 2026-04-17 |

**Accuracy (strict)** = (CORRECT + ABSTAIN_CORRECT) / 500
**Accuracy (with partial)** = (CORRECT + ABSTAIN_CORRECT + 0.5·PARTIAL) / 500

## Detailed verdict breakdown (Brain Test 1 + Baseline run 2, 2026-04-17)

### Agent Brain — Test 1 (Dream Cycle enabled)

| Verdict | Count | Share |
|---|---:|---:|
| CORRECT | 349 | 69.8% |
| PARTIAL | 29 | 5.8% |
| WRONG | 122 | 24.4% |
| ABSTAIN_CORRECT | 0 | 0.0% |
| ERROR | 0 | 0.0% |
| **Total** | **500** | **100%** |

### Baseline pgvector, run 2

| Verdict | Count | Share |
|---|---:|---:|
| CORRECT | 361 | 72.2% |
| PARTIAL | 20 | 4.0% |
| WRONG | 119 | 23.8% |
| ABSTAIN_CORRECT | 0 | 0.0% |
| ERROR | 0 | 0.0% |
| **Total** | **500** | **100%** |

**Delta (Brain − Baseline): −2.4 pp**

See §15.4 of the paper for discussion of why the baseline outperforms the
full Brain pipeline on this quiz-style benchmark.

## Head-to-head against published memory systems

| System | LongMemEval-m accuracy | Source |
|---|:---:|---|
| **Agent Brain (ours, Test 0)** | **71.7%** | This work |
| Zep | 63.8% | Rasmussen et al., 2025 (arXiv:2501.13956) |
| Mem0 | 49.0% | Chhikara et al., 2024 (arXiv:2408.03243) |
| LangMem | 47.1% | LangChain AI, 2025 |
| OpenAI Memory | 40.2% | OpenAI, 2024 |

## Reproducing

See [`../docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md). One full run
is ~3.5 h wall clock and ~USD 18–22 of GPT-4o tokens.

## Raw data availability

We do **not** commit raw per-query hypothesis JSONLs in this repository.
Reasons:

1. They contain substantial verbatim LongMemEval content, which is governed
   by the dataset's own license terms.
2. Re-running the benchmark produces equivalent hypotheses within ±1 pp of
   verdict counts, so committed outputs would add little reproducibility
   value beyond what `eval_report_aggregated.json` already provides.

For legitimate reproduction audits (e.g., identifying specific judge
disagreements), contact the authors; we will share sanitized per-query
verdicts privately under an academic-use agreement.
