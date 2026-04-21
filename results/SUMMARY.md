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

## Cross-system comparison — important caveat

**The published numbers we originally cited for Zep, Mem0, LangMem, and
OpenAI Memory are on the `LongMemEval-S` (Small) variant, while this work
uses `LongMemEval-M` (`m-cleaned`). They are therefore not directly
comparable.** An apples-to-apples comparison requires re-running each system
on `weaviate/longmemeval-m-cleaned` under identical judging conditions — we
have not done this for competing systems.

Additionally, Mem0 released a v2 update (~17 April 2026) reporting
substantially higher accuracy (~92% on LongMemEval, variant not fully
specified in the public posting at time of writing). Any comparison with
Mem0 should use its current version, not older 2024 numbers.

A corrected cross-system discussion is in preparation (paper v3). For now,
please treat the number `71.7%` as a single-system self-report on a clearly
specified dataset variant, and re-benchmark competing systems on the same
variant before drawing conclusions.

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
