# Limitations of LongMemEval as a Memory Benchmark

This document accompanies §15.5 of the paper and argues — from inside a
memory-systems team publishing a result on this benchmark —
why LongMemEval is **necessary but not sufficient** for evaluating agent
memory systems.

## What LongMemEval measures well

- **Factual recall** from a long conversational context.
- **Single-hop retrieval** (find the one turn containing the fact).
- **Robustness to conversational noise** (sessions include off-topic chat).

These are real capabilities and we take the signal seriously.

## What LongMemEval does **not** measure

### 1. Cross-session continuity
Remembering relevant context from a conversation held a week ago, when the
user starts a fresh session today. This is the central production workload
for long-lived agents — and the strongest argument for FSRS-based weighting
and usage-based reinforcement — yet LongMemEval's QA-pair format does not
probe it directly.

### 2. Relational reasoning across entities
Connecting facts that live in different memories through a shared entity
(e.g., "what happened with the tenant who reported the mold issue and also
complained about heating last winter?"). Our Knowledge Graph + graph-traversal
retrieval is designed for this; LongMemEval's single-hop questions do not
reward it.

### 3. Temporal reasoning
Deciding which version of a time-varying fact is currently true ("my
landlord's phone number was X last month; is it still X?"). Temporal
knowledge graphs — including our hybrid implementation with `owner_profiles`
and `owner_agent_memory` tables carrying explicit timestamps — are built for
this class of queries. LongMemEval does not stress it.

### 4. Creative-connection synthesis
The Dream Cycle's creative-connection phase (random walks in the knowledge
graph) produces implicit associations that surface in later recall. These
help multi-hop reasoning but **hurt** on single-hop quiz benchmarks because
they add noise to the top-5.

### 5. Real-world message ambiguity
LongMemEval sessions are curated QA-friendly dialogues. Production messages
contain typos, code-switching, vague references, attachments, and multi-intent
turns. Our production benchmark (§14.6 of the paper, P@5 = 0.787) is a closer
proxy for that workload but is not cross-comparable across systems.

## Consequence: our negative result on LongMemEval

Our own un-consolidated Brain (Test 0, 71.7%) outperforms our full pipeline
with Dream Cycle enabled (Test 1, 69.8%) by 1.9 pp. Our naked pgvector
baseline (72.2–73.9%) outperforms both. We interpret this as evidence that:

- **Consolidation is too aggressive for quiz-style lookups.** It compresses
  multiple related memories into summaries that lose the verbatim phone
  numbers, dates, and specific names the judge rewards.
- **Entity abstraction costs recall on single-hop lookups.** Entity-based
  retrieval adds noise when the answer lives in exactly one turn of one
  session.
- **Hybrid retrieval helps continuity but not quizzes.** Our 70/30
  vector/tsvector split trades some exact-match lexical recall for
  paraphrase robustness.

These are design choices favoring production continuity over quiz
performance. We report them transparently — see §15.4 of the paper.

## What would change this picture

A public benchmark that includes:

- **Multi-session continuity pairs.** "Given this week's conversation,
  what do you remember from three months ago about X?"
- **Cross-entity reasoning.** "Who else is connected to Y through Z, and
  what did they say about it?"
- **Temporal disambiguation.** Questions that only resolve correctly if
  the system tracks the time-validity of facts.
- **Creative-synthesis evaluation.** "What pattern connects these five
  sessions?" scored for insight, not factual recall.

We are interested in co-developing such a benchmark — see §18.5 of the
paper. If you work on memory benchmarks, please reach out.

## Why we still published a LongMemEval result

- It is the best-known public benchmark for memory-agent systems, so
  measuring on it anchors future comparisons.
- The number `71.7%` on `LongMemEval-M-cleaned` is what our system actually
  produces on a clearly specified dataset variant — we prefer to publish it,
  including the Dream Cycle regression, rather than stay silent.
- **We do not claim state-of-the-art.** Published peer numbers are on the
  `LongMemEval-S` variant, so cross-system deltas in earlier versions of
  this paper were not apples-to-apples. Mem0's v2 release (April 2026)
  reports substantially higher accuracy than our result. A corrected
  discussion is in paper v3.

## Bottom line

> On `weaviate/longmemeval-m-cleaned` our system achieves 71.7% accuracy.
> It is optimized for long-running production workloads — cross-session
> continuity, relational reasoning, temporal reasoning — rather than
> quiz-style factual recall. We publish the number transparently so that
> future systems can be compared on the same variant under the same
> judging rubric.

---

**See also:** [`METHODOLOGY.md`](METHODOLOGY.md),
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md), and §15 of the
[paper](https://doi.org/10.5281/zenodo.19673133).
