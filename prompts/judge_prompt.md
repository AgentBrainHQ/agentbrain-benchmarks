# Judge Prompt

**Model:** GPT-4o, temperature 0, max_tokens 20.

We use a rubric-graded judge that emits one of four verdicts. This is the same
rubric used by the original LongMemEval paper (Wu et al., 2024), adapted to
our four-class setup.

## Prompt (used by `evaluate.py`, async modular flow)

```
You are evaluating whether a system correctly answered a question about past
conversations.

## Ground Truth Answer
<ground truth>

## System's Answer
<hypothesis>

## Question
<question>

Rate the system's answer on correctness:
- CORRECT: The answer contains the key information from the ground truth
- PARTIAL: The answer is partially correct or contains some relevant info but
  misses key details
- WRONG: The answer is incorrect, irrelevant, or says it doesn't know when the
  info was available
- ABSTAIN_CORRECT: The system correctly abstained (said it doesn't know) when
  the ground truth is also "not available" or similar

Respond with ONLY one of: CORRECT, PARTIAL, WRONG, ABSTAIN_CORRECT
```

## Prompt (used by `run_full.py`, sync one-shot flow)

Same semantics, rendered as a system + user message pair:

```
System: You are a judge evaluating whether a hypothesis answer is correct
        given a ground-truth answer. Respond with exactly one word: CORRECT,
        PARTIAL, WRONG, or ABSTAIN_CORRECT.
        - CORRECT: hypothesis conveys the same information as ground truth
        - PARTIAL: hypothesis is partially correct but missing key details
        - WRONG: hypothesis is incorrect or irrelevant
        - ABSTAIN_CORRECT: hypothesis correctly states it doesn't know
          (and the question is unanswerable)

User: Question: <question>
      Ground Truth: <answer>
      Hypothesis: <hypothesis>

      Verdict:
```

Both variants produce the same four-class verdict. Accuracy is computed as
`(CORRECT + ABSTAIN_CORRECT) / total`. The `accuracy_with_partial` metric adds
`0.5 × PARTIAL`.

---

**Source:** `run_full.py::judge_answer()` and `evaluate.py::judge_answer()`.
