# Answer Generation Prompt

**Model:** GPT-4o, temperature 0, max_tokens 500.

## System

```
You are answering questions based on your memory of past conversations.
Use ONLY the provided memories to answer. Be specific and concise.
If the memories don't contain the answer, say "I don't have that information
in my memory."
```

## User

```
## Recalled Memories

[Memory 1 (relevance: 0.87)]:
<content of top-1 memory>

[Memory 2 (relevance: 0.82)]:
<content of top-2 memory>

... (up to top-k, k = RECALL_LIMIT = 10 by default)

## Question

<the LongMemEval question>

## Answer
```

The `relevance` score shown to the model is the top-level score returned by the
retrieval layer (Brain Recall returns `relevance`; the pgvector baseline
returns `similarity`). For transparency the prompt renders whichever field is
present.

---

**Source:** `run_full.py::generate_answer()`, `query.py::generate_answer()`,
`baseline.py::generate_answer()`.
