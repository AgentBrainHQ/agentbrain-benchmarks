"""
LongMemEval Benchmark Configuration
====================================
All secrets are read from environment variables.
Copy .env.example to .env and fill in your own keys before running.

System A: Brain API (store + Dream Cycle + recall)
System B: Direct pgvector cosine similarity (baseline)

Dataset: weaviate/longmemeval-m-cleaned (HuggingFace)
- docs: 237,655 conversation sessions
- queries: 500 evaluation questions with ground-truth answers
"""

import os
import sys
from pathlib import Path


def _require(var_name: str) -> str:
    """Read required env var or exit with helpful error."""
    val = os.environ.get(var_name)
    if not val:
        print(
            f"ERROR: environment variable {var_name} is not set.\n"
            f"Copy .env.example to .env, fill in your keys, then export them:\n"
            f"  set -a; source .env; set +a\n"
            f"Or run with: {var_name}=... python run_full.py",
            file=sys.stderr,
        )
        sys.exit(1)
    return val


# ---------------------------------------------------------------------------
# HuggingFace Dataset API
# ---------------------------------------------------------------------------
HF_DATASET = "weaviate/longmemeval-m-cleaned"
HF_API_BASE = "https://datasets-server.huggingface.co"
HF_ROWS_URL = f"{HF_API_BASE}/rows?dataset=weaviate%2Flongmemeval-m-cleaned"
HF_PAGE_SIZE = 100  # max rows per request


# ---------------------------------------------------------------------------
# Brain API + Database
# ---------------------------------------------------------------------------
# For reproduction: run your own Brain instance (see README) or contact the
# authors for evaluation access. The benchmark database is isolated from
# production to avoid interference.
BRAIN_API_URL = os.environ.get("BRAIN_API_URL", "https://api.agentbrain.ch")
BRAIN_DB_URL = _require("BRAIN_DB_URL")
BRAIN_DB_SERVICE_KEY = _require("BRAIN_DB_SERVICE_KEY")


# ---------------------------------------------------------------------------
# LLM for answer generation
# ---------------------------------------------------------------------------
# Answer model: GPT-4o via OpenAI direct (for comparability with published
# LongMemEval methodology). OpenRouter is an alternative for non-OpenAI models.
OPENAI_KEY = _require("OPENAI_API_KEY")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "gpt-4o")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o")


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------
MAX_CONCURRENT_STORES = int(os.environ.get("MAX_CONCURRENT_STORES", "2"))
MAX_CONCURRENT_RECALLS = int(os.environ.get("MAX_CONCURRENT_RECALLS", "10"))
MAX_CONCURRENT_LLM = int(os.environ.get("MAX_CONCURRENT_LLM", "5"))


# ---------------------------------------------------------------------------
# Paths (relative to this file)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
DATA_DIR = str(_HERE / "data")
RESULTS_DIR = str(_HERE / "results")


# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------
RECALL_LIMIT = int(os.environ.get("RECALL_LIMIT", "10"))
BASELINE_LIMIT = int(os.environ.get("BASELINE_LIMIT", "10"))


# ---------------------------------------------------------------------------
# Brain store settings
# ---------------------------------------------------------------------------
MEMORY_TYPE = os.environ.get("MEMORY_TYPE", "episodic")
SOURCE_TRUST = float(os.environ.get("SOURCE_TRUST", "0.9"))
