# rag-eval-gate

A CI/CD-integrated evaluation pipeline that acts as a **quality gate** for RAG (Retrieval-Augmented Generation) systems. Block bad PRs before they ship hallucinating AI to production.

[![PyPI version](https://badge.fury.io/py/rag-eval-gate.svg)](https://pypi.org/project/rag-eval-gate/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/ManikBodamwad/RAG-EVAL/actions/workflows/rag_eval.yml/badge.svg)](https://github.com/ManikBodamwad/RAG-EVAL/actions/workflows/rag_eval.yml)

---

## The Problem

You ship a RAG chatbot. A teammate changes the prompt template. The retriever now returns irrelevant context. The LLM starts hallucinating. Nobody catches it until users complain.

**rag-eval-gate** prevents this by running automated evaluations on every Pull Request — just like unit tests, but for AI output quality.

## How It Works

When a pull request is opened, the GitHub Action:

1. Loads a **golden evaluation dataset** (from Hugging Face or a local `.jsonl` file)
2. Runs each question through your **RAG pipeline**
3. Evaluates outputs using **Ragas metrics** with a Groq LLM judge
4. Computes a custom **Token Efficiency** metric (quality per output token)
5. Checks scores against **configurable thresholds** in `eval_config.yaml`
6. Pushes metrics to **Grafana Cloud** for trend tracking
7. Posts a **formatted score table** as a PR comment
8. **Fails the CI job** if any metric drops below threshold — blocking the merge

## Evaluation Metrics

| Metric | What It Measures | Default Threshold |
|--------|------------------|-------------------|
| **Faithfulness** | Are answers grounded in retrieved context? | ≥ 0.75 |
| **Context Relevance** | Is the retrieved context relevant to the question? | ≥ 0.70 |
| **Answer Correctness** | How accurate is the answer vs ground truth? | ≥ 0.65 |
| **Token Efficiency** | Quality per output token (`correctness / log(1 + tokens)`) | ≥ 0.50 |

The default LLM Judge is `groq/llama-3.3-70b-versatile` via LiteLLM — fast, free, and swappable.

## Quick Start

```bash
# Install from PyPI
pip install rag-eval-gate

# Set your Groq API key (free at console.groq.com)
export GROQ_API_KEY="your_api_key"

# Run evaluation
rag-eval run

# View formatted report
rag-eval report
```

### Try the Hallucination Demo 🚨

See `rag-eval-gate` catch a hallucinating AI in real-time. This demo intentionally forces the mock RAG pipeline to hallucinate an answer about "RLHF", proving that the quality gate works:

```bash
python examples/demo.py
```

## GitHub Actions Setup

Add this workflow to `.github/workflows/rag_eval.yml`:

```yaml
name: RAG Evaluation
on: [pull_request]

jobs:
  eval:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install rag-eval-gate
      - run: rag-eval run --config eval_config.yaml
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
```

Set `GROQ_API_KEY` in your GitHub repository secrets (Settings → Secrets → Actions).

## Configuration

Customize thresholds and model settings in `eval_config.yaml`:

```yaml
thresholds:
  faithfulness_min: 0.75
  context_relevance_min: 0.70
  answer_correctness_min: 0.65
  token_efficiency_min: 0.50

model:
  judge: "groq/llama-3.3-70b-versatile"
  rag_generator: "groq/llama-3.3-70b-versatile"
  embeddings: "sentence-transformers/all-MiniLM-L6-v2"

dataset:
  hf_repo: "manikbodamwad/rag-eval-golden"
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  GitHub Actions CI                   │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Golden Dataset (HF Hub / local JSONL)              │
│         │                                           │
│         ▼                                           │
│  RAG Pipeline (FAISS + Groq LLM via LiteLLM)       │
│         │                                           │
│         ▼                                           │
│  Ragas Evaluation (Faithfulness, Relevance, etc.)   │
│         │                                           │
│         ▼                                           │
│  Regression Gate (pass/fail vs thresholds)          │
│         │                                           │
│    ┌────┴────┐                                      │
│    ▼         ▼                                      │
│  ✅ Pass   ❌ Fail → Block PR merge                 │
│    │         │                                      │
│    ▼         ▼                                      │
│  PR Comment + Grafana Metrics Push                  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## Bring Your Own Pipeline

The library ships with a demo RAG pipeline, but you can plug in your own. Subclass `BaseRAGPipeline`, implement two methods, and point your config at it:

```python
# my_pipeline.py
from rag_eval import BaseRAGPipeline, RAGResult

class MyPipeline(BaseRAGPipeline):
    def init(self):
        """Called once before evaluation starts. Load your models here."""
        self.db = load_my_vectorstore()
        self.llm = load_my_llm()

    def query(self, question: str) -> RAGResult:
        """Called for each question in the test dataset."""
        docs = self.db.search(question, k=3)
        answer = self.llm.generate(question, docs)
        return RAGResult(
            question=question,
            answer=answer,
            contexts=[d.text for d in docs],
            input_tokens=...,   # optional, for token efficiency metric
            output_tokens=...,  # optional, for token efficiency metric
        )
```

Then set `pipeline.class` in your `eval_config.yaml`:

```yaml
pipeline:
  class: "my_pipeline.MyPipeline"

thresholds:
  faithfulness_min: 0.75
  context_relevance_min: 0.70
  answer_correctness_min: 0.65
  token_efficiency_min: 0.50
```

Run it:

```bash
export GROQ_API_KEY="..."
rag-eval run --config eval_config.yaml
```

The evaluator will import your class, call `init()` once, then call `query()` for each test question.

## Tech Stack

- **Evaluation**: [Ragas](https://github.com/explodinggradients/ragas) for LLM-as-judge metrics
- **LLM Provider**: [Groq](https://console.groq.com/) via [LiteLLM](https://github.com/BerriAI/litellm) (hot-swappable to OpenAI, Anthropic, etc.)
- **Embeddings**: [sentence-transformers](https://www.sbert.net/) (local, no API calls)
- **Vector Store**: [FAISS](https://github.com/facebookresearch/faiss) (CPU, local)
- **Dataset**: [Hugging Face Datasets](https://huggingface.co/datasets/manikbodamwad/rag-eval-golden)
- **Observability**: [Grafana Cloud](https://grafana.com/) via Influx Line Protocol
- **CLI**: [Click](https://click.palletsprojects.com/) + [Rich](https://github.com/Textualize/rich)

## Local Development

```bash
git clone https://github.com/ManikBodamwad/RAG-EVAL.git
cd RAG-EVAL
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env

# Run local evaluation
rag-eval run

# View formatted report
rag-eval report

# Run unit tests
python -m pytest tests/
```

## Test Dataset

The default test set is hosted at [`manikbodamwad/rag-eval-golden`](https://huggingface.co/datasets/manikbodamwad/rag-eval-golden) on Hugging Face. To use your own dataset, create a JSONL file with the following schema:

```jsonl
{"question": "What is X?", "ground_truth": "X is ...", "reference_context": "The passage that answers this..."}
```

Then specify the local path or your own HF repo in `eval_config.yaml`.

## License

MIT License.
