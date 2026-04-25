# rag-eval

A CI/CD-integrated evaluation pipeline for RAG systems. 

[![PyPI version](https://badge.fury.io/py/rag-eval.svg)](https://badge.fury.io/py/rag-eval)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![RAG Eval CI](https://github.com/manikbodamwad/rag-eval/actions/workflows/rag_eval.yml/badge.svg)](https://github.com/manikbodamwad/rag-eval/actions/workflows/rag_eval.yml)

`rag-eval` acts as a quality gate for your RAG applications. It evaluates Pull Requests and can block merges if the output quality drops below defined thresholds.

## How it works

When a pull request is opened, the Github Action:
1. Installs the `rag-eval` package.
2. Loads a golden evaluation dataset (from Hugging Face or a local file).
3. Runs the dataset through your Mock RAG pipeline.
4. Evaluates the outputs using Ragas metrics.
5. Checks scores against your defined thresholds in `eval_config.yaml`.
6. Pushes metrics to Grafana for trend tracking.
7. Posts a summary comment on the Pull Request.
8. Fails the CI job if any metric drops below the threshold.

## Evaluation Metrics

| Metric | What It Measures | Default Threshold |
|--------|------------------|-------------------|
| **Faithfulness** | Answers are grounded in retrieved context | ≥ 0.75 |
| **Context Relevance** | Retrieved context quality | ≥ 0.70 |
| **Answer Correctness** | Accuracy vs ground truth | ≥ 0.65 |
| **Token Efficiency** | `correctness / log(1 + tokens)` | ≥ 0.50 |

The default LLM Judge is `groq/llama-3.3-70b-versatile` via LiteLLM.

## Quick Start

```bash
# Install
pip install rag-eval

# Set API key
export GROQ_API_KEY="your_api_key"

# Run evaluation
rag-eval run

# View report
rag-eval report
```

### Try the Hallucination Demo 🚨
Want to see `rag-eval` catch a hallucinating AI in real-time? We built a cinematic terminal demo that intentionally forces our mock RAG pipeline to hallucinate an answer about "RLHF", proving that the quality gate works:

```bash
# Make sure GROQ_API_KEY is exported, then run:
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
      - run: pip install rag-eval
      - run: rag-eval run --config eval_config.yaml
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
```

Ensure you set `GROQ_API_KEY` in your GitHub repository secrets.

## Configuration

You can customize the passing thresholds and dataset endpoints in `eval_config.yaml`:

```yaml
thresholds:
  faithfulness_min: 0.75
  context_relevance_min: 0.70
  answer_correctness_min: 0.65
  token_efficiency_min: 0.50

dataset:
  hf_repo: "manikbodamwad/rag-eval-golden" 
```

## Local Development

```bash
git clone https://github.com/manikbodamwad/rag-eval
cd rag-eval
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env

# Run local evaluation
rag-eval run

# View formatted report
rag-eval report
```

## Golden Dataset

The default test set is pushed to `manikbodamwad/rag-eval-golden` on Hugging Face. To use your own dataset, create a JSONL file with the following schema:

```jsonl
{"question": "What is X?", "ground_truth": "X is ...", "reference_context": "The passage that answers this..."}
```

Then specify the local path or your own HF repo in `eval_config.yaml`.

## License

MIT License.
