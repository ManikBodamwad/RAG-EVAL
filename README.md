# rag-eval 🔬

**Production-grade CI/CD-integrated RAG Evaluation Pipeline**

[![PyPI version](https://badge.fury.io/py/rag-eval.svg)](https://badge.fury.io/py/rag-eval)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![RAG Eval CI](https://github.com/manikbodamwad/rag-eval/actions/workflows/rag_eval.yml/badge.svg)](https://github.com/manikbodamwad/rag-eval/actions/workflows/rag_eval.yml)
[![Dataset on HF](https://img.shields.io/badge/🤗%20Dataset-HF%20Hub-yellow)](https://huggingface.co/datasets/manikbodamwad/rag-eval-golden)
[![Grafana Dashboard](https://img.shields.io/badge/Grafana-Live%20Dashboard-orange?logo=grafana)](https://your-grafana-dashboard-url-here)

> **Companies ship AI chatbots but have no reliable way to know if a prompt or code change made things better or worse.**
>
> `rag-eval` solves this by acting as a **test coverage gate — but for AI quality**. It automatically evaluates every Pull Request and blocks the merge if output quality drops below a threshold.

---

## 🎯 The Problem This Solves

RAG pipelines are fragile: every corpus update, prompt change, or retriever tweak can silently degrade answer quality. Traditional CI/CD is blind to this.

```
Traditional CI:  Tests → Coverage → Lint              → Merge ✅
    AI-first CI: Tests → Coverage → RAG Eval → Quality Gate → Merge ✅
```

`rag-eval` adds the **RAG Eval → Quality Gate** layer to any CI pipeline. If faithfulness drops below 0.75, if answers diverge from ground truth, or if token efficiency degrades — **the PR is automatically blocked**.

---

## 🏗️ Architecture

```
                    ┌───────────────────────────────────────┐
                    │          GitHub Pull Request            │
                    └──────────────────┬────────────────────┘
                                       │ triggers
                                       ▼
                    ┌───────────────────────────────────────┐
                    │      GitHub Actions: rag_eval.yml      │
                    │                                        │
                    │  1. pip install rag-eval               │
                    │  2. Load 🤗 Golden Dataset (HF Hub)   │
                    │  3. Run Mock RAG Pipeline              │
                    │     └─ FAISS retrieval (local)         │
                    │     └─ Groq llama-3.3-70b generation   │
                    │  4. Ragas Evaluation (LLM Judge: Groq) │
                    │     ├─ Faithfulness                    │
                    │     ├─ Context Relevance               │
                    │     ├─ Answer Correctness              │
                    │     └─ Token Efficiency (custom)       │
                    │  5. Push scores → Grafana Cloud 📊    │
                    │  6. Post PR comment with table 💬      │
                    │  7. exit 0 ✅ or exit 1 ❌ gate        │
                    └───────────────────────────────────────┘
```

---

## 📊 Evaluation Metrics

| Metric | Type | What It Measures | Default Threshold |
|--------|------|------------------|-------------------|
| 🎯 **Faithfulness** | Ragas | Answers grounded in retrieved context (anti-hallucination) | ≥ 0.75 |
| 🔍 **Context Relevance** | Ragas | Retrieved context quality relative to the question | ≥ 0.70 |
| ✔️ **Answer Correctness** | Ragas | Accuracy vs. golden ground truth answer | ≥ 0.65 |
| ⚡ **Token Efficiency** | Custom SRE | Quality per output token — `correctness / log(1 + tokens)` | ≥ 0.50 |

**LLM Judge:** `groq/llama-3.3-70b-versatile` via LiteLLM — blazing Groq LPU inference, extremely generous free tier, hot-swappable via `JUDGE_MODEL` env var.

---

## ⚡ Quick Start

```bash
# Install
pip install rag-eval

# Set API key (free at console.groq.com)
export GROQ_API_KEY=gsk_your_key

# Run evaluation
rag-eval run

# View report
rag-eval report
```

---

## 💬 Example PR Comment

Every PR gets an automated quality report:

```markdown
## 🤖 RAG Evaluation Report — PR #42

## ❌ RAG Quality Gate: BLOCKED
> Answer Correctness dropped below threshold. Investigate recent prompt changes.

### 📊 Metric Scores

| Metric             | Score    | Visual       | Threshold | Status     |
|--------------------|----------|--------------|-----------|------------|
| 🎯 Faithfulness    | `0.8142` | `████████░░` | ≥ 0.75   | ✅ Pass    |
| 🔍 Context Rel.   | `0.7331` | `███████░░░` | ≥ 0.70   | ✅ Pass    |
| ✔️ Answer Corr.   | `0.6103` | `██████░░░░` | ≥ 0.65   | ❌ FAIL    |
| ⚡ Token Eff.     | `0.5844` | `█████░░░░░` | ≥ 0.50   | ✅ Pass    |

### ⚠️ Threshold Violations
- **Answer Correctness**: Got `0.6103`, need ≥ `0.65` (gap: `-0.0397`)

| 🤖 Judge Model | `groq/llama-3.3-70b-versatile` |
| 📦 Dataset     | `manikbodamwad/rag-eval-golden` |
| 📊 Samples     | `20`                            |
| ⏱️ Duration    | `127.3s`                        |
```

---

## 🔧 Add to Your CI in 3 Steps

### Step 1: Copy the workflow file

```yaml
# .github/workflows/rag_eval.yml
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

### Step 2: Set GitHub Secrets

Go to **Settings → Secrets and variables → Actions → New repository secret**:

| Secret | Required | Description |
|--------|----------|-------------|
| `GROQ_API_KEY` | ✅ Yes | Groq API key — [console.groq.com](https://console.groq.com) |
| `HF_TOKEN` | Only private datasets | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `GRAFANA_URL` | For dashboard | Grafana Cloud → Metrics → Send Metrics |
| `GRAFANA_USER` | For dashboard | Instance user ID from same page |
| `GRAFANA_API_KEY` | For dashboard | API token with MetricsPublisher scope |

### Step 3: Configure thresholds

```yaml
# eval_config.yaml
thresholds:
  faithfulness_min: 0.75
  context_relevance_min: 0.70
  answer_correctness_min: 0.65
  token_efficiency_min: 0.50

dataset:
  hf_repo: "your-username/rag-eval-golden"  # Your golden dataset on HF Hub
```

---

## 📦 Project Structure

```
rag-eval/
├── pyproject.toml                  # Package manifest (pip install rag-eval)
├── eval_config.yaml                # Thresholds — the regression gate config
├── golden_dataset.jsonl            # 20 QA pairs — also pushed to HF Hub
├── .env.example                    # Environment variable template
├── .github/
│   └── workflows/
│       └── rag_eval.yml            # 🔑 The CI quality gate
├── app/
│   ├── rag_pipeline.py             # Mock RAG app (FAISS + LangChain + Groq)
│   └── corpus/                     # 20 AI/ML concept documents
│       └── 01_transformers.txt ... 20_cicd_ml.txt
├── rag_eval/                       # 📦 Installable Python package
│   ├── __init__.py
│   ├── cli.py                      # `rag-eval` CLI: run + report commands
│   ├── evaluator.py                # Ragas orchestration + threshold gate
│   ├── reporter.py                 # JSON writer + PR comment formatter
│   └── grafana_push.py             # Influx Line Protocol → Grafana Cloud
└── scripts/
    └── push_dataset_to_hf.py       # One-time: push dataset to HF Hub
```

---

## 🛠️ CLI Reference

```
rag-eval run [OPTIONS]
  --config PATH           Path to eval_config.yaml  [default: eval_config.yaml]
  --output PATH           Path to save JSON report   [default: eval_report.json]
  --pr-number INT         GitHub PR number (auto-set in CI)
  --push-grafana          Push scores to Grafana Cloud
  --no-fail-on-gate       Dry-run — exit 0 even on failure
  --verbose               Enable debug logging

rag-eval report [OPTIONS]
  --input PATH            Path to eval_report.json   [default: eval_report.json]
  --format [table|json|markdown]                     [default: table]
```

---

## 🏗️ Local Development

```bash
git clone https://github.com/manikbodamwad/rag-eval
cd rag-eval
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env: set GROQ_API_KEY at minimum

# Push golden dataset to HF Hub (run once)
python scripts/push_dataset_to_hf.py --repo your-username/rag-eval-golden

# Update eval_config.yaml: set dataset.hf_repo to your repo

# Run evaluation
rag-eval run --verbose

# View formatted report
rag-eval report
```

---

## 📡 Grafana Cloud Dashboard

Evaluation scores are pushed as time-series metrics to Grafana Cloud after every CI run, enabling trend monitoring across PRs.

Metrics pushed: `faithfulness`, `context_relevance`, `answer_correctness`, `token_efficiency`, `overall_pass` — all tagged with `pr=<number>`.

**Setup:** Create a free account at [grafana.com](https://grafana.com) → Configure a Time Series panel → Query metric `rag_eval_scores` with label filter `{metric="faithfulness"}`.

---

## 🤗 Golden Dataset

**[manikbodamwad/rag-eval-golden](https://huggingface.co/datasets/manikbodamwad/rag-eval-golden)** — 20 expert QA pairs covering AI/ML concepts (Transformers, RAG, RLHF, Scaling Laws, MoE, Agents, LLMOps, Guardrails, CI/CD for ML, and more).

To build your own:

```jsonl
{"question": "What is X?", "ground_truth": "X is ...", "reference_context": "The passage that answers this..."}
```

```bash
python scripts/push_dataset_to_hf.py --repo your-username/your-dataset
```

---

## 🔗 Portfolio Context

This is **part of an AI infrastructure portfolio stack**:

| Project | Role |
|---------|------|
| **Project 2** — LLM Latency & Cost Router | Routes prompts to the right model intelligently |
| **Project 3** — RAG Eval Pipeline *(this)* | Evaluates if RAG output is faithful, correct, efficient |

Together: **route → evaluate → observe** — end-to-end AI infrastructure ownership.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**[PyPI](https://pypi.org/project/rag-eval)** · **[HF Dataset](https://huggingface.co/datasets/manikbodamwad/rag-eval-golden)** · **[Grafana Dashboard](https://your-grafana-url)** · **[GitHub](https://github.com/manikbodamwad/rag-eval)**

Built with [Ragas](https://ragas.io) · [Groq](https://groq.com) · [LangChain](https://langchain.com) · [FAISS](https://github.com/facebookresearch/faiss)

</div>
