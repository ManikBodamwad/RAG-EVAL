"""
rag_eval/evaluator.py

Handles evaluation pipeline orchestration, including dataset loading, RAG queries, and Ragas metrics calculation against defined thresholds.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


# Data Structures
@dataclass
class MetricScore:
    """Result for a single evaluation metric."""
    name: str
    score: float
    threshold: float
    passed: bool
    display_name: str = ""

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.name.replace("_", " ").title()


@dataclass
class EvaluationResult:
    """Complete evaluation result from a single pipeline run."""
    # Core metric scores
    faithfulness: float = 0.0
    context_relevance: float = 0.0
    answer_correctness: float = 0.0
    token_efficiency: float = 0.0

    # Gate results
    overall_pass: bool = False
    per_metric_pass: dict[str, bool] = field(default_factory=dict)
    failed_metrics: dict[str, dict] = field(default_factory=dict)

    # Metadata
    timestamp: str = ""
    pr_number: Optional[int] = None
    judge_model: str = ""
    rag_model: str = ""
    dataset_repo: str = ""
    num_samples: int = 0
    total_evaluation_time_s: float = 0.0

    # Per-sample details (for debugging)
    raw_scores: dict[str, Any] = field(default_factory=dict)
    sample_token_counts: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# Evaluator
class RagEvaluator:
    """
    Orchestrates RAG pipeline evaluation using Ragas.
    """

    DEFAULT_CONFIG_PATH = Path("eval_config.yaml")

    def __init__(
        self,
        config_path: Optional[Path] = None,
        pr_number: Optional[int] = None,
    ):
        self.config_path = Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH
        self.pr_number = pr_number
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Load and validate eval_config.yaml."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                f"Run from the project root or specify --config path."
            )
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {self.config_path}")
        return config

    def _load_dataset(self) -> list[dict]:
        """
        Load the golden evaluation dataset.
        """
        hf_repo = self.config.get("dataset", {}).get("hf_repo", "")
        local_fallback = self.config.get("dataset", {}).get("local_fallback", "golden_dataset.jsonl")

        # Try HF Hub first
        if hf_repo:
            try:
                from datasets import load_dataset
                logger.info(f"Loading golden dataset from HF Hub: {hf_repo}")
                split = self.config.get("dataset", {}).get("split", "train")
                hf_dataset = load_dataset(hf_repo, split=split)
                samples = list(hf_dataset)
                logger.info(f"Loaded {len(samples)} samples from HF Hub")
                return samples
            except Exception as e:
                logger.warning(f"HF Hub load failed ({e}), falling back to local dataset")

        # Fallback to local file
        local_path = Path(local_fallback)
        if not local_path.exists():
            raise FileNotFoundError(
                f"Neither HF Hub dataset ('{hf_repo}') nor local fallback ('{local_fallback}') found.\n"
                f"Run: python scripts/push_dataset_to_hf.py  to push the dataset to HF Hub."
            )
        logger.info(f"Loading golden dataset from local file: {local_path}")
        samples = []
        with open(local_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        logger.info(f"Loaded {len(samples)} samples from local dataset")
        return samples

    def _build_ragas_llm(self):
        """
        Create the Ragas-compatible LLM judge wrapper using LiteLLM.
        Hot-swappable via JUDGE_MODEL environment variable.
        """
        try:
            from ragas.llms import llm_factory
        except ImportError:
            raise ImportError("Ragas not installed. Run: pip install ragas>=0.2.0")

        judge_model = (
            os.environ.get("JUDGE_MODEL")
            or self.config.get("model", {}).get("judge", "groq/llama-3.3-70b-versatile")
        )
        logger.info(f"Initializing LLM judge: {judge_model}")
        return llm_factory(model=judge_model), judge_model

    def _run_rag_pipeline(self, samples: list[dict]) -> list[dict]:
        """
        Run each golden dataset question through the RAG pipeline.
        Returns a list of dicts with keys: question, answer, contexts, ground_truth, token_counts.
        """
        # Lazy import to avoid circular imports
        import sys
        from pathlib import Path

        # Add the project root to path so app/ is importable
        project_root = self.config_path.parent
        sys.path.insert(0, str(project_root))

        from app.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline()
        pipeline.build_index()  # Build once, reuse for all questions

        rag_samples = []
        for i, sample in enumerate(samples):
            question = sample["question"]
            ground_truth = sample.get("ground_truth", "")

            logger.info(f"  [{i+1}/{len(samples)}] Querying RAG: {question[:60]}...")
            try:
                result = pipeline.query(question)
                rag_samples.append({
                    "question": question,
                    "answer": result.answer,
                    "contexts": result.contexts,
                    "ground_truth": ground_truth,
                    "token_counts": {
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                    },
                })
            except Exception as e:
                logger.error(f"RAG pipeline failed for question {i+1}: {e}")
                rag_samples.append({
                    "question": question,
                    "answer": f"ERROR: Pipeline failed — {e}",
                    "contexts": [""],
                    "ground_truth": ground_truth,
                    "token_counts": {"input_tokens": 0, "output_tokens": 0},
                })

        return rag_samples

    def _compute_ragas_metrics(
        self,
        rag_samples: list[dict],
        eval_llm,
    ) -> dict[str, float]:
        """
        Run Ragas evaluation on the assembled pipeline outputs.
        Returns dict of metric_name -> average float score.
        """
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                AnswerCorrectness,
                ContextRelevance,
                Faithfulness,
            )
        except ImportError as e:
            raise ImportError(f"Missing evaluation dependency: {e}")

        # Build the Ragas EvaluationDataset format
        eval_data = {
            "question": [s["question"] for s in rag_samples],
            "answer": [s["answer"] for s in rag_samples],
            "contexts": [s["contexts"] for s in rag_samples],
            "ground_truth": [s["ground_truth"] for s in rag_samples],
            "reference": [s["ground_truth"] for s in rag_samples],  # Some Ragas versions use 'reference'
        }

        try:
            from ragas import EvaluationDataset
            ragas_dataset = EvaluationDataset.from_dict(eval_data)
        except (ImportError, AttributeError):
            # Fallback for older Ragas versions using HF Dataset
            ragas_dataset = Dataset.from_dict(eval_data)

        # Initialize metrics with our LLM judge
        metrics = [
            Faithfulness(llm=eval_llm),
            ContextRelevance(llm=eval_llm),
            AnswerCorrectness(llm=eval_llm),
        ]

        logger.info("Running Ragas evaluation (this may take a few minutes with Groq)...")
        results = evaluate(
            dataset=ragas_dataset,
            metrics=metrics,
            llm=eval_llm,
            raise_exceptions=False,
        )

        # Extract scalar averages
        scores = {}
        results_df = results.to_pandas()
        for col in results_df.columns:
            if col not in ("question", "answer", "contexts", "ground_truth", "reference"):
                scores[col] = float(results_df[col].dropna().mean())

        return scores

    def _compute_token_efficiency(self, rag_samples: list[dict], answer_correctness: float) -> float:
        """
        Custom Token Efficiency metric.

        Formula: answer_correctness / log(1 + avg_output_tokens)
        Normalized to [0, 1] against a reference baseline of 100 output tokens.

        This penalizes verbose answers that score similarly to concise ones,
        signaling cost-aware AI infrastructure to SRE reviewers.
        """
        token_counts = [
            s["token_counts"].get("output_tokens", 0)
            for s in rag_samples
            if s["token_counts"].get("output_tokens", 0) > 0
        ]

        if not token_counts:
            logger.warning("No token count data available, using default token efficiency")
            return answer_correctness / math.log(1 + 100)  # Assume 100 tokens baseline

        avg_output_tokens = sum(token_counts) / len(token_counts)

        # Normalize: score / log(1 + tokens), then scale by log(1 + 100) as baseline
        # This gives ~1.0 when answer_correctness=1.0 and output_tokens=100
        baseline = math.log(1 + 100)  # log(101) ≈ 4.615
        raw_efficiency = answer_correctness / math.log(1 + avg_output_tokens)
        normalized = min(raw_efficiency / (1.0 / baseline), 1.0)  # Cap at 1.0

        logger.debug(
            f"Token efficiency: correctness={answer_correctness:.3f}, "
            f"avg_tokens={avg_output_tokens:.0f}, efficiency={normalized:.3f}"
        )
        return round(normalized, 4)

    def _apply_gate(self, scores: dict[str, float]) -> tuple[bool, dict[str, dict]]:
        """
        Apply threshold gates from config.
        Returns (all_passed, {metric: {actual, threshold}} for failed metrics).
        """
        thresholds = self.config.get("thresholds", {})
        failures = {}

        metric_map = {
            "faithfulness_min": "faithfulness",
            "context_relevance_min": "context_relevance",
            "answer_correctness_min": "answer_correctness",
            "token_efficiency_min": "token_efficiency",
        }

        for threshold_key, metric_name in metric_map.items():
            min_val = thresholds.get(threshold_key, 0.0)
            actual = scores.get(metric_name, 0.0)

            if actual < min_val:
                failures[metric_name] = {
                    "actual": round(actual, 4),
                    "threshold": min_val,
                    "gap": round(min_val - actual, 4),
                }
                logger.warning(
                    f"  ❌ {metric_name}: {actual:.4f} < {min_val} (threshold) — GATE TRIGGERED"
                )
            else:
                logger.info(f"  ✅ {metric_name}: {actual:.4f} >= {min_val} — OK")

        return len(failures) == 0, failures

    def evaluate(self) -> EvaluationResult:
        """
        Run the full evaluation pipeline.
        """
        from datetime import datetime, timezone

        t_start = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()

        logger.info("Starting evaluation pipeline...")

        logger.info("Loading golden dataset...")
        samples = self._load_dataset()
        logger.info(f"  Loaded {len(samples)} evaluation samples")

        # Step 2: Initialize LLM judge
        logger.info("Initializing LLM judge...")
        eval_llm, judge_model = self._build_ragas_llm()

        # Step 3: Run RAG pipeline on all samples
        logger.info(f"Running RAG pipeline on {len(samples)} questions...")
        rag_samples = self._run_rag_pipeline(samples)

        # Step 4: Compute Ragas metrics
        logger.info("Computing evaluation metrics...")
        ragas_scores = self._compute_ragas_metrics(rag_samples, eval_llm)
        logger.info(f"Raw scores: {ragas_scores}")

        # Normalize metric names (Ragas may return different keys)
        faithfulness = ragas_scores.get("faithfulness", ragas_scores.get("Faithfulness", 0.0))
        context_relevance = ragas_scores.get(
            "context_relevance",
            ragas_scores.get("ContextRelevance", ragas_scores.get("context_precision", 0.0))
        )
        answer_correctness = ragas_scores.get(
            "answer_correctness",
            ragas_scores.get("AnswerCorrectness", 0.0)
        )

        # Step 5: Compute custom token efficiency metric
        logger.info("Computing token efficiency metric...")
        token_efficiency = self._compute_token_efficiency(rag_samples, answer_correctness)

        all_scores = {
            "faithfulness": round(faithfulness, 4),
            "context_relevance": round(context_relevance, 4),
            "answer_correctness": round(answer_correctness, 4),
            "token_efficiency": token_efficiency,
        }
        logger.info(f"Final scores: {all_scores}")

        # Step 6: Apply regression gate
        logger.info("Applying regression gate...")
        overall_pass, failed_metrics = self._apply_gate(all_scores)

        per_metric_pass = {
            metric: metric not in failed_metrics
            for metric in all_scores
        }

        total_time = time.perf_counter() - t_start

        # Build result
        rag_model = (
            os.environ.get("RAG_MODEL")
            or self.config.get("model", {}).get("rag_generator", "groq/llama-3.3-70b-versatile")
        )
        dataset_repo = self.config.get("dataset", {}).get("hf_repo", "local")

        result = EvaluationResult(
            faithfulness=all_scores["faithfulness"],
            context_relevance=all_scores["context_relevance"],
            answer_correctness=all_scores["answer_correctness"],
            token_efficiency=all_scores["token_efficiency"],
            overall_pass=overall_pass,
            per_metric_pass=per_metric_pass,
            failed_metrics=failed_metrics,
            timestamp=timestamp,
            pr_number=self.pr_number,
            judge_model=judge_model,
            rag_model=rag_model,
            dataset_repo=dataset_repo,
            num_samples=len(samples),
            total_evaluation_time_s=round(total_time, 2),
            raw_scores=ragas_scores,
            sample_token_counts=[s["token_counts"] for s in rag_samples],
        )

        verdict = "PASSED" if overall_pass else "FAILED"
        logger.info(f"Evaluation complete: {verdict}")
        logger.info(f"Time: {total_time:.1f}s | Samples: {len(samples)}")

        return result
