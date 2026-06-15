import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import pytest
import yaml

from rag_eval.evaluator import RagEvaluator, EvaluationResult, MetricScore

# Sample configuration for testing
SAMPLE_CONFIG = {
    "thresholds": {
        "faithfulness_min": 0.75,
        "context_relevance_min": 0.70,
        "answer_correctness_min": 0.65,
        "token_efficiency_min": 0.50,
    },
    "model": {
        "judge": "groq/llama-3.3-70b-versatile",
        "rag_generator": "groq/llama-3.3-70b-versatile",
        "embeddings": "sentence-transformers/all-MiniLM-L6-v2",
    },
    "dataset": {
        "hf_repo": "manikbodamwad/rag-eval-golden",
        "split": "train",
        "local_fallback": "golden_dataset.jsonl",
    },
}


@pytest.fixture
def temp_config_file(tmp_path):
    config_path = tmp_path / "eval_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(SAMPLE_CONFIG, f)
    return config_path


def test_metric_score_post_init():
    metric = MetricScore(name="faithfulness", score=0.8, threshold=0.75, passed=True)
    assert metric.display_name == "Faithfulness"

    metric_custom = MetricScore(name="token_efficiency", score=0.6, threshold=0.5, passed=True, display_name="Custom Name")
    assert metric_custom.display_name == "Custom Name"


def test_evaluator_init_load_config(temp_config_file):
    evaluator = RagEvaluator(config_path=temp_config_file)
    assert evaluator.config["thresholds"]["faithfulness_min"] == 0.75
    assert evaluator.config["model"]["judge"] == "groq/llama-3.3-70b-versatile"


def test_evaluator_init_config_not_found():
    with pytest.raises(FileNotFoundError):
        RagEvaluator(config_path=Path("non_existent_file.yaml"))


@patch("datasets.load_dataset")
def test_load_dataset_hf_success(mock_load, temp_config_file):
    mock_dataset = [{"question": "Q1", "ground_truth": "A1", "reference_context": "C1"}]
    mock_load.return_value = mock_dataset

    evaluator = RagEvaluator(config_path=temp_config_file)
    samples = evaluator._load_dataset()

    assert len(samples) == 1
    assert samples[0]["question"] == "Q1"
    mock_load.assert_called_once_with("manikbodamwad/rag-eval-golden", split="train")


@patch("datasets.load_dataset", side_effect=Exception("HF offline"))
def test_load_dataset_local_fallback(mock_load, temp_config_file, tmp_path):
    # Setup local file content
    local_file = tmp_path / "golden_dataset.jsonl"
    sample_data = {"question": "Q2", "ground_truth": "A2", "reference_context": "C2"}
    with open(local_file, "w") as f:
        f.write(json.dumps(sample_data) + "\n")

    # Update evaluator config to point to the temp local fallback
    evaluator = RagEvaluator(config_path=temp_config_file)
    evaluator.config["dataset"]["local_fallback"] = str(local_file)

    samples = evaluator._load_dataset()

    assert len(samples) == 1
    assert samples[0]["question"] == "Q2"


def test_compute_token_efficiency_empty(temp_config_file):
    evaluator = RagEvaluator(config_path=temp_config_file)
    # No token counts -> uses default 100 token baseline
    eff = evaluator._compute_token_efficiency([], 0.8)
    # expected = 0.8 / log(101) ≈ 0.1733
    assert math.isclose(eff, 0.8 / math.log(101), rel_tol=1e-3)


def test_compute_token_efficiency_custom(temp_config_file):
    evaluator = RagEvaluator(config_path=temp_config_file)
    samples = [
        {"token_counts": {"output_tokens": 100}},
        {"token_counts": {"output_tokens": 100}},
    ]
    eff = evaluator._compute_token_efficiency(samples, 0.8)
    assert math.isclose(eff, 0.8, rel_tol=1e-3)

    # Verbose case (e.g. 500 tokens instead of 100 baseline)
    samples_verbose = [
        {"token_counts": {"output_tokens": 500}},
    ]
    eff_verbose = evaluator._compute_token_efficiency(samples_verbose, 0.8)
    # average tokens = 500
    # log(501) ≈ 6.216
    # baseline = log(101) ≈ 4.615
    # efficiency = 0.8 / 6.216 / (1 / 4.615) = 0.8 * 4.615 / 6.216 ≈ 0.594
    assert eff_verbose < 0.65
    assert eff_verbose > 0.55


def test_apply_gate_pass(temp_config_file):
    evaluator = RagEvaluator(config_path=temp_config_file)
    scores = {
        "faithfulness": 0.80,
        "context_relevance": 0.75,
        "answer_correctness": 0.70,
        "token_efficiency": 0.60,
    }
    passed, failures = evaluator._apply_gate(scores)
    assert passed is True
    assert len(failures) == 0


def test_apply_gate_fail(temp_config_file):
    evaluator = RagEvaluator(config_path=temp_config_file)
    scores = {
        "faithfulness": 0.70,  # Below 0.75
        "context_relevance": 0.75,
        "answer_correctness": 0.60,  # Below 0.65
        "token_efficiency": 0.60,
    }
    passed, failures = evaluator._apply_gate(scores)
    assert passed is False
    assert "faithfulness" in failures
    assert "answer_correctness" in failures
    assert failures["faithfulness"]["actual"] == 0.70
    assert failures["faithfulness"]["threshold"] == 0.75


@patch("rag_eval.evaluator.RagEvaluator._load_dataset")
@patch("rag_eval.evaluator.RagEvaluator._build_ragas_llm")
@patch("rag_eval.evaluator.RagEvaluator._run_rag_pipeline")
@patch("rag_eval.evaluator.RagEvaluator._compute_ragas_metrics")
def test_evaluate_workflow(mock_metrics, mock_run, mock_build, mock_load, temp_config_file):
    # Mock data loading
    mock_load.return_value = [{"question": "Q", "ground_truth": "G"}]

    # Mock LLM judge creation
    mock_build.return_value = (MagicMock(), "groq/llama-3.3-70b-versatile")

    # Mock RAG pipeline queries
    mock_run.return_value = [{
        "question": "Q",
        "answer": "A",
        "contexts": ["C"],
        "ground_truth": "G",
        "token_counts": {"input_tokens": 10, "output_tokens": 100}
    }]

    # Mock Ragas metrics computations
    mock_metrics.return_value = {
        "faithfulness": 0.85,
        "context_relevance": 0.80,
        "answer_correctness": 0.75,
    }

    evaluator = RagEvaluator(config_path=temp_config_file)
    result = evaluator.evaluate()

    assert isinstance(result, EvaluationResult)
    assert result.faithfulness == 0.85
    assert result.context_relevance == 0.80
    assert result.answer_correctness == 0.75
    assert result.token_efficiency == 0.75  # Since output tokens is 100, matches correctness
    assert result.overall_pass is True
    assert len(result.failed_metrics) == 0
