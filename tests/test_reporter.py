import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from rag_eval.evaluator import EvaluationResult
from rag_eval.reporter import Reporter


@pytest.fixture
def sample_pass_result():
    return EvaluationResult(
        faithfulness=0.85,
        context_relevance=0.80,
        answer_correctness=0.75,
        token_efficiency=0.75,
        overall_pass=True,
        per_metric_pass={
            "faithfulness": True,
            "context_relevance": True,
            "answer_correctness": True,
            "token_efficiency": True,
        },
        failed_metrics={},
        timestamp="2026-06-15T12:00:00Z",
        pr_number=42,
        judge_model="groq/llama-3.3-70b-versatile",
        rag_model="groq/llama-3.3-70b-versatile",
        dataset_repo="manikbodamwad/rag-eval-golden",
        num_samples=1,
        total_evaluation_time_s=10.5,
    )


@pytest.fixture
def sample_fail_result():
    return EvaluationResult(
        faithfulness=0.60,  # Below threshold
        context_relevance=0.80,
        answer_correctness=0.75,
        token_efficiency=0.75,
        overall_pass=False,
        per_metric_pass={
            "faithfulness": False,
            "context_relevance": True,
            "answer_correctness": True,
            "token_efficiency": True,
        },
        failed_metrics={
            "faithfulness": {
                "actual": 0.60,
                "threshold": 0.75,
                "gap": 0.15,
            }
        },
        timestamp="2026-06-15T12:00:00Z",
        pr_number=42,
        judge_model="groq/llama-3.3-70b-versatile",
        rag_model="groq/llama-3.3-70b-versatile",
        dataset_repo="manikbodamwad/rag-eval-golden",
        num_samples=1,
        total_evaluation_time_s=10.5,
    )


def test_reporter_save_load_json(tmp_path, sample_pass_result):
    report_file = tmp_path / "test_report.json"
    reporter = Reporter(output_path=report_file)

    saved_path = reporter.save_json(sample_pass_result)
    assert saved_path == report_file
    assert report_file.exists()

    loaded_data = reporter.load_json()
    assert loaded_data["faithfulness"] == 0.85
    assert loaded_data["overall_pass"] is True
    assert loaded_data["pr_number"] == 42


def test_reporter_load_json_not_found():
    reporter = Reporter(output_path=Path("non_existent_report.json"))
    with pytest.raises(FileNotFoundError):
        reporter.load_json()


def test_reporter_format_pr_comment_pass(sample_pass_result):
    reporter = Reporter()
    comment = reporter.format_pr_comment(sample_pass_result)

    assert "RAG Evaluation Report — PR #42" in comment
    assert "✅ RAG Quality Gate: PASSED" in comment
    assert "All metrics meet their thresholds. This PR is safe to merge." in comment
    assert "| Faithfulness | `0.8500` |" in comment
    assert "groq/llama-3.3-70b-versatile" in comment


def test_reporter_format_pr_comment_fail(sample_fail_result):
    reporter = Reporter()
    comment = reporter.format_pr_comment(sample_fail_result)

    assert "RAG Evaluation Report — PR #42" in comment
    assert "❌ RAG Quality Gate: BLOCKED" in comment
    assert "1 metric(s) below threshold" in comment
    assert "### Threshold Violations" in comment
    assert "**Faithfulness**: Got `0.6000`, need ≥ `0.75` (gap: `-0.1500`)" in comment


def test_reporter_print_simple_table(capsys, sample_pass_result):
    reporter = Reporter()
    reporter._print_simple_table(sample_pass_result)
    captured = capsys.readouterr()

    assert "RAG EVALUATION REPORT" in captured.out
    assert "Faithfulness" in captured.out
    assert "0.8500" in captured.out
    assert "✅ PASS" in captured.out


@patch("rich.console.Console")
def test_reporter_print_rich_table(mock_console_cls, sample_fail_result):
    mock_console = MagicMock()
    mock_console_cls.return_value = mock_console

    reporter = Reporter()
    reporter.print_rich_table(sample_fail_result)

    # Verify Console was instantiated and print was called
    mock_console_cls.assert_called_once()
    assert mock_console.print.call_count >= 2
