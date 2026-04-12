"""
rag_eval/reporter.py

Saves JSON reports and formats Markdown for GitHub PR comments.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from rag_eval.evaluator import EvaluationResult


# Metric Display Configuration

METRIC_CONFIG = {
    "faithfulness": {
        "display_name": "Faithfulness",
        "description": "Answer grounded in retrieved context",
        "emoji": "",
    },
    "context_relevance": {
        "display_name": "Context Relevance",
        "description": "Retrieved context relevance to the question",
        "emoji": "",
    },
    "answer_correctness": {
        "display_name": "Answer Correctness",
        "description": "Accuracy vs. reference answer",
        "emoji": "",
    },
    "token_efficiency": {
        "display_name": "Token Efficiency",
        "description": "Quality per output token",
        "emoji": "",
    },
}


# Reporter class

class Reporter:
    """
    Generates evaluation reports.
    """

    def __init__(self, output_path: Optional[Path] = None):
        self.output_path = Path(output_path) if output_path else Path("eval_report.json")



    def save_json(self, result: EvaluationResult) -> Path:
        """Save evaluation result as a JSON report file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(result.to_json())

        file_size = self.output_path.stat().st_size
        print(f"Report saved: {self.output_path} ({file_size:,} bytes)")
        return self.output_path

    def load_json(self, path: Optional[Path] = None) -> dict:
        """Load an evaluation report from JSON."""
        report_path = Path(path) if path else self.output_path
        if not report_path.exists():
            raise FileNotFoundError(
                f"Report not found: {report_path}\n"
                f"Run 'rag-eval run' first to generate a report."
            )
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)



    def format_pr_comment(self, result: EvaluationResult) -> str:
        """
        Generate Markdown PR comment.
        """
        pr_ref = f"PR #{result.pr_number}" if result.pr_number else "Evaluation Run"
        timestamp = result.timestamp[:19].replace("T", " ") + " UTC" if result.timestamp else ""

        # Overall verdict
        if result.overall_pass:
            verdict_banner = "## ✅ RAG Quality Gate: PASSED"
            verdict_detail = "> All metrics meet their thresholds. This PR is safe to merge."
            verdict_color = "✅"
        else:
            failed_names = [
                METRIC_CONFIG.get(m, {}).get("display_name", m)
                for m in result.failed_metrics
            ]
            verdict_banner = "## ❌ RAG Quality Gate: BLOCKED"
            verdict_detail = (
                f"> **{len(result.failed_metrics)} metric(s) below threshold:** "
                f"{', '.join(failed_names)}. Investigate and fix before merging."
            )
            verdict_color = "❌"

        # Score table
        table_rows = []
        metrics_to_show = [
            ("faithfulness", result.faithfulness),
            ("context_relevance", result.context_relevance),
            ("answer_correctness", result.answer_correctness),
            ("token_efficiency", result.token_efficiency),
        ]

        # Get thresholds from result
        threshold_map = {
            "faithfulness": 0.75,
            "context_relevance": 0.70,
            "answer_correctness": 0.65,
            "token_efficiency": 0.50,
        }

        for metric_key, score in metrics_to_show:
            cfg = METRIC_CONFIG.get(metric_key, {})
            display = cfg.get("display_name", metric_key)
            emoji = cfg.get("emoji", "")
            threshold = threshold_map.get(metric_key, 0.0)
            passed = result.per_metric_pass.get(metric_key, True)
            status = "✅ Pass" if passed else "❌ **FAIL**"
            score_str = f"`{score:.4f}`"
            threshold_str = f"≥ {threshold}"

            # Score bar (visual indicator)
            bar_len = 10
            filled = round(score * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)

            table_rows.append(
                f"| {display} | {score_str} | `{bar}` | {threshold_str} | {status} |"
            )

        table_header = (
            "| Metric | Score | Visual | Threshold | Status |\n"
            "|--------|-------|--------|-----------|--------|"
        )
        table_body = "\n".join(table_rows)

        # Failure details section
        failure_details = ""
        if result.failed_metrics:
            failure_lines = []
            for metric, info in result.failed_metrics.items():
                display = METRIC_CONFIG.get(metric, {}).get("display_name", metric)
                failure_lines.append(
                    f"- **{display}**: Got `{info['actual']:.4f}`, "
                    f"need ≥ `{info['threshold']}` "
                    f"(gap: `-{info['gap']:.4f}`)"
                )
            failure_details = (
                "\n### Threshold Violations\n"
                + "\n".join(failure_lines)
            )

        meta_lines = [
            f"| Judge Model | `{result.judge_model}` |",
            f"| RAG Model | `{result.rag_model}` |",
            f"| Dataset | `{result.dataset_repo}` |",
            f"| Samples | `{result.num_samples}` |",
            f"| Duration | `{result.total_evaluation_time_s:.1f}s` |",
        ]
        if timestamp:
            meta_lines.append(f"| Evaluated | `{timestamp}` |")

        meta_table = (
            "| Property | Value |\n"
            "|----------|-------|\n"
            + "\n".join(meta_lines)
        )

        # Assemble full comment
        comment = f"""## RAG Evaluation Report — {pr_ref}

{verdict_banner}

{verdict_detail}

---

### Metric Scores

{table_header}
{table_body}
{failure_details}

---

### Evaluation Metadata

{meta_table}

---
<sup>Powered by <a href="https://github.com/manikbodamwad/rag-eval">rag-eval</a></sup>
"""
        return comment.strip()

    def print_rich_table(self, result: EvaluationResult) -> None:
        """Print formatted table to the terminal."""
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich import box
        except ImportError:
            # Fallback to simple print if Rich not available
            self._print_simple_table(result)
            return

        console = Console()

        # Overall verdict panel
        verdict_text = "✅ PASSED — All metrics above threshold" if result.overall_pass \
            else f"❌ FAILED — {len(result.failed_metrics)} metric(s) below threshold"
        verdict_style = "bold green" if result.overall_pass else "bold red"
        console.print(Panel(verdict_text, style=verdict_style, title="RAG Evaluation Verdict"))

        # Metrics table
        table = Table(
            title=f"Evaluation Scores",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Metric", style="bold white", width=22)
        table.add_column("Score", justify="center", width=10)
        table.add_column("Threshold", justify="center", width=10)
        table.add_column("Status", justify="center", width=12)

        threshold_map = {
            "faithfulness": 0.75,
            "context_relevance": 0.70,
            "answer_correctness": 0.65,
            "token_efficiency": 0.50,
        }

        for metric_key in ["faithfulness", "context_relevance", "answer_correctness", "token_efficiency"]:
            score = getattr(result, metric_key, 0.0)
            cfg = METRIC_CONFIG.get(metric_key, {})
            display = cfg.get("display_name", metric_key)
            threshold = threshold_map.get(metric_key, 0.0)
            passed = result.per_metric_pass.get(metric_key, True)

            score_style = "bold green" if passed else "bold red"
            status = "✅ Pass" if passed else "❌ FAIL"
            status_style = "green" if passed else "red"

            table.add_row(
                display,
                f"[{score_style}]{score:.4f}[/{score_style}]",
                f"≥ {threshold}",
                f"[{status_style}]{status}[/{status_style}]",
            )

        console.print(table)

        # Metadata
        console.print(f"\n[dim]Judge: {result.judge_model} | "
                      f"Dataset: {result.dataset_repo} | "
                      f"Samples: {result.num_samples} | "
                      f"Time: {result.total_evaluation_time_s:.1f}s[/dim]")

    def _print_simple_table(self, result: EvaluationResult) -> None:
        """Fallback simple table when Rich is not available."""
        print("\n" + "=" * 65)
        print("  RAG EVALUATION REPORT")
        print("=" * 65)
        print(f"{'Metric':<25} {'Score':>8} {'Threshold':>10} {'Status':>10}")
        print("-" * 65)
        metrics = [
            ("Faithfulness", result.faithfulness, 0.75),
            ("Context Relevance", result.context_relevance, 0.70),
            ("Answer Correctness", result.answer_correctness, 0.65),
            ("Token Efficiency", result.token_efficiency, 0.50),
        ]
        for name, score, threshold in metrics:
            status = "✅ PASS" if score >= threshold else "❌ FAIL"
            print(f"{name:<25} {score:>8.4f} {f'>= {threshold}':>10} {status:>10}")
        print("=" * 65)
        verdict = "✅ PASSED" if result.overall_pass else "❌ FAILED"
        print(f"\nOverall: {verdict} | Samples: {result.num_samples} | Time: {result.total_evaluation_time_s:.1f}s\n")
