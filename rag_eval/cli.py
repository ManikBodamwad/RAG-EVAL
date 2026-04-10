"""
rag_eval/cli.py — Command-Line Interface for rag-eval

Entry point: `rag-eval` (installed via pyproject.toml [project.scripts])

Commands:
  rag-eval run     — Run the full evaluation pipeline
  rag-eval report  — Print the last evaluation report as a formatted table
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import click

from rag_eval import __version__


def _setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity flag."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy third-party loggers unless in debug mode
    if not verbose:
        for noisy in ["httpx", "httpcore", "openai", "groq", "litellm", "sentence_transformers"]:
            logging.getLogger(noisy).setLevel(logging.WARNING)


# ──────────────────────────────────────────────────────────────────────────────
# Root CLI group
# ──────────────────────────────────────────────────────────────────────────────

@click.group()
@click.version_option(version=__version__, prog_name="rag-eval")
def cli():
    """
    \b
    ┌─────────────────────────────────────────────┐
    │  rag-eval — Automated RAG Evaluation Pipeline  │
    │  Quality gate for AI chatbots via Ragas + Groq │
    └─────────────────────────────────────────────┘

    Evaluate your RAG pipeline, track quality over PRs,
    and block merges when scores drop below thresholds.

    \b
    Quick Start:
      export GROQ_API_KEY=your_key
      rag-eval run
      rag-eval report
    """
    pass


# ──────────────────────────────────────────────────────────────────────────────
# rag-eval run
# ──────────────────────────────────────────────────────────────────────────────

@cli.command("run")
@click.option(
    "--config",
    "-c",
    default="eval_config.yaml",
    show_default=True,
    type=click.Path(exists=False),
    help="Path to eval_config.yaml (thresholds + model config).",
)
@click.option(
    "--output",
    "-o",
    default="eval_report.json",
    show_default=True,
    type=click.Path(),
    help="Path to save the JSON evaluation report.",
)
@click.option(
    "--pr-number",
    default=None,
    type=int,
    envvar="PR_NUMBER",
    help="GitHub PR number (set automatically in CI via ${{ github.event.number }}).",
)
@click.option(
    "--push-grafana/--no-push-grafana",
    default=True,
    show_default=True,
    help="Push scores to Grafana Cloud (requires GRAFANA_* env vars).",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable verbose debug logging.",
)
@click.option(
    "--fail-on-gate/--no-fail-on-gate",
    default=True,
    show_default=True,
    help="Exit with code 1 if regression gate triggers (default: True). "
         "Use --no-fail-on-gate for dry runs.",
)
def run_evaluation(
    config: str,
    output: str,
    pr_number: Optional[int],
    push_grafana: bool,
    verbose: bool,
    fail_on_gate: bool,
):
    """
    Run the full RAG evaluation pipeline.

    \b
    Steps:
      1. Load golden dataset from HF Hub (or local fallback)
      2. Run each question through the mock RAG pipeline
      3. Evaluate with Ragas (Faithfulness, Context Relevance, Answer Correctness)
      4. Compute Token Efficiency (custom SRE cost metric)
      5. Apply regression gate from eval_config.yaml
      6. Save JSON report and push scores to Grafana Cloud
      7. Exit 0 (pass) or 1 (gate triggered — blocks PR merge)

    \b
    Required Environment Variables:
      GROQ_API_KEY  — Groq API key for LLM judge + RAG generator
      HF_TOKEN      — Hugging Face token (only needed if dataset is private)

    Optional Environment Variables:
      JUDGE_MODEL   — Override LLM judge (default: groq/llama-3.3-70b-versatile)
      RAG_MODEL     — Override RAG generator (default: groq/llama-3.3-70b-versatile)
      PR_NUMBER     — GitHub PR number (set automatically in CI)
      GRAFANA_URL, GRAFANA_USER, GRAFANA_API_KEY — For Grafana Cloud push
    """
    _setup_logging(verbose)
    logger = logging.getLogger("rag_eval.cli")

    # Check for required API key
    if not os.environ.get("GROQ_API_KEY"):
        click.echo(
            click.style(
                "❌ Error: GROQ_API_KEY environment variable not set.\n"
                "   Get a free key at: https://console.groq.com\n"
                "   Then run: export GROQ_API_KEY=gsk_your_key",
                fg="red",
            ),
            err=True,
        )
        sys.exit(1)

    click.echo(click.style(f"\n🚀 rag-eval v{__version__} — Starting Evaluation\n", bold=True))

    try:
        from rag_eval.evaluator import RagEvaluator
        from rag_eval.reporter import Reporter
        from rag_eval.grafana_push import push_from_result

        # Run evaluation
        evaluator = RagEvaluator(
            config_path=Path(config),
            pr_number=pr_number,
        )
        result = evaluator.evaluate()

        # Save JSON report
        reporter = Reporter(output_path=Path(output))
        reporter.save_json(result)

        # Print formatted table
        click.echo()
        reporter.print_rich_table(result)

        # Push to Grafana Cloud (non-fatal)
        if push_grafana:
            run_id = os.environ.get("GITHUB_RUN_ID")
            click.echo("\n📡 Pushing scores to Grafana Cloud...")
            success = push_from_result(result, run_id=run_id)
            if not success:
                click.echo(
                    click.style("  ℹ️  Grafana push skipped or failed (non-fatal)", fg="yellow")
                )

        # Generate and display PR comment format (useful in CI logs)
        pr_comment = reporter.format_pr_comment(result)
        click.echo("\n" + "─" * 65)
        click.echo("📝 PR Comment Preview (this will be posted to GitHub):")
        click.echo("─" * 65)
        click.echo(pr_comment)

        # Save PR comment to file for GitHub Actions to pick up
        pr_comment_path = Path(output).parent / "pr_comment.md"
        pr_comment_path.write_text(pr_comment, encoding="utf-8")
        click.echo(f"\n💬 PR comment saved: {pr_comment_path}")

        # Regression gate decision
        if result.overall_pass:
            click.echo(
                click.style(
                    "\n✅ Regression gate: PASSED — All metrics above thresholds",
                    fg="green", bold=True
                )
            )
            sys.exit(0)
        else:
            failed = ", ".join(result.failed_metrics.keys())
            click.echo(
                click.style(
                    f"\n❌ Regression gate: TRIGGERED — Failed metrics: {failed}",
                    fg="red", bold=True
                )
            )
            if fail_on_gate:
                click.echo(
                    click.style(
                        "   Exiting with code 1 — this will block the PR from merging.",
                        fg="red"
                    )
                )
                sys.exit(1)
            else:
                click.echo(
                    click.style(
                        "   --no-fail-on-gate set: exiting with code 0 (dry run mode)",
                        fg="yellow"
                    )
                )
                sys.exit(0)

    except FileNotFoundError as e:
        click.echo(click.style(f"\n❌ File not found: {e}", fg="red"), err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(click.style(f"\n❌ Evaluation failed: {e}", fg="red"), err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# rag-eval report
# ──────────────────────────────────────────────────────────────────────────────

@cli.command("report")
@click.option(
    "--input",
    "-i",
    "input_path",
    default="eval_report.json",
    show_default=True,
    type=click.Path(),
    help="Path to the JSON evaluation report to display.",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    default="table",
    show_default=True,
    type=click.Choice(["table", "json", "markdown"], case_sensitive=False),
    help="Output format: table (rich), json (raw), or markdown (PR comment format).",
)
def show_report(input_path: str, output_format: str):
    """
    Print the last evaluation report in a formatted view.

    \b
    Formats:
      table     — Rich formatted table (default, for terminal viewing)
      json      — Raw JSON report
      markdown  — GitHub PR comment format
    """
    _setup_logging(False)

    from rag_eval.reporter import Reporter
    from rag_eval.evaluator import EvaluationResult

    reporter = Reporter(output_path=Path(input_path))

    try:
        data = reporter.load_json(path=Path(input_path))
    except FileNotFoundError as e:
        click.echo(click.style(f"❌ {e}", fg="red"), err=True)
        sys.exit(1)

    if output_format == "json":
        click.echo(json.dumps(data, indent=2, default=str))
        return

    # Reconstruct EvaluationResult from dict
    result = EvaluationResult(
        faithfulness=data.get("faithfulness", 0.0),
        context_relevance=data.get("context_relevance", 0.0),
        answer_correctness=data.get("answer_correctness", 0.0),
        token_efficiency=data.get("token_efficiency", 0.0),
        overall_pass=data.get("overall_pass", False),
        per_metric_pass=data.get("per_metric_pass", {}),
        failed_metrics=data.get("failed_metrics", {}),
        timestamp=data.get("timestamp", ""),
        pr_number=data.get("pr_number"),
        judge_model=data.get("judge_model", ""),
        rag_model=data.get("rag_model", ""),
        dataset_repo=data.get("dataset_repo", ""),
        num_samples=data.get("num_samples", 0),
        total_evaluation_time_s=data.get("total_evaluation_time_s", 0.0),
    )

    if output_format == "markdown":
        click.echo(reporter.format_pr_comment(result))
    else:
        reporter.print_rich_table(result)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
