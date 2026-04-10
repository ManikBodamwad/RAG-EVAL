"""
rag_eval/grafana_push.py — Push Evaluation Scores to Grafana Cloud

Pushes evaluation metrics as time-series data to Grafana Cloud using
Influx Line Protocol over HTTP — simpler than Prometheus Remote Write
(no protobuf/snappy needed), works with a single requests.post() call.

Required Environment Variables:
  GRAFANA_URL       — Influx write endpoint
                      (e.g., https://influx-prod-XX.grafana.net/api/v1/push/influx/write)
  GRAFANA_USER      — Grafana Cloud metrics instance user ID (numeric string)
  GRAFANA_API_KEY   — API token with MetricsPublisher scope

Setup Instructions:
  1. Go to https://grafana.com → log in → Your Stack → Metrics
  2. Click "Send Metrics" → "Grafana Alloy (or direct write)"
  3. Copy the URL (Prometheus remote write URL), then replace /api/prom/push
     with /api/v1/push/influx/write for the Influx endpoint
  4. Generate an API token with scope: metrics:write
  5. Note your instance User ID (the numeric value)
  6. Add GRAFANA_URL, GRAFANA_USER, GRAFANA_API_KEY to GitHub Secrets
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)


def push_evaluation_scores(
    scores: dict[str, float],
    pr_number: Optional[int] = None,
    run_id: Optional[str] = None,
    overall_pass: bool = True,
    grafana_url: Optional[str] = None,
    grafana_user: Optional[str] = None,
    grafana_api_key: Optional[str] = None,
) -> bool:
    """
    Push evaluation scores to Grafana Cloud via Influx Line Protocol.

    Args:
        scores: Dict of metric_name -> float score
        pr_number: GitHub PR number (used as a metric tag)
        run_id: GitHub Actions run ID (used as a metric tag)
        overall_pass: Whether the evaluation passed all thresholds
        grafana_url/user/api_key: Override env vars for testing

    Returns:
        True if push succeeded, False otherwise (non-fatal — CI should not fail on this)
    """
    try:
        import requests
    except ImportError:
        logger.warning("requests library not installed, skipping Grafana push")
        return False

    url = grafana_url or os.environ.get("GRAFANA_URL")
    user = grafana_user or os.environ.get("GRAFANA_USER")
    api_key = grafana_api_key or os.environ.get("GRAFANA_API_KEY")

    if not all([url, user, api_key]):
        logger.info(
            "Grafana credentials not set (GRAFANA_URL, GRAFANA_USER, GRAFANA_API_KEY). "
            "Skipping metrics push."
        )
        return False

    timestamp_ns = int(time.time() * 1e9)  # nanoseconds (Influx standard)

    # Build Influx Line Protocol payload
    # Format: measurement,tag1=val1,tag2=val2 field1=val1,field2=val2 timestamp
    tags = []
    if pr_number:
        tags.append(f"pr={pr_number}")
    if run_id:
        tags.append(f"run_id={run_id}")
    tags.append(f"pass={'true' if overall_pass else 'false'}")

    tag_str = ",".join(tags)
    lines = []

    # Push each metric as a separate time series
    for metric_name, score in scores.items():
        if not isinstance(score, (int, float)) or math.isnan(score):
            continue

        measurement = f"rag_eval_scores"
        tag_with_metric = f"{tag_str},metric={metric_name}" if tag_str else f"metric={metric_name}"
        lines.append(f"{measurement},{tag_with_metric} value={score:.6f} {timestamp_ns}")

    # Push overall pass/fail as a boolean metric (1.0 = pass, 0.0 = fail)
    pass_value = 1.0 if overall_pass else 0.0
    overall_tag = f"{tag_str},metric=overall_pass" if tag_str else "metric=overall_pass"
    lines.append(f"rag_eval_scores,{overall_tag} value={pass_value} {timestamp_ns}")

    payload = "\n".join(lines)
    logger.debug(f"Grafana push payload:\n{payload}")

    try:
        response = requests.post(
            url,
            data=payload,
            auth=(user, api_key),
            headers={"Content-Type": "text/plain"},
            timeout=10,
        )

        if response.status_code in (200, 204):
            logger.info(
                f"✅ Grafana push successful — {len(lines)} metrics pushed "
                f"(HTTP {response.status_code})"
            )
            return True
        else:
            logger.warning(
                f"⚠️  Grafana push failed — HTTP {response.status_code}: {response.text[:200]}"
            )
            return False

    except requests.exceptions.Timeout:
        logger.warning("⚠️  Grafana push timed out (10s). Metrics not pushed.")
        return False
    except requests.exceptions.ConnectionError as e:
        logger.warning(f"⚠️  Grafana push connection error: {e}")
        return False
    except Exception as e:
        logger.warning(f"⚠️  Grafana push unexpected error: {e}")
        return False


def push_from_result(result, run_id: Optional[str] = None) -> bool:
    """
    Convenience wrapper — push scores from an EvaluationResult object.

    Args:
        result: EvaluationResult instance
        run_id: Optional GitHub Actions run ID
    """
    scores = {
        "faithfulness": result.faithfulness,
        "context_relevance": result.context_relevance,
        "answer_correctness": result.answer_correctness,
        "token_efficiency": result.token_efficiency,
    }

    return push_evaluation_scores(
        scores=scores,
        pr_number=result.pr_number,
        run_id=run_id,
        overall_pass=result.overall_pass,
    )
