"""
rag_eval — Automated RAG Evaluation Pipeline

CI/CD-integrated quality gate for RAG systems using Ragas + Groq LLM judge.
"""

__version__ = "0.1.0"
__author__ = "Manik Bodamwad"

from rag_eval.evaluator import RagEvaluator, EvaluationResult
from rag_eval.reporter import Reporter

__all__ = ["RagEvaluator", "EvaluationResult", "Reporter", "__version__"]
