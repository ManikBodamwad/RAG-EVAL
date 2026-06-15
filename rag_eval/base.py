"""
rag_eval/base.py

Base classes for the RAG evaluation plugin system.

Users subclass BaseRAGPipeline to plug their own RAG pipeline into the evaluator.

Example:
    from rag_eval import BaseRAGPipeline, RAGResult

    class MyPipeline(BaseRAGPipeline):
        def init(self):
            self.db = load_my_vectorstore()
            self.llm = load_my_llm()

        def query(self, question: str) -> RAGResult:
            docs = self.db.search(question)
            answer = self.llm.generate(question, docs)
            return RAGResult(
                question=question,
                answer=answer,
                contexts=[d.text for d in docs],
            )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RAGResult:
    """Structured output from a single RAG pipeline query.

    Attributes:
        question: The input question.
        answer: The generated answer.
        contexts: List of retrieved context strings used to generate the answer.
        input_tokens: Number of prompt tokens consumed (for cost tracking).
        output_tokens: Number of completion tokens generated (for cost tracking).
    """
    question: str
    answer: str
    contexts: list[str]
    input_tokens: int = 0
    output_tokens: int = 0


class BaseRAGPipeline(ABC):
    """Abstract base class for RAG pipelines.

    Subclass this and implement ``init()`` and ``query()`` to plug your own
    RAG pipeline into rag-eval-gate.

    The evaluator will:
      1. Instantiate your class with no arguments
      2. Call ``init()`` once to let you build indexes / load models
      3. Call ``query(question)`` for each question in the test dataset
    """

    @abstractmethod
    def init(self) -> None:
        """Initialize the pipeline.

        This is where you should build your vector index, load models,
        establish connections, etc. Called once before any queries.
        """
        ...

    @abstractmethod
    def query(self, question: str) -> RAGResult:
        """Run a single question through the pipeline.

        Args:
            question: The user question to answer.

        Returns:
            A RAGResult with the answer, retrieved contexts, and token counts.
        """
        ...
