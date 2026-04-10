"""
app/rag_pipeline.py — Mock RAG Application

A production-realistic Retrieval-Augmented Generation pipeline used as
the evaluation target. This is the system being quality-gated by rag-eval.

Architecture:
  - Embedding: sentence-transformers/all-MiniLM-L6-v2 (local, no API cost)
  - Vector Store: FAISS CPU (in-memory, built at startup)
  - Generator: Groq LLM via LiteLLM (configurable via RAG_MODEL env var)
  - Retriever: Top-3 similarity search with cosine distance
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Optional imports with helpful error messages
# ──────────────────────────────────────────────────────────────────────────────

try:
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError as e:
    raise ImportError(
        f"LangChain packages not installed: {e}\n"
        "Run: pip install langchain-community langchain-huggingface faiss-cpu"
    ) from e

try:
    import litellm
    litellm.set_verbose = False
except ImportError as e:
    raise ImportError(
        f"LiteLLM not installed: {e}\n"
        "Run: pip install litellm"
    ) from e


# ──────────────────────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RAGResult:
    """Structured output from a single RAG pipeline query."""
    question: str
    answer: str
    contexts: list[str]          # Retrieved document chunks (k items)
    input_tokens: int = 0        # Prompt token count (for cost tracking)
    output_tokens: int = 0       # Completion token count (for cost tracking)
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    model: str = ""
    sources: list[str] = field(default_factory=list)  # Source filenames


# ──────────────────────────────────────────────────────────────────────────────
# RAG Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    A mock RAG pipeline over an AI/ML concept corpus.

    Usage:
        pipeline = RAGPipeline()
        result = pipeline.query("What is retrieval-augmented generation?")
        print(result.answer)
        print(result.contexts)
    """

    DEFAULT_CORPUS_DIR = Path(__file__).parent / "corpus"
    DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_TOP_K = 3
    DEFAULT_CHUNK_SIZE = 800
    DEFAULT_CHUNK_OVERLAP = 100

    # LiteLLM model name — Groq's blazing LPU inference
    DEFAULT_MODEL = "groq/llama-3.3-70b-versatile"

    SYSTEM_PROMPT = """You are a precise, knowledgeable AI assistant specializing in machine learning and AI concepts.

Your task is to answer the user's question using ONLY the provided context passages.

Rules:
- Base your answer strictly on the provided context. Do NOT invent or assume facts not present.
- If the context does not contain enough information to answer, say: "The provided context does not contain sufficient information to answer this question."
- Be concise and direct. Avoid unnecessary padding or repetition.
- Do not mention "the context" or "the passages" in your answer — just answer naturally.
"""

    def __init__(
        self,
        corpus_dir: Optional[Path] = None,
        embedding_model: Optional[str] = None,
        rag_model: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
    ):
        self.corpus_dir = Path(corpus_dir) if corpus_dir else self.DEFAULT_CORPUS_DIR
        self.embedding_model_name = (
            embedding_model
            or os.environ.get("EMBEDDING_MODEL", self.DEFAULT_EMBEDDING_MODEL)
        )
        self.model = (
            rag_model
            or os.environ.get("RAG_MODEL", self.DEFAULT_MODEL)
        )
        self.top_k = top_k
        self._vectorstore: Optional[FAISS] = None
        self._embeddings: Optional[HuggingFaceEmbeddings] = None

    # ── Initialization ─────────────────────────────────────────────────────────

    def _load_embeddings(self) -> HuggingFaceEmbeddings:
        """Load the local sentence-transformer embedding model."""
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def _load_corpus(self) -> list[Document]:
        """Load all .txt files from the corpus directory as LangChain Documents."""
        if not self.corpus_dir.exists():
            raise FileNotFoundError(
                f"Corpus directory not found: {self.corpus_dir}\n"
                f"Expected at: {self.corpus_dir.resolve()}"
            )

        documents = []
        txt_files = sorted(self.corpus_dir.glob("*.txt"))
        if not txt_files:
            raise ValueError(f"No .txt files found in corpus directory: {self.corpus_dir}")

        logger.info(f"Loading {len(txt_files)} corpus documents from {self.corpus_dir}")

        for txt_file in txt_files:
            content = txt_file.read_text(encoding="utf-8").strip()
            if content:
                # Split into chunks by paragraph for better retrieval granularity
                chunks = self._chunk_text(content, txt_file.stem)
                documents.extend(chunks)

        logger.info(f"Created {len(documents)} document chunks from {len(txt_files)} files")
        return documents

    def _chunk_text(self, text: str, source: str) -> list[Document]:
        """
        Simple paragraph-aware chunking with overlap.
        Splits on double newlines (paragraphs) first, then by character limit.
        """
        # Try to split by paragraphs first
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) < self.DEFAULT_CHUNK_SIZE:
                current_chunk = f"{current_chunk}\n\n{para}".strip()
            else:
                if current_chunk:
                    chunks.append(Document(
                        page_content=current_chunk,
                        metadata={"source": source}
                    ))
                current_chunk = para

        if current_chunk:
            chunks.append(Document(
                page_content=current_chunk,
                metadata={"source": source}
            ))

        return chunks if chunks else [Document(page_content=text, metadata={"source": source})]

    def build_index(self) -> None:
        """Build the FAISS index from the corpus. Called lazily on first query."""
        logger.info("Building FAISS vector index...")
        self._embeddings = self._load_embeddings()
        documents = self._load_corpus()
        self._vectorstore = FAISS.from_documents(documents, self._embeddings)
        logger.info(f"FAISS index built with {len(documents)} chunks")

    def _ensure_index(self) -> None:
        """Lazily build the index on first use."""
        if self._vectorstore is None:
            self.build_index()

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(self, question: str) -> tuple[list[str], list[str], float]:
        """
        Retrieve the top-k most relevant document chunks for a question.

        Returns:
            (contexts, sources, retrieval_time_ms)
        """
        self._ensure_index()

        t0 = time.perf_counter()
        docs = self._vectorstore.similarity_search(question, k=self.top_k)
        retrieval_time_ms = (time.perf_counter() - t0) * 1000

        contexts = [doc.page_content for doc in docs]
        sources = [doc.metadata.get("source", "unknown") for doc in docs]

        return contexts, sources, retrieval_time_ms

    # ── Generation ─────────────────────────────────────────────────────────────

    def generate(self, question: str, contexts: list[str]) -> tuple[str, int, int, float]:
        """
        Generate an answer using Groq LLM via LiteLLM, grounded in retrieved contexts.

        Returns:
            (answer, input_tokens, output_tokens, generation_time_ms)
        """
        context_block = "\n\n---\n\n".join(
            f"[Context {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)
        )

        user_message = (
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        t0 = time.perf_counter()
        response = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,  # Low temperature for factual, deterministic responses
            max_tokens=512,
        )
        generation_time_ms = (time.perf_counter() - t0) * 1000

        answer = response.choices[0].message.content.strip()
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        return answer, input_tokens, output_tokens, generation_time_ms

    # ── Full Pipeline ──────────────────────────────────────────────────────────

    def query(self, question: str) -> RAGResult:
        """
        Run the full RAG pipeline: retrieve → generate → return structured result.

        Args:
            question: The user's question to answer.

        Returns:
            RAGResult with answer, contexts, token counts, and timing metrics.
        """
        logger.debug(f"Processing query: {question[:80]}...")

        # 1. Retrieve relevant context
        contexts, sources, retrieval_time_ms = self.retrieve(question)

        # 2. Generate answer grounded in context
        answer, input_tokens, output_tokens, generation_time_ms = self.generate(
            question, contexts
        )

        return RAGResult(
            question=question,
            answer=answer,
            contexts=contexts,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms,
            model=self.model,
            sources=sources,
        )

    def batch_query(self, questions: list[str]) -> list[RAGResult]:
        """
        Run the RAG pipeline on multiple questions sequentially.
        Reuses the same FAISS index across all queries.
        """
        self._ensure_index()
        results = []
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}: {question[:60]}...")
            try:
                result = self.query(question)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process question {i+1}: {e}")
                # Return a failed result rather than crashing the whole batch
                results.append(RAGResult(
                    question=question,
                    answer=f"ERROR: {str(e)}",
                    contexts=[],
                    sources=[],
                ))
        return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI Test Runner (python app/rag_pipeline.py)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("🔧 Initializing RAG Pipeline...")
    pipeline = RAGPipeline()

    test_question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What is retrieval-augmented generation and how does it reduce hallucinations?"
    )

    print(f"\n❓ Question: {test_question}\n")
    result = pipeline.query(test_question)

    print(f"✅ Answer:\n{result.answer}\n")
    print(f"📄 Sources: {result.sources}")
    print(f"🔢 Tokens: {result.input_tokens} in / {result.output_tokens} out")
    print(f"⏱️  Retrieval: {result.retrieval_time_ms:.1f}ms | Generation: {result.generation_time_ms:.1f}ms")
