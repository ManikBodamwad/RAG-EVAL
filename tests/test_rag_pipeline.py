import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from app.rag_pipeline import RAGPipeline, RAGResult


@pytest.fixture
def mock_pipeline_dependencies():
    """Mock the heavy models and vector database calls."""
    with patch("app.rag_pipeline.HuggingFaceEmbeddings") as mock_embeddings_cls, \
         patch("app.rag_pipeline.FAISS") as mock_faiss_cls, \
         patch("litellm.completion") as mock_completion:
        
        # Setup mock embeddings
        mock_embeddings = MagicMock()
        mock_embeddings_cls.return_value = mock_embeddings
        
        # Setup mock vectorstore
        mock_vectorstore = MagicMock()
        mock_faiss_cls.from_documents.return_value = mock_vectorstore
        
        yield {
            "embeddings_cls": mock_embeddings_cls,
            "embeddings": mock_embeddings,
            "faiss_cls": mock_faiss_cls,
            "vectorstore": mock_vectorstore,
            "completion": mock_completion,
        }


def test_chunk_text_paragraphs():
    pipeline = RAGPipeline(top_k=2)
    text = "Paragraph 1 is short.\n\nParagraph 2 is also short and simple."
    chunks = pipeline._chunk_text(text, "source_doc")
    
    assert len(chunks) == 1
    assert chunks[0].page_content == "Paragraph 1 is short.\n\nParagraph 2 is also short and simple."
    assert chunks[0].metadata["source"] == "source_doc"


def test_chunk_text_long_paragraphs():
    pipeline = RAGPipeline(top_k=2)
    # Force splitting by creating paragraphs that exceed default chunk size when combined
    p1 = "a" * 500
    p2 = "b" * 400
    text = f"{p1}\n\n{p2}"
    chunks = pipeline._chunk_text(text, "source_doc")
    
    assert len(chunks) == 2
    assert chunks[0].page_content == p1
    assert chunks[1].page_content == p2


@patch("app.rag_pipeline.RAGPipeline._load_corpus")
def test_build_index(mock_load, mock_pipeline_dependencies):
    mock_load.return_value = [Document(page_content="Content", metadata={"source": "doc"})]
    
    pipeline = RAGPipeline()
    pipeline.build_index()
    
    mock_pipeline_dependencies["embeddings_cls"].assert_called_once()
    mock_pipeline_dependencies["faiss_cls"].from_documents.assert_called_once()
    assert pipeline._vectorstore is not None


def test_retrieve(mock_pipeline_dependencies):
    pipeline = RAGPipeline(top_k=2)
    pipeline._vectorstore = mock_pipeline_dependencies["vectorstore"]
    
    mock_docs = [
        Document(page_content="Ctx 1", metadata={"source": "src1"}),
        Document(page_content="Ctx 2", metadata={"source": "src2"}),
    ]
    mock_pipeline_dependencies["vectorstore"].similarity_search.return_value = mock_docs
    
    contexts, sources, time_ms = pipeline.retrieve("What is AI?")
    
    assert contexts == ["Ctx 1", "Ctx 2"]
    assert sources == ["src1", "src2"]
    assert time_ms >= 0.0
    mock_pipeline_dependencies["vectorstore"].similarity_search.assert_called_once_with("What is AI?", k=2)


def test_generate(mock_pipeline_dependencies):
    pipeline = RAGPipeline()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Hello answer"))]
    mock_response.usage = MagicMock(prompt_tokens=15, completion_tokens=20)
    mock_pipeline_dependencies["completion"].return_value = mock_response
    
    ans, in_tok, out_tok, time_ms = pipeline.generate("Q?", ["C1", "C2"])
    
    assert ans == "Hello answer"
    assert in_tok == 15
    assert out_tok == 20
    assert time_ms >= 0.0
    mock_pipeline_dependencies["completion"].assert_called_once()


@patch("app.rag_pipeline.RAGPipeline.retrieve")
@patch("app.rag_pipeline.RAGPipeline.generate")
def test_query_flow(mock_gen, mock_ret):
    mock_ret.return_value = (["Ctx"], ["Src"], 5.0)
    mock_gen.return_value = ("Answer", 10, 20, 200.0)
    
    pipeline = RAGPipeline()
    res = pipeline.query("Q")
    
    assert isinstance(res, RAGResult)
    assert res.question == "Q"
    assert res.answer == "Answer"
    assert res.contexts == ["Ctx"]
    assert res.input_tokens == 10
    assert res.output_tokens == 20
    assert res.retrieval_time_ms == 5.0
    assert res.generation_time_ms == 200.0


@patch("app.rag_pipeline.RAGPipeline._ensure_index")
@patch("app.rag_pipeline.RAGPipeline.query")
def test_batch_query(mock_query, mock_ensure):
    pipeline = RAGPipeline()
    
    # Return valid result for Q1, and fail for Q2
    res_success = RAGResult(question="Q1", answer="A1", contexts=["C1"])
    mock_query.side_effect = [res_success, Exception("Generation failed")]
    
    results = pipeline.batch_query(["Q1", "Q2"])
    
    assert len(results) == 2
    assert results[0].answer == "A1"
    assert results[1].answer == "ERROR: Generation failed"
    assert results[1].contexts == []
    mock_ensure.assert_called_once()
