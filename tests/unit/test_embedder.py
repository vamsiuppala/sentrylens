"""
Unit tests for ErrorEmbedder.
"""
import pytest
import numpy as np

from sentrylens.embeddings.embedder import ErrorEmbedder
from sentrylens.core.models import AERIErrorRecord
from sentrylens.config import settings


@pytest.fixture
def sample_error():
    """Create a sample error record for testing."""
    return AERIErrorRecord(
        error_id="test-123",
        error_type="NullPointerException",
        error_message="Object reference not set to an instance of an object",
        stack_trace="""at com.example.service.UserService.getUser(UserService.java:42)
at com.example.controller.UserController.handleRequest(UserController.java:28)
at javax.servlet.http.HttpServlet.service(HttpServlet.java:750)"""
    )


@pytest.fixture
def embedder():
    """Create embedder instance."""
    return ErrorEmbedder(device="cpu", batch_size=8)


class TestErrorEmbedder:
    """Tests for ErrorEmbedder class."""
    
    def test_initialization(self, embedder):
        """Test embedder initializes correctly."""
        assert embedder.model_name == settings.EMBEDDING_MODEL
        assert embedder.embedding_dim == settings.EMBEDDING_DIMENSION
        assert embedder.device in ["cpu", "cuda"]
    
    def test_prepare_text(self, embedder, sample_error):
        """Test text preparation from error record."""
        text = embedder.prepare_text(sample_error)
        
        assert "Error Type: NullPointerException" in text
        assert "Message: Object reference not set" in text
        assert "Stack Trace:" in text
        assert "UserService.java:42" in text
    
    def test_embed_single(self, embedder, sample_error):
        """Test embedding a single error."""
        embedding = embedder.embed_single(sample_error)
        
        assert embedding.error_id == sample_error.error_id
        assert len(embedding.embedding) == settings.EMBEDDING_DIMENSION
        assert embedding.model_name == embedder.model_name
        assert all(isinstance(x, float) for x in embedding.embedding)
    
    def test_embed_batch(self, embedder):
        """Test batch embedding."""
        errors = [
            AERIErrorRecord(
                error_id=f"test-{i}",
                error_type="TestException",
                error_message=f"Test message {i}",
                stack_trace=f"at test.class.method(Test.java:{i})"
            )
            for i in range(5)
        ]
        
        embeddings = embedder.embed_batch(errors, show_progress=False)
        
        assert len(embeddings) == 5
        assert all(len(emb.embedding) == settings.EMBEDDING_DIMENSION for emb in embeddings)
        assert all(emb.model_name == embedder.model_name for emb in embeddings)
    
    def test_embed_empty_batch(self, embedder):
        """Test embedding empty list returns empty list."""
        embeddings = embedder.embed_batch([])
        assert embeddings == []
    
    def test_cache_key_generation(self, embedder, sample_error):
        """Test cache key is consistent."""
        key1 = embedder.get_cache_key(sample_error)
        key2 = embedder.get_cache_key(sample_error)
        
        assert key1 == key2
        assert len(key1) == 64  # SHA256 hex digest length