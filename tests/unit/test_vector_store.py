"""
Unit tests for FAISSVectorStore.
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile

from src.sentrylens.embeddings.vector_store import FAISSVectorStore
from src.sentrylens.core.models import AERIErrorRecord, ErrorEmbedding
from configs.settings import settings


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    embeddings = []
    for i in range(10):
        embedding = ErrorEmbedding(
            error_id=f"test-error-{i}",
            embedding=np.random.randn(settings.EMBEDDING_DIMENSION).tolist(),
            model_name="test-model"
        )
        embeddings.append(embedding)
    return embeddings


@pytest.fixture
def sample_errors():
    """Create sample error records."""
    errors = []
    for i in range(10):
        error = AERIErrorRecord(
            error_id=f"test-error-{i}",
            error_type=f"TestException{i}",
            error_message=f"Test message {i}",
            stack_trace=f"at test.class.method(Test.java:{i})"
        )
        errors.append(error)
    return errors


class TestFAISSVectorStore:
    """Tests for FAISSVectorStore class."""
    
    def test_initialization(self):
        """Test vector store initializes correctly."""
        store = FAISSVectorStore(index_type="IndexFlatIP")
        
        assert store.dimension == settings.EMBEDDING_DIMENSION
        assert store.index_type == "IndexFlatIP"
        assert store.index.ntotal == 0
        assert len(store.error_ids) == 0
    
    def test_add_embeddings(self, sample_embeddings, sample_errors):
        """Test adding embeddings to store."""
        store = FAISSVectorStore(index_type="IndexFlatIP")
        store.add_embeddings(sample_embeddings, sample_errors)
        
        assert store.index.ntotal == 10
        assert len(store.error_ids) == 10
        assert len(store.id_to_index) == 10
    
    def test_search(self, sample_embeddings):
        """Test similarity search."""
        store = FAISSVectorStore(index_type="IndexFlatIP")
        store.add_embeddings(sample_embeddings)
        
        # Search using first embedding
        query = sample_embeddings[0].embedding
        results = store.search(query, top_k=5)
        
        assert len(results) == 5
        assert results[0][0] == "test-error-0"  # Should find itself first
        assert all(isinstance(score, float) for _, score in results)
        assert all(-1.0 <= score <= 1.0 + 1e-6 for _, score in results)  # Cosine similarity range
    
    def test_search_by_error_id(self, sample_embeddings):
        """Test searching by error ID."""
        store = FAISSVectorStore(index_type="IndexFlatIP")
        store.add_embeddings(sample_embeddings)
        
        results = store.search_by_error_id("test-error-0", top_k=3, exclude_self=True)
        
        assert len(results) == 3
        assert all(error_id != "test-error-0" for error_id, _ in results)
    
    def test_save_and_load(self, sample_embeddings, sample_errors):
        """Test saving and loading vector store."""
        # Create and populate store
        store = FAISSVectorStore(index_type="IndexFlatIP")
        store.add_embeddings(sample_embeddings, sample_errors)
        
        # Save to temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_index"
            store.save(save_path)
            
            # Load from saved location
            loaded_store = FAISSVectorStore.load(save_path)
            
            # Verify loaded store matches original
            assert loaded_store.index.ntotal == store.index.ntotal
            assert loaded_store.error_ids == store.error_ids
            assert loaded_store.id_to_index == store.id_to_index
            assert loaded_store.dimension == store.dimension
    
    def test_get_stats(self, sample_embeddings):
        """Test statistics retrieval."""
        store = FAISSVectorStore(index_type="IndexFlatIP")
        store.add_embeddings(sample_embeddings)
        
        stats = store.get_stats()
        
        assert stats['total_vectors'] == 10
        assert stats['dimension'] == settings.EMBEDDING_DIMENSION
        assert stats['index_type'] == "IndexFlatIP"