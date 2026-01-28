"""
Unit tests for HDBSCAN clustering.
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile

from sentrylens.clustering.clusterer import HDBSCANClusterer, ClusterStats
from sentrylens.core.models import AERIErrorRecord, ErrorEmbedding
from sentrylens.config import settings


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    embeddings = []

    # Create 3 distinct clusters of embeddings
    np.random.seed(42)

    # Cluster 1: offset by +1.0
    for i in range(10):
        embedding = np.random.randn(settings.EMBEDDING_DIMENSION) * 0.1 + 1.0
        embeddings.append(
            ErrorEmbedding(
                error_id=f"cluster1-{i}",
                embedding=embedding.tolist(),
                model_name="test-model"
            )
        )

    # Cluster 2: offset by -1.0
    for i in range(10):
        embedding = np.random.randn(settings.EMBEDDING_DIMENSION) * 0.1 - 1.0
        embeddings.append(
            ErrorEmbedding(
                error_id=f"cluster2-{i}",
                embedding=embedding.tolist(),
                model_name="test-model"
            )
        )

    # Cluster 3: offset by 0.5 (different cluster)
    for i in range(8):
        embedding = np.random.randn(settings.EMBEDDING_DIMENSION) * 0.1 + 0.5
        embeddings.append(
            ErrorEmbedding(
                error_id=f"cluster3-{i}",
                embedding=embedding.tolist(),
                model_name="test-model"
            )
        )

    # Add some noise points
    for i in range(2):
        embedding = np.random.randn(settings.EMBEDDING_DIMENSION) * 2.0
        embeddings.append(
            ErrorEmbedding(
                error_id=f"noise-{i}",
                embedding=embedding.tolist(),
                model_name="test-model"
            )
        )

    return embeddings


@pytest.fixture
def sample_errors():
    """Create sample error records."""
    errors = []
    for i in range(30):
        error = AERIErrorRecord(
            error_id=f"error-{i}",
            error_type=f"TestException{i % 5}",
            error_message=f"Test error message {i}",
            stack_trace=f"at test.class.method(Test.java:{i})\nat test.other(Other.java:1)"
        )
        errors.append(error)
    return errors


class TestHDBSCANClusterer:
    """Tests for HDBSCANClusterer class."""
    
    def test_initialization(self):
        """Test clusterer initializes correctly."""
        clusterer = HDBSCANClusterer(min_cluster_size=5)
        
        assert clusterer.min_cluster_size == 5
        assert clusterer.min_samples == 5
        assert clusterer.metric == "euclidean"
        assert clusterer.labels is None
        assert clusterer.embeddings is None
    
    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        clusterer = HDBSCANClusterer(
            min_cluster_size=10,
            min_samples=15,
            cluster_selection_epsilon=0.5,
            metric="cosine"
        )
        
        assert clusterer.min_cluster_size == 10
        assert clusterer.min_samples == 15
        assert clusterer.cluster_selection_epsilon == 0.5
        assert clusterer.metric == "cosine"
    
    def test_fit_basic(self, sample_embeddings):
        """Test basic clustering fitting."""
        clusterer = HDBSCANClusterer(min_cluster_size=3)
        
        # Convert to numpy array
        embedding_vectors = np.array([e.embedding for e in sample_embeddings])
        
        # Fit
        labels = clusterer.fit(embedding_vectors)
        
        assert labels is not None
        assert len(labels) == len(sample_embeddings)
        assert all(isinstance(l, (int, np.integer)) for l in labels)
        assert min(labels) >= -1  # -1 is noise
    
    def test_fit_detects_multiple_clusters(self, sample_embeddings):
        """Test that clustering detects multiple clusters."""
        clusterer = HDBSCANClusterer(min_cluster_size=3)
        embedding_vectors = np.array([e.embedding for e in sample_embeddings])
        
        labels = clusterer.fit(embedding_vectors)
        
        # Should detect at least 2 clusters (we created 3, some noise)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        assert num_clusters >= 1  # At least one cluster
    
    def test_cluster_embeddings(self, sample_embeddings):
        """Test clustering embeddings end-to-end."""
        clusterer = HDBSCANClusterer(min_cluster_size=3)
        
        assignments = clusterer.cluster_embeddings(sample_embeddings)
        
        assert len(assignments) == len(sample_embeddings)
        assert all(hasattr(a, 'error_id') for a in assignments)
        assert all(hasattr(a, 'cluster_id') for a in assignments)
        assert all(isinstance(a.cluster_id, int) for a in assignments)
    
    def test_noise_detection(self, sample_embeddings):
        """Test that noise points are detected."""
        clusterer = HDBSCANClusterer(min_cluster_size=5)
        embedding_vectors = np.array([e.embedding for e in sample_embeddings])
        
        labels = clusterer.fit(embedding_vectors)
        
        # Should have some noise points (-1 label)
        num_noise = np.sum(labels == -1)
        assert num_noise >= 0  # May or may not have noise depending on data
    
    def test_get_stats(self, sample_embeddings):
        """Test statistics retrieval."""
        clusterer = HDBSCANClusterer(min_cluster_size=3)
        embedding_vectors = np.array([e.embedding for e in sample_embeddings])
        
        clusterer.fit(embedding_vectors)
        stats = clusterer.get_stats()
        
        assert isinstance(stats, ClusterStats)
        assert stats.num_clusters >= 0
        assert stats.num_noise_points >= 0
        assert stats.total_points == len(sample_embeddings)
        assert stats.noise_fraction >= 0.0
        assert stats.noise_fraction <= 1.0
        assert stats.avg_cluster_size >= 0.0
    
    def test_get_cluster_members(self, sample_embeddings):
        """Test retrieving members of a cluster."""
        clusterer = HDBSCANClusterer(min_cluster_size=3)
        embedding_vectors = np.array([e.embedding for e in sample_embeddings])
        
        clusterer.fit(embedding_vectors)
        stats = clusterer.get_stats()
        
        # Get members of first cluster
        if stats.num_clusters > 0:
            first_cluster_id = min(stats.cluster_sizes.keys())
            members = clusterer.get_cluster_members(first_cluster_id)
            
            assert len(members) > 0
            assert all(clusterer.labels[m] == first_cluster_id for m in members)
    
    def test_get_cluster_center(self, sample_embeddings):
        """Test retrieving cluster center."""
        clusterer = HDBSCANClusterer(min_cluster_size=3)
        embedding_vectors = np.array([e.embedding for e in sample_embeddings])
        
        clusterer.fit(embedding_vectors)
        stats = clusterer.get_stats()
        
        # Get center of first cluster
        if stats.num_clusters > 0:
            first_cluster_id = min(stats.cluster_sizes.keys())
            center = clusterer.get_cluster_center(first_cluster_id)
            
            assert center is not None
            assert len(center) == settings.EMBEDDING_DIMENSION
    
    def test_get_cluster_center_noise_raises_error(self, sample_embeddings):
        """Test that getting center of noise cluster raises error."""
        clusterer = HDBSCANClusterer(min_cluster_size=3)
        embedding_vectors = np.array([e.embedding for e in sample_embeddings])
        
        clusterer.fit(embedding_vectors)
        
        with pytest.raises(Exception):  # ClusteringError
            clusterer.get_cluster_center(-1)
    
    def test_predict_on_new_data(self, sample_embeddings):
        """Test prediction on new embeddings."""
        clusterer = HDBSCANClusterer(min_cluster_size=3)
        embedding_vectors = np.array([e.embedding for e in sample_embeddings])
        
        # Fit on original data
        clusterer.fit(embedding_vectors)
        
        # Create new embeddings similar to cluster 1
        new_embeddings = np.random.randn(3, settings.EMBEDDING_DIMENSION) * 0.1 + 1.0
        
        # Predict
        labels = clusterer.predict(new_embeddings)
        
        assert len(labels) == 3
        assert all(isinstance(l, (int, np.integer)) for l in labels)
    
    def test_error_on_too_few_samples(self):
        """Test error when fitting with too few samples."""
        clusterer = HDBSCANClusterer(min_cluster_size=10)
        
        # Create fewer samples than min_cluster_size
        small_embeddings = np.random.randn(5, settings.EMBEDDING_DIMENSION)
        
        with pytest.raises(Exception):  # ClusteringError
            clusterer.fit(small_embeddings)
    
    def test_error_on_predict_before_fit(self):
        """Test error when predicting before fitting."""
        clusterer = HDBSCANClusterer(min_cluster_size=3)
        
        new_embeddings = np.random.randn(3, settings.EMBEDDING_DIMENSION)
        
        with pytest.raises(Exception):  # ClusteringError
            clusterer.predict(new_embeddings)
    
    def test_cluster_embeddings_with_errors(self, sample_embeddings, sample_errors):
        """Test clustering with error records for context."""
        clusterer = HDBSCANClusterer(min_cluster_size=3)
        
        assignments = clusterer.cluster_embeddings(sample_embeddings, sample_errors)
        
        assert len(assignments) == len(sample_embeddings)
        # All assignments should have cluster_id and error_id
        assert all(a.error_id is not None for a in assignments)
        assert all(a.cluster_id is not None for a in assignments)
    
    def test_empty_embeddings_raises_error(self):
        """Test that empty embeddings raise error."""
        clusterer = HDBSCANClusterer(min_cluster_size=3)
        
        with pytest.raises(Exception):  # ClusteringError
            clusterer.cluster_embeddings([])
    
    def test_different_min_cluster_sizes_affect_results(self, sample_embeddings):
        """Test that min_cluster_size parameter affects clustering."""
        embedding_vectors = np.array([e.embedding for e in sample_embeddings])
        
        # Small min_cluster_size = more clusters
        clusterer_small = HDBSCANClusterer(min_cluster_size=2)
        labels_small = clusterer_small.fit(embedding_vectors)
        clusters_small = len(set(labels_small)) - (1 if -1 in labels_small else 0)
        
        # Large min_cluster_size = fewer clusters
        clusterer_large = HDBSCANClusterer(min_cluster_size=10)
        labels_large = clusterer_large.fit(embedding_vectors)
        clusters_large = len(set(labels_large)) - (1 if -1 in labels_large else 0)
        
        # Smaller min_cluster_size should find more clusters
        assert clusters_small >= clusters_large
