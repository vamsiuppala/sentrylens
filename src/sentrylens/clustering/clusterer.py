"""
HDBSCAN clustering implementation for error grouping.
Provides density-based clustering with automatic optimal epsilon detection.
"""
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import hdbscan

from src.sentrylens.core.models import (
    AERIErrorRecord,
    ErrorEmbedding,
    ClusterAssignment,
)
from src.sentrylens.core.exceptions import ClusteringError
from src.sentrylens.utils.logger import logger
from configs.settings import settings


@dataclass
class ClusterStats:
    """Statistics about clustering results."""
    
    num_clusters: int
    num_noise_points: int
    total_points: int
    cluster_sizes: Dict[int, int]
    avg_cluster_size: float
    largest_cluster_size: int
    smallest_cluster_size: int
    noise_fraction: float
    silhouette_score: Optional[float] = None


class HDBSCANClusterer:
    """
    Density-based clustering using HDBSCAN.
    
    HDBSCAN is a clustering algorithm that:
    1. Identifies density-based clusters (not spherical like K-means)
    2. Automatically detects number of clusters (no need to specify K)
    3. Marks low-density points as noise (-1 label)
    4. Is robust to outliers
    
    Why HDBSCAN for error clustering?
    - Error types have different shapes/densities in embedding space
    - Some errors are rare outliers (noise points)
    - We don't know optimal number of clusters a priori
    - More interpretable than hard clustering methods
    """
    
    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        cluster_selection_epsilon: float = 0.0,
        algorithm: str = "best",
        metric: str = "euclidean"
    ):
        """
        Initialize HDBSCAN clusterer.
        
        Args:
            min_cluster_size: Minimum number of samples in a cluster.
                Smaller values = more clusters, more noise points.
                Typical range: 3-20. Default 5 is conservative.
            
            min_samples: Number of samples in neighborhood for density.
                If None, defaults to min_cluster_size.
                Lower = more clusters detected.
            
            cluster_selection_epsilon: Distance threshold for cluster selection.
                0.0 = use default (most stable clusters).
                Higher = fewer, larger clusters.
            
            algorithm: Which algorithm to use ("best", "generic", "prims_kdtree", 
                or "prims_balltree"). "best" auto-selects.
            
            metric: Distance metric ("euclidean", "cosine", "manhattan", etc.)
                For embeddings, euclidean is standard.
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples or min_cluster_size
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.algorithm = algorithm
        self.metric = metric
        
        self.clusterer: Optional[hdbscan.HDBSCAN] = None
        self.labels: Optional[np.ndarray] = None
        self.embeddings: Optional[np.ndarray] = None
        
        logger.info(
            "Initialized HDBSCANClusterer",
            min_cluster_size=min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            algorithm=algorithm,
            metric=metric
        )
    
    def fit(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit HDBSCAN on embeddings and return cluster labels.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            
        Returns:
            Cluster labels (1D array, -1 for noise points)
            
        Raises:
            ClusteringError: If clustering fails
        """
        if embeddings.shape[0] < self.min_cluster_size:
            raise ClusteringError(
                f"Number of samples ({embeddings.shape[0]}) is less than "
                f"min_cluster_size ({self.min_cluster_size})"
            )
        
        logger.info(
            "Starting HDBSCAN clustering",
            num_samples=embeddings.shape[0],
            num_features=embeddings.shape[1]
        )
        
        try:
            # Create and fit clusterer
            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                cluster_selection_epsilon=self.cluster_selection_epsilon,
                algorithm=self.algorithm,
                metric=self.metric,
                prediction_data=True  # Enable prediction on new data
            )
            
            # Fit and get labels
            self.labels = self.clusterer.fit_predict(embeddings)
            self.embeddings = embeddings
            
            # Log results
            num_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
            num_noise = np.sum(self.labels == -1)
            
            logger.info(
                "HDBSCAN clustering complete",
                num_clusters=num_clusters,
                num_noise_points=num_noise,
                total_points=len(self.labels)
            )
            
            return self.labels
        
        except Exception as e:
            raise ClusteringError(f"HDBSCAN fitting failed: {e}")
    
    def cluster_embeddings(
        self,
        embeddings: List[ErrorEmbedding],
        errors: Optional[List[AERIErrorRecord]] = None
    ) -> List[ClusterAssignment]:
        """
        Cluster error embeddings and return assignments.
        
        Args:
            embeddings: List of ErrorEmbedding objects
            errors: List of corresponding AERIErrorRecord objects (for size info)
            
        Returns:
            List of ClusterAssignment objects
        """
        if not embeddings:
            raise ClusteringError("No embeddings provided")
        
        # Convert to numpy array
        embedding_vectors = np.array([e.embedding for e in embeddings])
        
        # Fit clustering
        labels = self.fit(embedding_vectors)
        
        # Create assignments
        assignments = []
        
        # Get cluster statistics
        stats = self.get_stats()
        
        for embedding, label in zip(embeddings, labels):
            cluster_id = int(label)
            cluster_size = stats.cluster_sizes.get(cluster_id, 0) if cluster_id != -1 else None
            
            # Compute distance to cluster center if not noise
            distance_to_centroid = None
            if cluster_id != -1 and self.clusterer is not None:
                try:
                    # Get cluster centroids
                    mask = self.labels == cluster_id
                    centroid = self.embeddings[mask].mean(axis=0)
                    distance_to_centroid = float(
                        np.linalg.norm(embedding_vectors[labels == cluster_id].mean(axis=0) - 
                                     np.array(embedding.embedding))
                    )
                except Exception:
                    distance_to_centroid = None
            
            assignment = ClusterAssignment(
                error_id=embedding.error_id,
                cluster_id=cluster_id,
                distance_to_centroid=distance_to_centroid,
                cluster_size=cluster_size
            )
            assignments.append(assignment)
        
        logger.info(
            "Created cluster assignments",
            num_assignments=len(assignments),
            num_clusters=stats.num_clusters,
            num_noise=stats.num_noise_points
        )
        
        return assignments
    
    def get_stats(self) -> ClusterStats:
        """
        Get clustering statistics.
        
        Returns:
            ClusterStats object with detailed information
            
        Raises:
            ClusteringError: If clustering hasn't been run yet
        """
        if self.labels is None:
            raise ClusteringError("Clustering hasn't been fit yet")
        
        # Count clusters (excluding noise label -1)
        unique_labels = set(self.labels)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        # Count noise points
        num_noise = np.sum(self.labels == -1)
        total_points = len(self.labels)
        
        # Get cluster sizes
        cluster_sizes = {}
        for label in unique_labels:
            if label != -1:  # Skip noise
                cluster_sizes[int(label)] = int(np.sum(self.labels == label))
        
        # Compute statistics
        avg_cluster_size = np.mean(list(cluster_sizes.values())) if cluster_sizes else 0.0
        largest_cluster = max(cluster_sizes.values()) if cluster_sizes else 0
        smallest_cluster = min(cluster_sizes.values()) if cluster_sizes else 0
        noise_fraction = num_noise / total_points if total_points > 0 else 0.0
        
        return ClusterStats(
            num_clusters=num_clusters,
            num_noise_points=num_noise,
            total_points=total_points,
            cluster_sizes=cluster_sizes,
            avg_cluster_size=avg_cluster_size,
            largest_cluster_size=largest_cluster,
            smallest_cluster_size=smallest_cluster,
            noise_fraction=noise_fraction
        )
    
    def get_cluster_members(self, cluster_id: int) -> np.ndarray:
        """
        Get indices of points in a cluster.
        
        Args:
            cluster_id: Cluster ID (-1 for noise)
            
        Returns:
            Array of indices
        """
        if self.labels is None:
            raise ClusteringError("Clustering hasn't been fit yet")
        
        return np.where(self.labels == cluster_id)[0]
    
    def get_cluster_center(self, cluster_id: int) -> np.ndarray:
        """
        Get the center (mean) of a cluster.
        
        Args:
            cluster_id: Cluster ID (cannot be -1 for noise)
            
        Returns:
            Cluster center as array
            
        Raises:
            ClusteringError: If cluster doesn't exist or is noise
        """
        if cluster_id == -1:
            raise ClusteringError("Cannot get center of noise cluster (-1)")
        
        if self.embeddings is None or self.labels is None:
            raise ClusteringError("Clustering hasn't been fit yet")
        
        members = self.get_cluster_members(cluster_id)
        if len(members) == 0:
            raise ClusteringError(f"Cluster {cluster_id} has no members")
        
        return self.embeddings[members].mean(axis=0)
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new embeddings using approximate prediction.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            
        Returns:
            Predicted cluster labels
            
        Raises:
            ClusteringError: If clusterer hasn't been fit or prediction fails
        """
        if self.clusterer is None:
            raise ClusteringError("Clusterer hasn't been fit yet")
        
        try:
            labels, strengths = hdbscan.approximate_predict(
                self.clusterer,
                embeddings
            )
            return labels
        except Exception as e:
            raise ClusteringError(f"Prediction failed: {e}")
