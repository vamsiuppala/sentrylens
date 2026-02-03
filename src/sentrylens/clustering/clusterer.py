"""
HDBSCAN clustering implementation for error grouping.
Provides density-based clustering with automatic optimal epsilon detection.
"""
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import hdbscan

from anthropic import Anthropic
from dotenv import load_dotenv

from sentrylens.core.models import (
    AERIErrorRecord,
    ErrorEmbedding,
    ClusterAssignment,
)
from sentrylens.core.exceptions import ClusteringError
from sentrylens.utils.logger import logger
from sentrylens.config import settings


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
    cluster_labels: Dict[int, str] = field(default_factory=dict)


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
        errors: Optional[List[AERIErrorRecord]] = None,
        generate_labels: bool = False,
        label_progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Tuple[List[ClusterAssignment], ClusterStats]:
        """
        Cluster error embeddings and return assignments with statistics.

        Args:
            embeddings: List of ErrorEmbedding objects
            errors: List of corresponding AERIErrorRecord objects (required if generate_labels=True)
            generate_labels: Whether to generate human-readable cluster labels using Claude API
            label_progress_callback: Optional callback(cluster_id, label) for label generation progress

        Returns:
            Tuple of (List of ClusterAssignment objects, ClusterStats with optional labels)
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

        # Generate labels if requested
        if generate_labels and errors:
            logger.info("Generating cluster labels...")
            stats.cluster_labels = self.generate_labels(
                errors=errors,
                assignments=assignments,
                progress_callback=label_progress_callback,
            )

        logger.info(
            "Created cluster assignments",
            num_assignments=len(assignments),
            num_clusters=stats.num_clusters,
            num_noise=stats.num_noise_points,
            num_labels=len(stats.cluster_labels),
        )

        return assignments, stats
    
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

    def generate_labels(
        self,
        errors: List[AERIErrorRecord],
        assignments: List[ClusterAssignment],
        model: str = "claude-3-5-haiku-20241022",
        max_sample: int = 5,
        max_workers: int = 10,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[int, str]:
        """
        Generate human-readable labels for all clusters using Claude API.

        Uses parallel API calls for faster processing of many clusters.

        Args:
            errors: List of all error records
            assignments: List of cluster assignments
            model: Claude model to use for label generation
            max_sample: Maximum errors to sample per cluster for labeling
            max_workers: Maximum parallel API calls (default: 10)
            progress_callback: Optional callback(cluster_id, label) for progress updates

        Returns:
            Dict mapping cluster_id to human-readable label
        """
        # Group errors by cluster
        error_by_id = {e.error_id: e for e in errors}
        clusters: Dict[int, List[AERIErrorRecord]] = {}

        for assignment in assignments:
            cid = assignment.cluster_id
            if cid == -1:  # Skip noise
                continue
            if assignment.error_id in error_by_id:
                if cid not in clusters:
                    clusters[cid] = []
                clusters[cid].append(error_by_id[assignment.error_id])

        # Generate labels
        labels: Dict[int, str] = {}

        load_dotenv()
        api_key = os.environ.get("ANTHROPIC_API_KEY")

        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set, using fallback labels")
            for cid, cluster_errors in clusters.items():
                label = self._fallback_label(cluster_errors)
                labels[cid] = label
                if progress_callback:
                    progress_callback(cid, label)
            return labels

        client = Anthropic(api_key=api_key)

        # Parallel label generation
        logger.info(f"Generating labels for {len(clusters)} clusters with {max_workers} workers")

        def generate_for_cluster(cid: int, cluster_errors: List[AERIErrorRecord]) -> Tuple[int, str]:
            label = self._generate_single_label(cluster_errors, client, model, max_sample)
            return cid, label

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(generate_for_cluster, cid, cluster_errors): cid
                for cid, cluster_errors in clusters.items()
            }

            for future in as_completed(futures):
                try:
                    cid, label = future.result()
                    labels[cid] = label
                    if progress_callback:
                        progress_callback(cid, label)
                except Exception as e:
                    cid = futures[future]
                    logger.warning(f"Label generation failed for cluster {cid}", error=str(e))
                    labels[cid] = self._fallback_label(clusters[cid])
                    if progress_callback:
                        progress_callback(cid, labels[cid])

        logger.info("Generated cluster labels", num_labels=len(labels))
        return labels

    def _generate_single_label(
        self,
        errors: List[AERIErrorRecord],
        client: Anthropic,
        model: str,
        max_sample: int,
    ) -> str:
        """Generate a label for a single cluster using Claude API."""
        if not errors:
            return "Empty cluster"

        # Sample errors
        sample = errors[:max_sample]

        # Build context
        error_descriptions = []
        for i, error in enumerate(sample, 1):
            desc = f"{i}. Type: {error.error_type}\n   Message: {error.error_message[:200]}"
            error_descriptions.append(desc)

        errors_text = "\n".join(error_descriptions)

        prompt = f"""Analyze these related errors from a software error cluster and generate a short, descriptive label.

Errors in cluster:
{errors_text}

Requirements for the label:
- Maximum 6 words
- Describe the common issue/pattern
- Use technical but clear language
- Examples: "Database connection timeout", "Null pointer in user service", "File permission denied errors"

Respond with ONLY the label, nothing else."""

        try:
            response = client.messages.create(
                model=model,
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}]
            )

            label = response.content[0].text.strip().strip('"\'')
            logger.debug("Generated cluster label", label=label, num_errors=len(errors))
            return label

        except Exception as e:
            logger.warning("Label generation failed, using fallback", error=str(e))
            return self._fallback_label(errors)

    def _fallback_label(self, errors: List[AERIErrorRecord]) -> str:
        """Generate a simple fallback label from the most common error type."""
        if not errors:
            return "Empty cluster"

        # Count error types
        type_counts: Dict[str, int] = {}
        for error in errors:
            error_type = error.error_type or "Unknown"
            # Simplify Java exception names (e.g., java.lang.NullPointerException -> NullPointerException)
            if "." in error_type:
                error_type = error_type.split(".")[-1]
            type_counts[error_type] = type_counts.get(error_type, 0) + 1

        # Return most common type
        most_common = max(type_counts, key=type_counts.get)
        return most_common
