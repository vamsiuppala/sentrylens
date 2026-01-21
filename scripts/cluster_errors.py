#!/usr/bin/env python3
"""
Run HDBSCAN clustering on error embeddings and save cluster assignments.

Usage:
    python scripts/cluster_errors.py --input data/embeddings/embeddings_*.json
    python scripts/cluster_errors.py --input data/embeddings/embeddings_*.json --min-cluster-size 10
    python scripts/cluster_errors.py --input data/embeddings/embeddings_*.json --min-cluster-size 5 --output data/clusters/clusters_*.json
"""
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np

from src.sentrylens.data.loader import AERIDataLoader
from src.sentrylens.clustering.clusterer import HDBSCANClusterer
from src.sentrylens.core.models import ProcessedDataset
from src.sentrylens.core.exceptions import ClusteringError
from src.sentrylens.utils.logger import logger
from configs.settings import settings


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run HDBSCAN clustering on error embeddings"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to embeddings JSON file (or pattern like embeddings_*.json)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for clusters (default: auto-generated)"
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="Minimum cluster size (default: 5). Smaller = more clusters."
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        help="Minimum samples in neighborhood (default: same as min-cluster-size)"
    )
    parser.add_argument(
        "--cluster-selection-epsilon",
        type=float,
        default=0.0,
        help="Distance threshold for cluster selection (default: 0.0)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "cosine", "manhattan"],
        help="Distance metric (default: euclidean)"
    )
    
    return parser.parse_args()


def main():
    """Main clustering pipeline."""
    args = parse_args()
    
    logger.info(
        "Starting clustering",
        input_file=str(args.input),
        min_cluster_size=args.min_cluster_size,
        metric=args.metric
    )
    
    try:
        # Load embeddings dataset
        logger.info("Loading embeddings dataset...")
        loader = AERIDataLoader()
        dataset = loader.load_processed_dataset(args.input)
        
        if not dataset.embeddings:
            logger.error("No embeddings in dataset. Run generate_embeddings.py first.")
            return 1
        
        logger.info(f"Loaded {len(dataset.embeddings)} embeddings")
        
        # Initialize clusterer
        logger.info("Initializing HDBSCAN clusterer...")
        clusterer = HDBSCANClusterer(
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            cluster_selection_epsilon=args.cluster_selection_epsilon,
            metric=args.metric
        )
        
        # Perform clustering
        logger.info("Running clustering...")
        cluster_assignments = clusterer.cluster_embeddings(
            dataset.embeddings,
            dataset.errors
        )
        
        # Update dataset with clusters
        dataset.clusters = cluster_assignments
        
        # Get statistics
        stats = clusterer.get_stats()
        logger.info(
            "Clustering complete",
            num_clusters=stats.num_clusters,
            num_noise_points=stats.num_noise_points,
            total_points=stats.total_points,
            noise_fraction=f"{stats.noise_fraction:.2%}",
            avg_cluster_size=f"{stats.avg_cluster_size:.1f}",
            largest_cluster=stats.largest_cluster_size,
            smallest_cluster=stats.smallest_cluster_size
        )
        
        # Save clusters
        if args.output:
            output_path = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = settings.PROCESSED_DATA_DIR / f"clusters_{timestamp}.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("Saving cluster assignments...", output_path=str(output_path))
        with open(output_path, 'w') as f:
            json.dump(dataset.model_dump(mode='json'), f, indent=2, default=str)
        
        logger.info("Cluster assignments saved successfully")
        
        # Print detailed cluster information
        logger.info("\n" + "="*60)
        logger.info("CLUSTER SUMMARY")
        logger.info("="*60)
        
        logger.info(f"Total clusters: {stats.num_clusters}")
        logger.info(f"Noise points: {stats.num_noise_points}")
        logger.info(f"Noise fraction: {stats.noise_fraction:.2%}")
        logger.info(f"Average cluster size: {stats.avg_cluster_size:.1f}")
        logger.info(f"Largest cluster: {stats.largest_cluster_size} errors")
        logger.info(f"Smallest cluster: {stats.smallest_cluster_size} errors")
        
        if stats.num_clusters > 0:
            logger.info("\nCluster size distribution:")
            for cluster_id in sorted(stats.cluster_sizes.keys()):
                size = stats.cluster_sizes[cluster_id]
                percentage = (size / stats.total_points) * 100
                logger.info(f"  Cluster {cluster_id:3d}: {size:4d} errors ({percentage:5.1f}%)")
        
        logger.info("="*60)
        logger.info(f"\nOutput file: {output_path}")
        
        return 0
    
    except ClusteringError as e:
        logger.error(f"Clustering failed: {e}")
        return 1
    except Exception as e:
        logger.exception("Unexpected error during clustering")
        return 1


if __name__ == "__main__":
    exit(main())
