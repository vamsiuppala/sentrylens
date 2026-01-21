#!/usr/bin/env python3
"""
Generate embeddings for processed error data and create FAISS index.

Usage:
    python scripts/generate_embeddings.py --input data/processed/processed_dataset_20240120.json
    python scripts/generate_embeddings.py --input data/processed/processed_dataset_20240120.json --use-gpu
"""
import argparse
import json
from pathlib import Path
from datetime import datetime

from src.sentrylens.data.loader import AERIDataLoader
from src.sentrylens.embeddings.embedder import ErrorEmbedder
from src.sentrylens.embeddings.vector_store import FAISSVectorStore
from src.sentrylens.core.models import ProcessedDataset
from src.sentrylens.core.exceptions import EmbeddingError, VectorStoreError
from src.sentrylens.utils.logger import logger
from configs.settings import settings


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings and create FAISS index"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to processed dataset JSON file"
    )
    parser.add_argument(
        "--output-embeddings",
        type=Path,
        help="Output path for embeddings (default: auto-generated)"
    )
    parser.add_argument(
        "--output-index",
        type=Path,
        help="Output path for FAISS index (default: auto-generated)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=settings.EMBEDDING_MODEL,
        help=f"Embedding model to use (default: {settings.EMBEDDING_MODEL})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.BATCH_SIZE,
        help=f"Batch size for embedding (default: {settings.BATCH_SIZE})"
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU if available"
    )
    parser.add_argument(
        "--index-type",
        type=str,
        default="IndexFlatIP",
        choices=["IndexFlatL2", "IndexFlatIP", "IndexIVFFlat"],
        help="FAISS index type (default: IndexFlatIP for cosine similarity)"
    )
    
    return parser.parse_args()


def main():
    """Main embedding generation pipeline."""
    args = parse_args()
    
    logger.info(
        "Starting embedding generation",
        input_file=str(args.input),
        model=args.model,
        batch_size=args.batch_size,
        use_gpu=args.use_gpu
    )
    
    try:
        # Load processed dataset
        logger.info("Loading processed dataset...")
        loader = AERIDataLoader()
        dataset = loader.load_processed_dataset(args.input)
        
        errors = dataset.errors
        logger.info(f"Loaded {len(errors)} errors")
        
        if not errors:
            logger.error("No errors in dataset. Exiting.")
            return 1
        
        # Initialize embedder
        logger.info("Initializing embedder...")
        device = "cuda" if args.use_gpu else "cpu"
        embedder = ErrorEmbedder(
            model_name=args.model,
            device=device,
            batch_size=args.batch_size
        )
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = embedder.embed_batch(errors, show_progress=True)
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Update dataset with embeddings
        dataset.embeddings = embeddings
        
        # Save updated dataset
        if args.output_embeddings:
            output_path = args.output_embeddings
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = settings.EMBEDDINGS_DIR / f"embeddings_{timestamp}.json"
        
        logger.info("Saving embeddings...", output_path=str(output_path))
        with open(output_path, 'w') as f:
            json.dump(dataset.model_dump(mode='json'), f, indent=2, default=str)
        
        logger.info("Embeddings saved successfully")
        
        # Create FAISS index
        logger.info("Creating FAISS vector store...")
        vector_store = FAISSVectorStore(
            dimension=embedder.embedding_dim,
            index_type=args.index_type
        )
        
        # Add embeddings to vector store
        vector_store.add_embeddings(embeddings, errors)
        
        # Save vector store
        if args.output_index:
            index_path = args.output_index
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            index_path = settings.INDEXES_DIR / f"faiss_index_{timestamp}"
        
        logger.info("Saving FAISS index...", index_path=str(index_path))
        saved_path = vector_store.save(index_path)
        
        # Print statistics
        stats = vector_store.get_stats()
        logger.info(
            "Vector store created successfully! ✅",
            **stats,
            embeddings_file=str(output_path),
            index_path=str(saved_path)
        )
        
        # Test search functionality
        logger.info("\nTesting similarity search...")
        test_query = embeddings[0].embedding
        results = vector_store.search(test_query, top_k=5)
        
        logger.info("Top 5 similar errors to first error:")
        for rank, (error_id, score) in enumerate(results, 1):
            error = next(e for e in errors if e.error_id == error_id)
            logger.info(
                f"  {rank}. {error_id[:12]}... | {error.error_type} | Score: {score:.4f}"
            )
        
        return 0
    
    except (EmbeddingError, VectorStoreError) as e:
        logger.error(f"Embedding generation failed: {e}")
        return 1
    except Exception as e:
        logger.exception("Unexpected error during embedding generation")
        return 1


if __name__ == "__main__":
    exit(main())