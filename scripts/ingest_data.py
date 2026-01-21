#!/usr/bin/env python3
"""
Main data ingestion script.
Loads raw AERI data, validates it, and saves processed dataset.

Usage:
    python scripts/ingest_data.py --limit 500
    python scripts/ingest_data.py --pattern "*.json" --limit-per-file 100
"""
import argparse
from pathlib import Path

from src.sentrylens.data.loader import AERIDataLoader
from src.sentrylens.core.models import ProcessedDataset
from src.sentrylens.core.exceptions import DataLoadError, DataValidationError
from src.sentrylens.utils.logger import logger
from configs.settings import settings


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest and validate AERI error data"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=settings.AERI_DATA_DIR,
        help="Directory containing AERI JSON files"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help="File pattern to match (default: *.json)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=settings.SAMPLE_SIZE,
        help="Total number of records to load (default: from settings)"
    )
    parser.add_argument(
        "--limit-per-file",
        type=int,
        help="Max records per file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for processed dataset"
    )
    
    return parser.parse_args()


def main():
    """Main ingestion pipeline."""
    args = parse_args()
    
    logger.info(
        "Starting data ingestion",
        data_dir=str(args.data_dir),
        pattern=args.pattern,
        limit=args.limit
    )
    
    try:
        # Initialize loader
        loader = AERIDataLoader(data_dir=args.data_dir)
        
        # Load and validate data
        logger.info("Loading raw data...")
        errors = loader.load_from_directory(
            pattern=args.pattern,
            limit_per_file=args.limit_per_file,
            total_limit=args.limit
        )
        
        if not errors:
            logger.error("No valid errors loaded. Exiting.")
            return 1
        
        # Create processed dataset
        logger.info("Creating processed dataset...")
        dataset = ProcessedDataset(
            errors=errors,
            source_file=str(args.data_dir),
        )
        
        # Save processed dataset
        output_path = loader.save_processed_dataset(dataset, args.output)
        
        # Summary statistics
        logger.info(
            "Ingestion complete! ✅",
            total_errors=dataset.total_errors,
            output_path=str(output_path),
            unique_error_types=len(set(e.error_type for e in errors)),
            avg_stack_trace_length=sum(len(e.stack_trace) for e in errors) / len(errors)
        )
        
        return 0
        
    except (DataLoadError, DataValidationError) as e:
        logger.error(f"Ingestion failed: {e}")
        return 1
    except Exception as e:
        logger.exception("Unexpected error during ingestion")
        return 1


if __name__ == "__main__":
    exit(main())