#!/usr/bin/env python3
"""
Generate embeddings for error data and create Hnswlib index.

Usage:
    python scripts/generate_embeddings.py --input data/processed/aeri_20250122_162620.jsonl
    python scripts/generate_embeddings.py --input data/processed/aeri_20250122_162620.jsonl --use-gpu
"""
import argparse
import json
import hashlib
from pathlib import Path

from sentrylens.embeddings.embedder import ErrorEmbedder
from sentrylens.embeddings.vector_store import HnswlibVectorStore
from sentrylens.core.models import AERIErrorRecord as AERIRecord
from sentrylens.config import settings


def load_jsonl_errors(jsonl_path: Path) -> list[dict]:
    """Load raw AERI dicts from JSONL file."""
    errors = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            errors.append(json.loads(line))
    return errors


def normalize_to_aeri_record(raw_error: dict) -> AERIRecord:
    """Convert raw AERI dict to AERIErrorRecord."""
    # Format first stacktrace only (limit frames to avoid exceeding max length)
    stacktraces = raw_error.get('stacktraces', [])
    stack_trace_str = "Stack trace unavailable"

    if stacktraces and isinstance(stacktraces[0], list):
        frames = stacktraces[0][:20]  # Limit to first 20 frames
        frame_strs = [
            f"  at {frame.get('cN', 'Unknown')}.{frame.get('mN', 'method')} ({frame.get('fN', 'file')}:{frame.get('lN', '?')})"
            for frame in frames if isinstance(frame, dict)
        ]
        stack_trace_str = "Stack trace:\n" + "\n".join(frame_strs)

    # Generate error_id if missing
    error_id = raw_error.get('error_id')
    if not error_id:
        content = f"{raw_error.get('summary', '')}:{raw_error.get('kind', '')}"
        error_id = hashlib.md5(content.encode()).hexdigest()

    return AERIRecord(
        error_id=error_id,
        error_type=raw_error.get('kind', 'Unknown'),
        error_message=raw_error.get('summary', ''),
        stack_trace=stack_trace_str,
        java_version=raw_error.get('javaRuntimeVersion', ''),
        os_name=raw_error.get('osgiOs', ''),
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings and create Hnswlib index"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to JSONL file (e.g., aeri_20250122_162620.jsonl)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=settings.EMBEDDING_MODEL,
        help=f"Embedding model (default: {settings.EMBEDDING_MODEL})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.BATCH_SIZE,
        help=f"Batch size (default: {settings.BATCH_SIZE})"
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU if available"
    )

    return parser.parse_args()


def main():
    """Main embedding generation pipeline."""
    args = parse_args()

    print(f"📥 Loading JSONL from {args.input}...")
    raw_errors = load_jsonl_errors(args.input)
    print(f"✅ Loaded {len(raw_errors)} raw errors")

    if not raw_errors:
        print("❌ No errors loaded")
        return 1

    # Normalize to AERIErrorRecord
    print("🔄 Normalizing records...")
    errors = [normalize_to_aeri_record(e) for e in raw_errors]
    print(f"✅ Normalized {len(errors)} records")

    # Initialize embedder
    print(f"🤖 Initializing embedder ({args.model})...")
    device = "cuda" if args.use_gpu else "cpu"
    embedder = ErrorEmbedder(
        model_name=args.model,
        device=device,
        batch_size=args.batch_size
    )

    # Generate embeddings
    print("⚙️ Generating embeddings...")
    embeddings = embedder.embed_batch(errors, show_progress=True)
    print(f"✅ Generated {len(embeddings)} embeddings")

    # Extract timestamp from input filename (e.g., aeri_20250122_162620.jsonl)
    timestamp = args.input.stem.replace('aeri_', '')

    # Save embeddings
    embeddings_path = settings.EMBEDDINGS_DIR / f"embeddings_{timestamp}.json"
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings_data = {
        'total_embeddings': len(embeddings),
        'embeddings': [e.model_dump(mode='json') for e in embeddings]
    }

    with open(embeddings_path, 'w') as f:
        json.dump(embeddings_data, f, indent=2, default=str)
    print(f"💾 Saved embeddings to {embeddings_path}")

    # Create vector store
    print("🗂️ Creating Hnswlib index...")
    vector_store = HnswlibVectorStore(dimension=embedder.embedding_dim)
    vector_store.add_embeddings(embeddings, errors)

    # Save index
    index_path = settings.INDEXES_DIR / f"hnswlib_index_{timestamp}"
    vector_store.save(index_path)
    print(f"💾 Saved index to {index_path}")

    # Test search
    print("\n🔍 Testing similarity search...")
    test_results = vector_store.search(embeddings[0].embedding, top_k=5)
    print("Top 5 similar errors:")
    for rank, (error_id, score) in enumerate(test_results, 1):
        error = next(e for e in errors if e.error_id == error_id)
        print(f"  {rank}. {error_id[:12]}... | {error.error_type} | Score: {score:.4f}")

    print("\n✅ Done!")
    return 0


if __name__ == "__main__":
    exit(main())