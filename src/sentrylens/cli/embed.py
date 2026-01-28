"""Embed command - Generate embeddings and create Hnswlib index."""
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from sentrylens.config import settings
from sentrylens.core.models import AERIErrorRecord, ErrorEmbedding
from sentrylens.embeddings.embedder import ErrorEmbedder
from sentrylens.embeddings.vector_store import HnswlibVectorStore
from sentrylens.utils.logger import logger

console = Console()


def embed(
    input_file: Path = typer.Argument(
        ...,
        help="Input JSONL file from ingest step",
    ),
    model: str = typer.Option(
        "all-MiniLM-L6-v2",
        "--model", "-m",
        help="Sentence-transformers model name",
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size", "-b",
        help="Batch size for embedding generation",
    ),
    use_gpu: bool = typer.Option(
        False,
        "--use-gpu",
        help="Use GPU for embedding generation",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output directory for embeddings and index",
    ),
) -> tuple[Path, Path]:
    """Generate embeddings for errors and create Hnswlib vector index."""
    console.print("[bold blue]SentryLens[/bold blue] - Embedding Generation")
    console.print()

    if not input_file.exists():
        console.print(f"[red]Error:[/red] Input file not found: {input_file}")
        raise typer.Exit(1)

    logger.info(
        "Starting embedding generation",
        input=str(input_file),
        model=model,
        batch_size=batch_size,
        use_gpu=use_gpu,
    )

    # Load errors from JSONL
    console.print(f"Loading errors from: {input_file}")
    errors = []
    with open(input_file, 'r') as f:
        for line in f:
            errors.append(AERIErrorRecord.model_validate_json(line))

    console.print(f"[green]Loaded {len(errors)} errors[/green]")

    # Initialize embedder
    device = "cuda" if use_gpu else "cpu"
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Loading embedding model...", total=None)
        embedder = ErrorEmbedder(model_name=model, device=device)

    console.print(f"[green]Model loaded:[/green] {model}")

    # Generate embeddings
    console.print("Generating embeddings...")
    embeddings = []

    with Progress(console=console) as progress:
        task = progress.add_task("Embedding errors...", total=len(errors))

        for i in range(0, len(errors), batch_size):
            batch = errors[i:i + batch_size]
            batch_embeddings = embedder.embed_batch(batch)
            embeddings.extend(batch_embeddings)
            progress.update(task, advance=len(batch))

    console.print(f"[green]Generated {len(embeddings)} embeddings[/green]")

    # Create vector store
    console.print("Creating Hnswlib index...")
    vector_store = HnswlibVectorStore(dimension=settings.EMBEDDING_DIMENSION)
    vector_store.add_embeddings(embeddings, errors)

    # Generate output paths
    timestamp = input_file.stem.replace('aeri_', '')

    if output_dir:
        embeddings_path = output_dir / f"embeddings_{timestamp}.json"
        index_path = output_dir / f"hnswlib_index_{timestamp}"
    else:
        embeddings_path = settings.EMBEDDINGS_DIR / f"embeddings_{timestamp}.json"
        index_path = settings.INDEXES_DIR / f"hnswlib_index_{timestamp}"

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    # Save embeddings JSON
    console.print(f"Saving embeddings to: {embeddings_path}")
    embeddings_data = {
        'model': model,
        'dimension': settings.EMBEDDING_DIMENSION,
        'total_embeddings': len(embeddings),
        'source_file': str(input_file),
        'created_at': datetime.now().isoformat(),
        'embeddings': [e.model_dump(mode='json') for e in embeddings],
    }
    with open(embeddings_path, 'w') as f:
        json.dump(embeddings_data, f, indent=2, default=str)

    # Save vector store
    console.print(f"Saving index to: {index_path}")
    vector_store.save(index_path)

    console.print()
    console.print("[green]Embedding complete![/green]")
    console.print(f"  Embeddings: {embeddings_path}")
    console.print(f"  Index: {index_path}")

    logger.info(
        "Embedding complete",
        embeddings_path=str(embeddings_path),
        index_path=str(index_path),
        total=len(embeddings),
    )

    return embeddings_path, index_path
