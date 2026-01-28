"""Pipeline command - Run full pipeline (ingest -> embed -> cluster)."""
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from sentrylens.config import settings
from sentrylens.cli.ingest import ingest
from sentrylens.cli.embed import embed
from sentrylens.cli.cluster import cluster
from sentrylens.utils.logger import logger

console = Console()


def pipeline(
    input_dir: Optional[Path] = typer.Option(
        None,
        "--input", "-i",
        help="Input directory containing AERI JSON files (default: from config)",
    ),
    sample_size: Optional[int] = typer.Option(
        None,
        "--sample-size", "-n",
        help="Limit number of records to process",
    ),
    model: str = typer.Option(
        "all-MiniLM-L6-v2",
        "--model", "-m",
        help="Sentence-transformers model for embeddings",
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size", "-b",
        help="Batch size for embedding generation",
    ),
    min_cluster_size: int = typer.Option(
        5,
        "--min-cluster-size",
        help="Minimum cluster size for HDBSCAN",
    ),
    use_gpu: bool = typer.Option(
        False,
        "--use-gpu",
        help="Use GPU for embedding generation",
    ),
):
    """Run full pipeline: ingest -> embed -> cluster.

    This chains all processing steps and automatically passes outputs
    between stages. After completion, you can run the agent:

        sentrylens agent <vector_store_path> <cluster_data_path>
    """
    console.print("[bold blue]SentryLens[/bold blue] - Full Pipeline")
    console.print("=" * 50)
    console.print()

    logger.info("Starting full pipeline")

    # Step 1: Ingest
    console.print("[bold cyan]Step 1/3: Data Ingestion[/bold cyan]")
    console.print("-" * 40)

    try:
        jsonl_path = ingest(
            input_dir=input_dir,
            output=None,
            sample_size=sample_size,
        )
    except SystemExit:
        console.print("[red]Pipeline failed at ingestion step[/red]")
        raise typer.Exit(1)

    console.print()

    # Step 2: Embed
    console.print("[bold cyan]Step 2/3: Embedding Generation[/bold cyan]")
    console.print("-" * 40)

    try:
        embeddings_path, index_path = embed(
            input_file=jsonl_path,
            model=model,
            batch_size=batch_size,
            use_gpu=use_gpu,
            output_dir=None,
        )
    except SystemExit:
        console.print("[red]Pipeline failed at embedding step[/red]")
        raise typer.Exit(1)

    console.print()

    # Step 3: Cluster
    console.print("[bold cyan]Step 3/3: Clustering[/bold cyan]")
    console.print("-" * 40)

    try:
        clusters_path = cluster(
            embeddings_file=embeddings_path,
            errors_file=jsonl_path,
            min_cluster_size=min_cluster_size,
            min_samples=None,
            epsilon=0.0,
            metric="euclidean",
            output=None,
        )
    except SystemExit:
        console.print("[red]Pipeline failed at clustering step[/red]")
        raise typer.Exit(1)

    # Summary
    console.print()
    console.print("=" * 50)
    console.print("[bold green]Pipeline Complete![/bold green]")
    console.print("=" * 50)
    console.print()
    console.print("Generated files:")
    console.print(f"  JSONL:      {jsonl_path}")
    console.print(f"  Embeddings: {embeddings_path}")
    console.print(f"  Index:      {index_path}")
    console.print(f"  Clusters:   {clusters_path}")
    console.print()
    console.print("To start the agent:")
    console.print(f"  [cyan]sentrylens agent {index_path} {clusters_path}[/cyan]")

    logger.info(
        "Pipeline complete",
        jsonl=str(jsonl_path),
        embeddings=str(embeddings_path),
        index=str(index_path),
        clusters=str(clusters_path),
    )
