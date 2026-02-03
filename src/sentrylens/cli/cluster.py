"""Cluster command - Run HDBSCAN clustering on embeddings."""
import json
import hashlib
from pathlib import Path
from typing import Optional
from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table

from sentrylens.config import settings
from sentrylens.core.models import ErrorEmbedding, AERIErrorRecord
from sentrylens.clustering.clusterer import HDBSCANClusterer
from sentrylens.utils.logger import logger

console = Console()


def load_embeddings_file(file_path: Path) -> list[ErrorEmbedding]:
    """Load embeddings from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    embeddings_data = data.get('embeddings', [])
    return [ErrorEmbedding(**e) for e in embeddings_data]


def load_errors_jsonl(file_path: Path) -> list[AERIErrorRecord]:
    """Load errors from JSONL file."""
    errors = []
    with open(file_path, 'r') as f:
        for line in f:
            raw = json.loads(line)

            # Handle both formats: direct AERIErrorRecord or raw AERI data
            if 'stack_trace' in raw:
                # Already in AERIErrorRecord format
                errors.append(AERIErrorRecord.model_validate(raw))
            else:
                # Raw AERI format - convert
                stacktraces = raw.get('stacktraces', [])
                stack_trace_str = "Stack trace unavailable"
                if stacktraces and isinstance(stacktraces[0], list):
                    frames = stacktraces[0][:20]
                    frame_strs = [
                        f"  at {frame.get('cN', 'Unknown')}.{frame.get('mN', 'method')} ({frame.get('fN', 'file')}:{frame.get('lN', '?')})"
                        for frame in frames if isinstance(frame, dict)
                    ]
                    stack_trace_str = "Stack trace:\n" + "\n".join(frame_strs)

                error_id = raw.get('error_id')
                if not error_id:
                    content = f"{raw.get('summary', '')}:{raw.get('kind', '')}"
                    error_id = hashlib.md5(content.encode()).hexdigest()

                errors.append(AERIErrorRecord(
                    error_id=error_id,
                    error_type=raw.get('kind', 'Unknown'),
                    error_message=raw.get('summary', ''),
                    stack_trace=stack_trace_str,
                    java_version=raw.get('javaRuntimeVersion', ''),
                    os_name=raw.get('osgiOs', ''),
                ))
    return errors


def cluster(
    embeddings_file: Path = typer.Argument(
        ...,
        help="Embeddings JSON file from embed step",
    ),
    errors_file: Path = typer.Argument(
        ...,
        help="Errors JSONL file from ingest step",
    ),
    min_cluster_size: int = typer.Option(
        5,
        "--min-cluster-size",
        help="Minimum cluster size (smaller = more clusters)",
    ),
    min_samples: Optional[int] = typer.Option(
        None,
        "--min-samples",
        help="Minimum samples in neighborhood (default: same as min-cluster-size)",
    ),
    epsilon: float = typer.Option(
        0.0,
        "--epsilon",
        help="Distance threshold for cluster selection",
    ),
    metric: str = typer.Option(
        "euclidean",
        "--metric",
        help="Distance metric: euclidean, cosine, manhattan",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output JSON file (default: auto-generated)",
    ),
    generate_labels: bool = typer.Option(
        True,
        "--labels/--no-labels",
        help="Generate human-readable cluster labels using Claude API",
    ),
) -> Path:
    """Run HDBSCAN clustering on error embeddings."""
    console.print("[bold blue]SentryLens[/bold blue] - Clustering")
    console.print()

    if not embeddings_file.exists():
        console.print(f"[red]Error:[/red] Embeddings file not found: {embeddings_file}")
        raise typer.Exit(1)

    if not errors_file.exists():
        console.print(f"[red]Error:[/red] Errors file not found: {errors_file}")
        raise typer.Exit(1)

    logger.info(
        "Starting clustering",
        embeddings=str(embeddings_file),
        errors=str(errors_file),
        min_cluster_size=min_cluster_size,
        metric=metric,
    )

    # Load data
    console.print(f"Loading embeddings from: {embeddings_file}")
    embeddings = load_embeddings_file(embeddings_file)
    console.print(f"[green]Loaded {len(embeddings)} embeddings[/green]")

    console.print(f"Loading errors from: {errors_file}")
    errors = load_errors_jsonl(errors_file)
    console.print(f"[green]Loaded {len(errors)} errors[/green]")

    # Initialize clusterer
    console.print("Running HDBSCAN clustering...")
    clusterer = HDBSCANClusterer(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        metric=metric,
    )

    # Progress callback for label generation
    def on_label(cluster_id: int, label: str):
        console.print(f"  Cluster {cluster_id}: [cyan]{label}[/cyan]")

    # Perform clustering (with optional label generation)
    if generate_labels:
        console.print("Clustering and generating labels...")
    cluster_assignments, stats = clusterer.cluster_embeddings(
        embeddings,
        errors=errors,
        generate_labels=generate_labels,
        label_progress_callback=on_label if generate_labels else None,
    )

    # Display results
    console.print()
    table = Table(title="Clustering Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total clusters", str(stats.num_clusters))
    table.add_row("Noise points", str(stats.num_noise_points))
    table.add_row("Noise fraction", f"{stats.noise_fraction:.1%}")
    table.add_row("Average cluster size", f"{stats.avg_cluster_size:.1f}")
    table.add_row("Largest cluster", str(stats.largest_cluster_size))
    table.add_row("Smallest cluster", str(stats.smallest_cluster_size))
    if stats.cluster_labels:
        table.add_row("Labels generated", str(len(stats.cluster_labels)))

    console.print(table)

    # Generate output path
    if output is None:
        timestamp = embeddings_file.stem.replace('embeddings_', '')
        output = settings.PROCESSED_DATA_DIR / f"clusters_{timestamp}.json"

    output.parent.mkdir(parents=True, exist_ok=True)

    # Save results
    console.print(f"\nSaving clusters to: {output}")
    output_data = {
        'total_clusters': stats.num_clusters,
        'total_points': stats.total_points,
        'noise_points': stats.num_noise_points,
        'clusters': [c.model_dump(mode='json') for c in cluster_assignments],
        'errors': [e.model_dump(mode='json') for e in errors],
        'cluster_sizes': stats.cluster_sizes,
        'cluster_labels': stats.cluster_labels,
        'source_embeddings_file': str(embeddings_file),
        'source_errors_file': str(errors_file),
        'created_at': datetime.now().isoformat(),
    }

    with open(output, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    console.print(f"[green]Clustering complete![/green]")
    logger.info("Clustering complete", output=str(output), num_clusters=stats.num_clusters)

    return output
