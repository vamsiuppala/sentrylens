"""Ingest command - Load AERI JSON data to JSONL format."""
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from sentrylens.config import settings
from sentrylens.data.loader import AERIDataLoader
from sentrylens.utils.logger import logger

console = Console()


def ingest(
    input_dir: Optional[Path] = typer.Option(
        None,
        "--input", "-i",
        help="Input directory containing AERI JSON files (default: from config)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output JSONL file path (default: auto-generated)",
    ),
    sample_size: Optional[int] = typer.Option(
        None,
        "--sample-size", "-n",
        help="Limit number of records to ingest",
    ),
) -> Path:
    """Load AERI JSON data and convert to JSONL format."""
    console.print("[bold blue]SentryLens[/bold blue] - Data Ingestion")
    console.print()

    # Use config defaults if not specified
    input_dir = input_dir or settings.AERI_DATA_DIR
    sample_size = sample_size or settings.SAMPLE_SIZE

    logger.info("Starting data ingestion", input_dir=str(input_dir), sample_size=sample_size)

    if not input_dir.exists():
        console.print(f"[red]Error:[/red] Input directory not found: {input_dir}")
        raise typer.Exit(1)

    # Load data
    loader = AERIDataLoader(data_dir=input_dir)

    try:
        errors = loader.load_from_directory(total_limit=sample_size)
    except Exception as e:
        console.print(f"[red]Error loading data:[/red] {e}")
        raise typer.Exit(1)

    if not errors:
        console.print("[yellow]Warning:[/yellow] No errors loaded")
        raise typer.Exit(1)

    console.print(f"[green]Loaded {len(errors)} error records[/green]")

    # Generate output path if not specified
    if output is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = settings.PROCESSED_DATA_DIR / f"aeri_{timestamp}.jsonl"

    output.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSONL
    logger.info("Saving to JSONL", output=str(output))

    with open(output, 'w') as f:
        for error in errors:
            f.write(error.model_dump_json() + '\n')

    console.print(f"[green]Saved to:[/green] {output}")
    logger.info("Ingestion complete", total_errors=len(errors), output=str(output))

    return output
