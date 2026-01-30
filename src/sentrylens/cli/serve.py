"""Serve command - Start the FastAPI server."""
from pathlib import Path

import typer
from rich.console import Console

console = Console()


def serve(
    vector_store: Path = typer.Argument(
        ...,
        help="Path to Hnswlib vector store directory",
    ),
    cluster_data: Path = typer.Argument(
        ...,
        help="Path to clusters JSON file",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Host to bind to",
    ),
    port: int = typer.Option(
        8000,
        "--port", "-p",
        help="Port to bind to",
    ),
):
    """Start the SentryLens API server."""
    console.print("[bold blue]SentryLens[/bold blue] - API Server")
    console.print()

    if not vector_store.exists():
        console.print(f"[red]Error:[/red] Vector store not found: {vector_store}")
        raise typer.Exit(1)

    if not cluster_data.exists():
        console.print(f"[red]Error:[/red] Cluster data not found: {cluster_data}")
        raise typer.Exit(1)

    console.print(f"Vector store: {vector_store}")
    console.print(f"Cluster data: {cluster_data}")
    console.print("Loading data...")

    from sentrylens.api.main import app, init_app
    init_app(vector_store, cluster_data)

    console.print(f"Starting server at [green]http://{host}:{port}[/green]")
    console.print(f"API docs at [cyan]http://{host}:{port}/docs[/cyan]")
    console.print()

    import uvicorn
    uvicorn.run(app, host=host, port=port)
