"""Main CLI application."""
import typer

from sentrylens.cli.ingest import ingest
from sentrylens.cli.embed import embed
from sentrylens.cli.cluster import cluster
from sentrylens.cli.agent import agent
from sentrylens.cli.pipeline import pipeline
from sentrylens.cli.serve import serve

app = typer.Typer(
    name="sentrylens",
    help="Agentic AI system for error triage.",
    add_completion=False,
)

app.command()(ingest)
app.command()(embed)
app.command()(cluster)
app.command()(agent)
app.command()(pipeline)
app.command()(serve)


if __name__ == "__main__":
    app()
