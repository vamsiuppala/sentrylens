"""Agent command - Run interactive triage agent."""
import sys
from pathlib import Path

import typer
from rich.console import Console

from sentrylens.agent import TriageAgent
from sentrylens.utils.logger import logger

console = Console()


def agent(
    vector_store: Path = typer.Argument(
        ...,
        help="Path to Hnswlib vector store directory",
    ),
    cluster_data: Path = typer.Argument(
        ...,
        help="Path to clusters JSON file",
    ),
    model: str = typer.Option(
        "claude-3-5-haiku-20241022",
        "--model", "-m",
        help="Claude model to use",
    ),
    max_turns: int = typer.Option(
        10,
        "--max-turns",
        help="Maximum ReAct turns per query",
    ),
    example: bool = typer.Option(
        False,
        "--example",
        help="Run programmatic example instead of interactive mode",
    ),
):
    """Start interactive error triage agent powered by Claude."""
    console.print("[bold blue]SentryLens[/bold blue] - Error Triage Agent")
    console.print()

    if not vector_store.exists():
        console.print(f"[red]Error:[/red] Vector store not found: {vector_store}")
        raise typer.Exit(1)

    if not cluster_data.exists():
        console.print(f"[red]Error:[/red] Cluster data not found: {cluster_data}")
        raise typer.Exit(1)

    logger.info(
        "Starting agent",
        vector_store=str(vector_store),
        cluster_data=str(cluster_data),
        model=model,
    )

    try:
        console.print("Initializing agent...")
        triage_agent = TriageAgent(
            vector_store_path=vector_store,
            cluster_data_path=cluster_data,
            model=model,
            max_turns=max_turns,
        )

        console.print(f"[green]Agent initialized[/green]")
        console.print(f"  Knowledge base: {len(triage_agent.errors_dict)} errors")
        num_clusters = len(set(
            c.cluster_id for c in triage_agent.clusters_dict.values()
            if c.cluster_id != -1
        ))
        console.print(f"  Clusters: {num_clusters}")
        console.print(f"  Model: {model}")
        console.print()

        if example:
            _run_example(triage_agent)
        else:
            _run_interactive(triage_agent)

    except ValueError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        console.print()
        console.print("Setup instructions:")
        console.print("  1. Get your API key from: https://console.anthropic.com/")
        console.print("  2. Set environment variable: export ANTHROPIC_API_KEY=sk-ant-...")
        raise typer.Exit(1)

    except KeyboardInterrupt:
        console.print("\n\nInterrupted by user.")
        sys.exit(0)

    except Exception as e:
        logger.exception("Agent error")
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


def _run_interactive(triage_agent: TriageAgent):
    """Run interactive chat session."""
    console.print("[bold]Interactive Mode[/bold]")
    console.print("-" * 40)
    console.print()
    console.print("Example queries:")
    console.print("  - 'Help me understand error <error_id>'")
    console.print("  - 'Find errors similar to NullPointerException'")
    console.print("  - 'What are common error patterns?'")
    console.print()
    console.print("Type 'exit' or 'quit' to end session.")
    console.print()

    triage_agent.chat()


def _run_example(triage_agent: TriageAgent):
    """Run programmatic example."""
    console.print("[bold]Programmatic Example[/bold]")
    console.print("-" * 40)
    console.print()

    console.print("1. Finding similar errors:")
    response = triage_agent.run("Find errors similar to NullPointerException")
    console.print(response)
    console.print()

    console.print("2. Analyzing a stack trace:")
    stack_trace = """at com.example.UserService.getUser(UserService.java:42)
at com.example.UserController.handleRequest(UserController.java:28)
at javax.servlet.http.HttpServlet.service(HttpServlet.java:750)"""

    response = triage_agent.run(f"Help me understand this stack trace:\n{stack_trace}")
    console.print(response)
