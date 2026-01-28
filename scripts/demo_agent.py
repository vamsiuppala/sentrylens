#!/usr/bin/env python
"""
Demo script for the ReAct error triage agent using Claude API.

Usage:
    # Interactive mode
    python scripts/demo_agent.py --vector-store data/indexes/hnswlib_index_* --cluster-data data/processed/clusters_*.json

    # Programmatic example
    python scripts/demo_agent.py --vector-store data/indexes/hnswlib_index_* --cluster-data data/processed/clusters_*.json --example
"""
import argparse
import sys
from pathlib import Path

from sentrylens.agent import TriageAgent
from sentrylens.utils.logger import logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SentryLens Error Triage Agent Demo"
    )
    parser.add_argument(
        "--vector-store",
        type=Path,
        required=True,
        help="Path to Hnswlib vector store directory"
    )
    parser.add_argument(
        "--cluster-data",
        type=Path,
        required=True,
        help="Path to cluster data JSON file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-haiku-20241022",
        help="Claude model to use (default: claude-3-5-haiku-20241022)"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum ReAct turns (default: 10)"
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Run programmatic example instead of interactive mode"
    )
    return parser.parse_args()


def main(args):
    """Run interactive agent demo."""
    print("\n" + "=" * 60)
    print("SentryLens Error Triage Agent - Interactive Demo")
    print("=" * 60)

    print("\n✓ Prerequisites Check:")
    print("  • ANTHROPIC_API_KEY environment variable set")
    print("  • Error embeddings and clustering data available")

    try:
        # Initialize agent
        print("\n✓ Initializing agent...")
        agent = TriageAgent(
            vector_store_path=args.vector_store,
            cluster_data_path=args.cluster_data,
            model=args.model,
            max_turns=args.max_turns,
        )

        print(f"✓ Agent initialized successfully")
        print(f"✓ Knowledge base: {len(agent.errors_dict)} errors")
        print(f"✓ Clustering info: {len(set(c.cluster_id for c in agent.clusters_dict.values() if c.cluster_id != -1))} clusters")
        print(f"✓ Model: {agent.model} (Claude API)")

        # Run interactive chat
        print("\n" + "-" * 60)
        print("Starting interactive session...")
        print("-" * 60)
        print("\nExample queries:")
        print("  • 'Help me understand and fix error 3d0af1bba06c8bc40f9eb7c7a56da2c5'")
        print("  • 'Find errors similar to NullPointerException'")
        print("  • 'What are common error patterns in the data?'")
        print("  • 'Analyze this stack trace: at com.example.Service.process(Service.java:42)'")
        print("\nType 'help' for more examples or 'exit' to quit.\n")

        agent.chat()

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print(f"\n✗ Error: {e}")
        print("\nSetup Instructions:")
        print("  1. Get your Claude API key from: https://console.anthropic.com/")
        print("  2. Create .env file:")
        print("     cp .env.example .env")
        print("  3. Add your API key to .env:")
        print("     ANTHROPIC_API_KEY=sk-ant-...")
        print("  4. Run this script again")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n✗ Error: {e}")
        sys.exit(1)


def example_programmatic_usage(args):
    """Run programmatic example."""
    print("\n" + "=" * 60)
    print("Programmatic Usage Example (Claude API)")
    print("=" * 60)

    try:
        print("\nInitializing agent with Claude API...")
        agent = TriageAgent(
            vector_store_path=args.vector_store,
            cluster_data_path=args.cluster_data,
            model=args.model,
            max_turns=5,
        )
        print("✓ Agent initialized")

        # Example 1: Find similar errors
        print("\n1. Finding similar errors:")
        print("-" * 40)
        response = agent.run("Find errors similar to NullPointerException")
        print(response)

        # Example 2: Analyze stack trace
        print("\n2. Analyzing a stack trace:")
        print("-" * 40)
        stack_trace = """at com.example.UserService.getUser(UserService.java:42)
at com.example.UserController.handleRequest(UserController.java:28)
at javax.servlet.http.HttpServlet.service(HttpServlet.java:750)"""

        response = agent.run(f"Help me understand this stack trace:\n{stack_trace}")
        print(response)

    except ValueError as e:
        print(f"\n✗ Configuration error: {e}")
        print("Make sure ANTHROPIC_API_KEY is set")


if __name__ == "__main__":
    args = parse_args()

    if args.example:
        example_programmatic_usage(args)
    else:
        main(args)
