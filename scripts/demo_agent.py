#!/usr/bin/env python
"""
Demo script for the ReAct error triage agent using Ollama.

Shows how to use the TriageAgent with a local Ollama LLM.

Prerequisites:
    1. Install Ollama: https://ollama.ai/
    2. Pull a model: ollama pull gemma3:270m
    3. Start Ollama server: ollama serve

Usage:
    # Interactive mode
    python scripts/demo_agent.py

    # Programmatic usage
    python scripts/demo_agent.py --example

Or in code:
    from src.sentrylens.agent import TriageAgent

    agent = TriageAgent(
        ollama_base_url="http://localhost:11434",
        model="gemma3:270m"
    )
    response = agent.run("Help me fix NullPointerException")
    print(response)
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.sentrylens.agent import TriageAgent
from src.sentrylens.utils.logger import logger


def main():
    """Run interactive agent demo."""
    print("\n" + "=" * 60)
    print("SentryLens Error Triage Agent - Interactive Demo")
    print("=" * 60)

    print("\n✓ Prerequisites Check:")
    print("  • Ollama server running at http://localhost:11434")
    print("  • Model available (e.g., gemma3:270m)")

    try:
        # Initialize agent
        print("\n✓ Initializing agent...")
        agent = TriageAgent(
            ollama_base_url="http://localhost:11434",
            model="gemma3:270m",
        )

        print(f"✓ Agent initialized successfully")
        print(f"✓ Knowledge base: {len(agent.errors_dict)} errors")
        print(f"✓ Clustering info: {len(set(c.cluster_id for c in agent.clusters_dict.values() if c.cluster_id != -1))} clusters")
        print(f"✓ Model: {agent.model} (Ollama)")

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

    except ConnectionError as e:
        logger.error(f"Ollama connection failed: {e}")
        print(f"\n✗ Error: {e}")
        print("\nSetup Instructions:")
        print("  1. Install Ollama from: https://ollama.ai/")
        print("  2. Start the server: ollama serve")
        print("  3. In another terminal, pull a model:")
        print("     ollama pull gemma3:270m")
        print("  4. Run this script again")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n✗ Error: {e}")
        sys.exit(1)


def example_programmatic_usage():
    """
    Example of using the agent programmatically.

    This function shows how to use the agent without interactive mode.
    """
    print("\n" + "=" * 60)
    print("Programmatic Usage Example (Ollama)")
    print("=" * 60)

    try:
        # Initialize agent with Ollama
        print("\nInitializing agent with Ollama...")
        agent = TriageAgent(
            ollama_base_url="http://localhost:11434",
            model="gemma3:270m",
            max_turns=5,
        )
        print("✓ Agent initialized")

        # Example 1: Ask about a specific error
        print("\n1. Getting information about a specific error:")
        print("-" * 40)
        response = agent.run(
            "Help me understand and fix error 3d0af1bba06c8bc40f9eb7c7a56da2c5"
        )
        print(response)

        # Example 2: Find similar errors
        print("\n2. Finding similar errors:")
        print("-" * 40)
        response = agent.run(
            "Find errors similar to NullPointerException"
        )
        print(response)

        # Example 3: Analyze stack trace
        print("\n3. Analyzing a stack trace:")
        print("-" * 40)
        stack_trace = """at com.example.UserService.getUser(UserService.java:42)
at com.example.UserController.handleRequest(UserController.java:28)
at javax.servlet.http.HttpServlet.service(HttpServlet.java:750)"""

        response = agent.run(
            f"Help me understand this stack trace:\n{stack_trace}"
        )
        print(response)

    except ConnectionError as e:
        print(f"\n✗ Ollama connection failed: {e}")
        print("Make sure Ollama is running: ollama serve")


if __name__ == "__main__":
    # Check if running in demo mode or programmatic mode
    if len(sys.argv) > 1 and sys.argv[1] == "--example":
        # Run example programmatic usage
        example_programmatic_usage()
    else:
        # Run interactive chat
        main()
