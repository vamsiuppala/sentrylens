"""
ReAct Agent for error triage.

Implements the ReAct (Reasoning + Acting) pattern using Claude API.
Leverages Claude's native tool_use capability for clean reasoning + action loops.
"""
import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from anthropic import Anthropic

# Fix for macOS OpenMP compatibility
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from sentrylens.data.loader import AERIDataLoader
from sentrylens.embeddings.vector_store import HnswlibVectorStore
from sentrylens.embeddings.embedder import ErrorEmbedder
from sentrylens.core.models import AERIErrorRecord, ClusterAssignment
from sentrylens.agent.tools import TriageTools
from sentrylens.agent.prompts import TRIAGE_AGENT_SYSTEM_PROMPT
from sentrylens.utils.logger import logger


class TriageAgent:
    """
    ReAct agent for error triage and fix suggestions.

    Architecture:
    - Uses Claude API for reasoning via Anthropic SDK
    - Leverages native tool_use capability for clean tool calling
    - Integrates with Hnswlib vector store for similarity search
    - Leverages embedding and clustering from Steps 1-3
    - Implements ReAct loop for multi-turn reasoning with claude-3-5-sonnet
    """

    def __init__(
        self,
        vector_store_path: Path,
        cluster_data_path: Path,
        model: str = "claude-3-5-haiku-20241022",
        max_turns: int = 10,
    ):
        """
        Initialize the triage agent.

        Args:
            vector_store_path: Path to Hnswlib vector store
            cluster_data_path: Path to cluster data JSON file
            model: Claude model to use (default: claude-3-5-haiku-20241022)
            max_turns: Maximum number of ReAct turns
        """
        self.model = model
        self.max_turns = max_turns
        self.cluster_data_path = Path(cluster_data_path)
        self.vector_store_path = Path(vector_store_path)

        # Load environment variables from .env file
        load_dotenv()

        # Initialize Claude client (uses ANTHROPIC_API_KEY environment variable)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. "
                "Set it in .env file or as an environment variable."
            )

        self.client = Anthropic(api_key=api_key)

        logger.info(
            "Initializing TriageAgent",
            model=model,
            max_turns=max_turns,
            vector_store_path=str(self.vector_store_path),
            cluster_data_path=str(self.cluster_data_path),
        )

        # Step 1: Load vector store
        logger.info("Loading vector store", path=str(self.vector_store_path))
        self.vector_store = HnswlibVectorStore.load(self.vector_store_path)

        # Step 2: Initialize embedder (Step 2 artifact)
        logger.info("Initializing embedder")
        self.embedder = ErrorEmbedder()

        # Step 3: Load dataset with errors and clusters
        logger.info("Loading error data and clusters")
        self.errors_dict, self.clusters_dict = self._load_data()

        # Step 4: Initialize tools
        logger.info("Initializing tools")
        self.tools = TriageTools(
            vector_store=self.vector_store,
            embedder=self.embedder,
            errors_dict=self.errors_dict,
            clusters_dict=self.clusters_dict,
        )

        logger.info(
            "TriageAgent initialized successfully",
            total_errors=len(self.errors_dict),
            total_clusters=len(set(
                c.cluster_id for c in self.clusters_dict.values()
                if c.cluster_id != -1
            )),
        )

    def _load_data(self) -> tuple[Dict[str, AERIErrorRecord], Dict[str, ClusterAssignment]]:
        """
        Load error records and cluster assignments from cluster_data_path.

        Returns:
            Tuple of (errors_dict, clusters_dict)
        """
        logger.info("Loading cluster data", path=str(self.cluster_data_path))

        with open(self.cluster_data_path, 'r') as f:
            cluster_data = json.load(f)

        errors_dict = {}
        clusters_dict = {}

        if "clusters" in cluster_data:
            for cluster_item in cluster_data["clusters"]:
                cluster_assign = ClusterAssignment(**cluster_item)
                clusters_dict[cluster_assign.error_id] = cluster_assign

        if "errors" in cluster_data:
            for error_item in cluster_data["errors"]:
                error_record = AERIErrorRecord(**error_item)
                errors_dict[error_record.error_id] = error_record

        logger.info(
            "Data loaded",
            total_errors=len(errors_dict),
            total_assignments=len(clusters_dict),
        )

        return errors_dict, clusters_dict


    def run(self, user_query: str) -> str:
        """
        Run the ReAct agent on a user query.

        Implements the ReAct loop using Claude's native tool_use:
        1. Add user message to conversation
        2. Call Claude with tools
        3. If tool call: execute tool and add result to conversation
        4. Repeat until model provides final answer or max turns reached

        Args:
            user_query: User's question or request

        Returns:
            Final answer from the agent
        """
        logger.info("Running agent", query_length=len(user_query))

        # Initialize conversation messages
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": user_query}
        ]

        for turn in range(self.max_turns):
            logger.debug(f"ReAct turn {turn + 1}/{self.max_turns}")

            # Call Claude with tools
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=TRIAGE_AGENT_SYSTEM_PROMPT,
                    tools=self.tools.get_tool_schemas(),
                    messages=messages,
                )

                logger.debug(
                    "Claude response",
                    stop_reason=response.stop_reason,
                    num_content_blocks=len(response.content),
                )

            except Exception as e:
                logger.error(f"Claude request failed: {e}")
                return f"Error communicating with Claude API: {e}"

            # Check if Claude wants to use a tool
            if response.stop_reason == "tool_use":
                # Extract tool use block
                tool_use_block = None
                text_blocks = []

                for block in response.content:
                    if block.type == "tool_use":
                        tool_use_block = block
                    elif block.type == "text":
                        text_blocks.append(block.text)

                if tool_use_block:
                    # Add Claude's response to messages
                    messages.append({"role": "assistant", "content": response.content})

                    # Execute the tool
                    logger.debug(
                        "Executing tool",
                        tool_name=tool_use_block.name,
                    )

                    result = self._execute_tool(
                        tool_name=tool_use_block.name,
                        tool_input=tool_use_block.input,
                    )

                    # Add tool result to messages
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_block.id,
                                "content": result,
                            }
                        ]
                    })

                    continue

            # No tool use - Claude provided final answer
            if response.stop_reason == "end_turn":
                # Extract text from response
                for block in response.content:
                    if block.type == "text":
                        logger.info("Agent provided final answer")
                        return block.text

        # Max turns reached
        logger.warning(f"Max turns ({self.max_turns}) reached without final answer")
        return "Max reasoning turns reached. Please try a simpler query."


    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of tool to execute
            tool_input: Input parameters for the tool

        Returns:
            Tool result as string
        """
        try:
            if tool_name == "search_similar_errors":
                return self.tools.search_similar_errors(
                    query_text=str(tool_input.get("query_text", "")),
                    top_k=int(tool_input.get("top_k", 5)),
                )

            elif tool_name == "analyze_stack_trace":
                return self.tools.analyze_stack_trace(
                    stack_trace=str(tool_input.get("stack_trace", "")),
                )

            elif tool_name == "suggest_fix":
                return self.tools.suggest_fix(
                    error_id=str(tool_input.get("error_id", "")),
                )

            else:
                return json.dumps({
                    "error": f"Unknown tool: {tool_name}",
                })

        except Exception as e:
            logger.error(
                "Tool execution failed",
                tool_name=tool_name,
                error=str(e),
            )
            return json.dumps({
                "error": f"Tool execution failed: {str(e)}",
            })

    def chat(self) -> None:
        """
        Interactive chat mode for the agent.

        Allows users to ask multiple questions in a session.
        Type 'exit' or 'quit' to end the session.
        """
        logger.info("Starting interactive chat mode")
        print("\n" + "=" * 60)
        print("SentryLens Error Triage Agent - Interactive Mode")
        print("=" * 60)
        print(f"Model: {self.model} (Claude API)")
        print(f"Knowledge base: {len(self.errors_dict)} errors")
        print(f"Max reasoning turns: {self.max_turns}")
        print("Type 'help' for examples or 'exit' to quit\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['exit', 'quit']:
                    print("\nGoodbye!")
                    break

                if user_input.lower() == 'help':
                    self._print_help()
                    continue

                # Run agent
                print("\nAgent thinking...\n")
                response = self.run(user_input)
                print(f"Agent: {response}\n")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Chat error: {e}")
                print(f"\nError: {e}\n")

    def _print_help(self) -> None:
        """Print help message with example queries."""
        examples = [
            "Help me understand error 12345",
            "Find errors similar to NullPointerException",
            "What common patterns exist in our error data?",
            "Generate a fix for OutOfMemoryError in this stack trace: ...",
        ]

        print("\n=== Example Queries ===")
        for i, example in enumerate(examples, 1):
            print(f"{i}. {example}")
        print("\nType your query or 'exit' to quit.\n")
