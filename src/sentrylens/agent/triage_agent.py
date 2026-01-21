"""
ReAct Agent for error triage.

Implements the ReAct (Reasoning + Acting) pattern using Ollama local LLM.
Orchestrates the reasoning process with tool-augmented responses.
"""
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
import requests

from src.sentrylens.data.loader import AERIDataLoader
from src.sentrylens.embeddings.vector_store import FAISSVectorStore
from src.sentrylens.embeddings.embedder import ErrorEmbedder
from src.sentrylens.core.models import AERIErrorRecord, ClusterAssignment
from src.sentrylens.agent.tools import TriageTools
from src.sentrylens.agent.prompts import (
    OLLAMA_AGENT_SYSTEM_PROMPT,
)
from src.sentrylens.utils.logger import logger


class TriageAgent:
    """
    ReAct agent for error triage and fix suggestions.

    Architecture:
    - Uses Ollama local LLM for reasoning
    - Implements prompt-based tool calling (no native tool_use support)
    - Parses tool calls from text using regex
    - Integrates with FAISS vector store for similarity search
    - Leverages embedding and clustering from Steps 1-3
    - Implements ReAct loop for multi-turn reasoning
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        vector_store_path: Optional[Path] = None,
        ollama_base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        max_turns: int = 10,
        timeout: int = 120,
    ):
        """
        Initialize the triage agent.

        Args:
            data_dir: Base data directory (default: project data/)
            vector_store_path: Path to FAISS vector store
            ollama_base_url: URL where Ollama server is running
            model: Ollama model to use (e.g., "llama3.1:8b", "codellama:13b")
            max_turns: Maximum number of ReAct turns
            timeout: Request timeout in seconds
        """
        self.data_dir = data_dir or Path("data")
        self.ollama_base_url = ollama_base_url
        self.model = model
        self.max_turns = max_turns
        self.timeout = timeout

        logger.info(
            "Initializing TriageAgent",
            ollama_url=ollama_base_url,
            model=model,
            max_turns=max_turns,
        )

        # Check Ollama connection
        self._check_ollama_connection()

        # Step 1: Load vector store (Step 2 artifact)
        if vector_store_path is None:
            # Auto-detect latest vector store
            vector_store_path = self._find_latest_vector_store()

        logger.info("Loading vector store", path=str(vector_store_path))
        self.vector_store = FAISSVectorStore.load(vector_store_path)

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

    def _find_latest_vector_store(self) -> Path:
        """
        Auto-detect the latest vector store directory.

        Returns:
            Path to latest vector store
        """
        indexes_dir = self.data_dir / "indexes"
        if not indexes_dir.exists():
            raise FileNotFoundError(f"Indexes directory not found: {indexes_dir}")

        # Find all vector store directories
        vector_stores = sorted(
            [d for d in indexes_dir.iterdir() if d.is_dir()],
            key=lambda p: p.name,
            reverse=True,  # Newest first
        )

        if not vector_stores:
            raise FileNotFoundError(f"No vector stores found in {indexes_dir}")

        path = vector_stores[0]
        logger.info("Auto-detected vector store", path=str(path))
        return path

    def _load_data(self) -> tuple[Dict[str, AERIErrorRecord], Dict[str, ClusterAssignment]]:
        """
        Load error records and cluster assignments.

        Returns:
            Tuple of (errors_dict, clusters_dict)
        """
        # Find latest processed dataset
        processed_dir = self.data_dir / "processed"
        processed_files = sorted(
            [f for f in processed_dir.glob("clusters_*.json")],
            reverse=True,
        )

        if not processed_files:
            raise FileNotFoundError(f"No cluster data found in {processed_dir}")

        cluster_file = processed_files[0]
        logger.info("Loading cluster data", path=str(cluster_file))

        # Load cluster data
        with open(cluster_file, 'r') as f:
            cluster_data = json.load(f)

        # Parse errors and clusters
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

    def _check_ollama_connection(self) -> None:
        """
        Check if Ollama server is reachable.

        Raises:
            ConnectionError: If Ollama server is not responding
        """
        try:
            response = requests.get(
                f"{self.ollama_base_url}/api/tags",
                timeout=5,
            )
            if response.status_code == 200:
                logger.info("Successfully connected to Ollama server")
            else:
                raise ConnectionError(f"Ollama server returned {response.status_code}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.ollama_base_url}. "
                f"Make sure Ollama is running: ollama serve"
            )
        except Exception as e:
            raise ConnectionError(f"Error connecting to Ollama: {e}")

    def run(self, user_query: str) -> str:
        """
        Run the ReAct agent on a user query.

        Implements the ReAct loop:
        1. Build prompt with conversation history
        2. Call Ollama local LLM
        3. Parse response for tool calls using regex
        4. If tool call: execute tool and add result to history
        5. Repeat until model provides final answer or max turns reached

        Args:
            user_query: User's question or request

        Returns:
            Final answer from the agent
        """
        logger.info("Running agent", query_length=len(user_query))

        # Initialize conversation history
        messages: List[Dict[str, str]] = []

        for turn in range(self.max_turns):
            logger.debug(f"ReAct turn {turn + 1}/{self.max_turns}")

            # Build full prompt with conversation history
            full_prompt = self._build_prompt(messages, user_query)

            # Call Ollama
            try:
                response = requests.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": full_prompt,
                        "stream": False,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()

                response_text = response.json()["response"]

            except requests.exceptions.Timeout:
                logger.error("Ollama request timed out")
                return (
                    "Request timed out. The model took too long to respond. "
                    "Try a simpler query or increase the timeout."
                )
            except requests.exceptions.ConnectionError:
                logger.error("Failed to connect to Ollama")
                return (
                    "Cannot connect to Ollama server. "
                    "Make sure it's running: ollama serve"
                )
            except Exception as e:
                logger.error(f"Ollama request failed: {e}")
                return f"Error communicating with Ollama: {e}"

            logger.debug("Ollama response", response_length=len(response_text))

            # Parse response for tool calls
            tool_call = self._parse_tool_call(response_text)

            if tool_call:
                # Tool call detected
                logger.debug(
                    "Executing tool",
                    tool_name=tool_call["name"],
                )

                # Execute the tool
                result = self._execute_tool(
                    tool_name=tool_call["name"],
                    tool_input=tool_call["args"],
                )

                # Add assistant thought/action and tool result to history
                messages.append(("assistant", response_text))
                messages.append(("observation", result))

            else:
                # No tool call - model provided final answer
                logger.info("Agent provided final answer")
                return response_text

        # Max turns reached
        logger.warning(f"Max turns ({self.max_turns}) reached without final answer")
        return "Max reasoning turns reached. Please try a simpler query."

    def _build_prompt(self, messages: List[tuple], user_query: str) -> str:
        """
        Build full prompt for Ollama including system prompt and conversation.

        Args:
            messages: List of (role, content) tuples from conversation
            user_query: The original user query

        Returns:
            Full prompt string for Ollama
        """
        prompt_parts = [OLLAMA_AGENT_SYSTEM_PROMPT, "\n\n"]

        # Add conversation history
        for role, content in messages:
            if role == "assistant":
                prompt_parts.append(f"ASSISTANT:\n{content}\n\n")
            elif role == "observation":
                prompt_parts.append(f"OBSERVATION:\n{content}\n\n")

        # Add current user query on first turn
        if not messages:
            prompt_parts.append(f"USER: {user_query}\n\n")

        # Prompt for next response
        prompt_parts.append("ASSISTANT:")

        return "".join(prompt_parts)

    def _parse_tool_call(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse tool call from Ollama response using regex.

        Expects format:
            THOUGHT: [reasoning]
            ACTION: tool_name(param1="value1", param2=value2)

        Args:
            response_text: Response text from Ollama

        Returns:
            Dict with 'name' and 'args' keys, or None if no tool call
        """
        # Match ACTION: tool_name(...)
        action_pattern = r'ACTION:\s*(\w+)\s*\((.*?)\)\s*(?:\n|$)'
        match = re.search(action_pattern, response_text, re.IGNORECASE)

        if not match:
            return None

        tool_name = match.group(1)
        args_str = match.group(2)

        # Parse arguments
        args = {}
        if args_str.strip():
            # Match key="value" or key=value patterns
            arg_pattern = r'(\w+)\s*=\s*["\']?([^"\',)]*)["\']?'
            arg_matches = re.findall(arg_pattern, args_str)

            for key, value in arg_matches:
                # Try to parse as number if possible
                try:
                    args[key] = int(value)
                except ValueError:
                    try:
                        args[key] = float(value)
                    except ValueError:
                        args[key] = value.strip()

        logger.debug(
            "Parsed tool call",
            tool_name=tool_name,
            args=args,
        )

        return {"name": tool_name, "args": args}

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
        print(f"Model: {self.model} (via Ollama at {self.ollama_base_url})")
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
