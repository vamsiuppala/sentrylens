"""
ReAct Agent Tools for error triage.

Implements three main tools:
1. search_similar_errors: Find errors similar to a query using vector similarity
2. analyze_stack_trace: Parse stack traces for structured information
3. suggest_fix: Generate fix recommendations based on error context
"""
import re
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

from src.sentrylens.core.models import AERIErrorRecord, ClusterAssignment
from src.sentrylens.embeddings.vector_store import FAISSVectorStore
from src.sentrylens.embeddings.embedder import ErrorEmbedder
from src.sentrylens.utils.logger import logger


class TriageTools:
    """
    Collection of tools for error triage.
    Each tool integrates with the embeddings, clustering, and vector store
    created in Steps 1-3.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedder: ErrorEmbedder,
        errors_dict: Dict[str, AERIErrorRecord],
        clusters_dict: Dict[str, ClusterAssignment],
    ):
        """
        Initialize tools with references to Step 1-3 infrastructure.

        Args:
            vector_store: FAISS vector store for similarity search (Step 2)
            embedder: ErrorEmbedder for generating embeddings (Step 2)
            errors_dict: Dictionary mapping error_id to AERIErrorRecord (Step 1)
            clusters_dict: Dictionary mapping error_id to ClusterAssignment (Step 3)
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.errors_dict = errors_dict
        self.clusters_dict = clusters_dict

        logger.info(
            "Initialized TriageTools",
            total_errors=len(errors_dict),
            total_clusters=len(set(c.cluster_id for c in clusters_dict.values())),
        )

    def search_similar_errors(self, query_text: str, top_k: int = 5) -> str:
        """
        Find errors similar to a query using vector similarity search.

        Uses the ErrorEmbedder and FAISSVectorStore from Step 2 to find semantically
        similar errors based on embedding distance.

        Args:
            query_text: Text to search for (can be error type, message, or stack trace)
            top_k: Number of similar errors to return

        Returns:
            JSON string with list of similar errors
        """
        logger.info(
            "Searching for similar errors",
            query_text_length=len(query_text),
            top_k=top_k,
        )

        try:
            # Create temporary error record from query text
            from src.sentrylens.core.models import AERIErrorRecord

            temp_error = AERIErrorRecord(
                error_id="query",
                error_type="Query",
                error_message=query_text[:200],
                stack_trace=query_text if "\n" in query_text else "Query stack trace",
            )

            # Generate embedding for query (Step 2: Embedder)
            query_embedding_obj = self.embedder.embed_single(temp_error)
            query_embedding = query_embedding_obj.embedding

            # Search vector store (Step 2: FAISS)
            results = self.vector_store.search(
                query_embedding, top_k=top_k
            )

            # Retrieve full error details and cluster info
            similar_errors = []
            for error_id, similarity_score in results:
                error = self.errors_dict.get(error_id)
                if not error:
                    continue

                cluster_assignment = self.clusters_dict.get(error_id)

                similar_errors.append({
                    "error_id": error_id,
                    "error_type": error.error_type,
                    "error_message": error.error_message[:150],  # Truncate
                    "similarity_score": f"{similarity_score:.4f}",
                    "cluster_id": (
                        cluster_assignment.cluster_id
                        if cluster_assignment
                        else None
                    ),
                    "is_noise": (
                        cluster_assignment.is_noise
                        if cluster_assignment
                        else None
                    ),
                    "cluster_size": (
                        cluster_assignment.cluster_size
                        if cluster_assignment
                        else None
                    ),
                })

            logger.info(
                "Similar errors found",
                num_results=len(similar_errors),
            )

            return json.dumps(similar_errors, indent=2)

        except Exception as e:
            error_msg = f"Error searching for similar errors: {str(e)}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg, "results": []})

    def analyze_stack_trace(self, stack_trace: str) -> str:
        """
        Parse stack trace into structured information.

        Extracts exception type, frames, line numbers, and identifies root cause.
        This is a standalone tool that doesn't require embeddings or clustering.

        Args:
            stack_trace: Full stack trace text

        Returns:
            JSON string with structured stack trace analysis
        """
        logger.info(
            "Analyzing stack trace",
            trace_length=len(stack_trace),
        )

        try:
            frames = []
            exception_type = None

            # Parse stack trace for frames
            # Match lines like: "at com.example.Service.process(Service.java:42)"
            frame_pattern = r'\s*at\s+([^\(]+)\(([^:]+):(\d+)\)'

            for line in stack_trace.split('\n'):
                # Extract exception type from first line if present
                if not exception_type and ':' in line and ' at ' not in line:
                    parts = line.split(':')
                    if len(parts) >= 1:
                        exception_type = parts[0].strip()

                # Extract frames
                match = re.match(frame_pattern, line)
                if match:
                    method = match.group(1).strip()
                    file_name = match.group(2).strip()
                    line_num = match.group(3).strip()

                    frames.append({
                        "method": method,
                        "file": file_name,
                        "line_number": int(line_num),
                    })

            # Identify root cause (first non-library frame)
            root_cause = None
            library_packages = [
                'java.', 'javax.', 'sun.', 'org.eclipse.',
                'org.springframework.', 'com.sun.'
            ]

            for frame in frames:
                is_library = any(
                    lib in frame['method']
                    for lib in library_packages
                )
                if not is_library:
                    root_cause = frame
                    break

            # If no root cause found, use first frame
            if not root_cause and frames:
                root_cause = frames[0]

            # Default exception type if not found
            if not exception_type:
                exception_type = "Unknown Exception"

            result = {
                "exception_type": exception_type,
                "total_frames": len(frames),
                "root_cause": root_cause,
                "key_frames": frames[:5],
                "stack_depth": len(frames),
            }

            logger.info(
                "Stack trace analyzed",
                exception_type=exception_type,
                num_frames=len(frames),
            )

            return json.dumps(result, indent=2)

        except Exception as e:
            error_msg = f"Error analyzing stack trace: {str(e)}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg, "exception_type": "Unknown"})

    def suggest_fix(self, error_id: str) -> str:
        """
        Generate fix recommendations based on error context.

        Combines information from:
        - Step 1: Error details and stack traces
        - Step 2: Similar errors from vector search
        - Step 3: Cluster context and pattern information

        Args:
            error_id: ID of error to generate fix for

        Returns:
            JSON string with fix recommendations
        """
        logger.info("Generating fix suggestion", error_id=error_id)

        try:
            # Step 1: Get error details
            error = self.errors_dict.get(error_id)
            if not error:
                return json.dumps({
                    "error": f"Error {error_id} not found",
                    "suggestion": None,
                })

            # Step 2: Find similar errors
            similar_results = self.search_similar_errors(
                f"{error.error_type}: {error.error_message}",
                top_k=5,
            )
            try:
                similar = json.loads(similar_results)
            except json.JSONDecodeError:
                similar = []

            # Step 3: Get cluster context
            cluster_assignment = self.clusters_dict.get(error_id)
            cluster_context = {}

            if cluster_assignment and not cluster_assignment.is_noise:
                # Get all errors in same cluster
                cluster_members = [
                    eid for eid, c in self.clusters_dict.items()
                    if c.cluster_id == cluster_assignment.cluster_id
                ]

                # Get error types distribution
                error_types = [
                    self.errors_dict[eid].error_type
                    for eid in cluster_members[:10]
                    if eid in self.errors_dict
                ]

                cluster_context = {
                    "cluster_id": cluster_assignment.cluster_id,
                    "cluster_size": cluster_assignment.cluster_size,
                    "common_error_types": list(set(error_types)),
                    "is_common_pattern": (
                        cluster_assignment.cluster_size > 10
                        if cluster_assignment.cluster_size else False
                    ),
                }

            # Analyze the target error's stack trace
            analysis_result = self.analyze_stack_trace(error.stack_trace)
            try:
                stack_analysis = json.loads(analysis_result)
            except json.JSONDecodeError:
                stack_analysis = {}

            # Build comprehensive context
            suggestion = self._generate_suggestion(
                error=error,
                stack_analysis=stack_analysis,
                similar_errors=similar[:3],  # Top 3
                cluster_context=cluster_context,
            )

            logger.info(
                "Fix suggestion generated",
                error_id=error_id,
                cluster_size=cluster_context.get("cluster_size"),
            )

            return json.dumps({
                "error_id": error_id,
                "error_type": error.error_type,
                "error_message": error.error_message,
                "root_cause": stack_analysis.get("root_cause"),
                "cluster_context": cluster_context,
                "suggestion": suggestion,
            }, indent=2)

        except Exception as e:
            error_msg = f"Error generating fix suggestion: {str(e)}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg, "suggestion": None})

    def _generate_suggestion(
        self,
        error: AERIErrorRecord,
        stack_analysis: Dict[str, Any],
        similar_errors: List[Dict[str, Any]],
        cluster_context: Dict[str, Any],
    ) -> str:
        """
        Generate textual suggestion based on error analysis.

        Args:
            error: The target error
            stack_analysis: Analysis of the stack trace
            similar_errors: List of similar errors
            cluster_context: Clustering context

        Returns:
            Textual suggestion string
        """
        suggestions = []

        # Rule 1: Common patterns in cluster
        if cluster_context.get("is_common_pattern"):
            cluster_size = cluster_context.get("cluster_size", 0)
            suggestions.append(
                f"This is a COMMON pattern ({cluster_size} similar errors). "
                f"Prioritize this fix as it affects multiple instances."
            )

        # Rule 2: NullPointerException
        if "NullPointerException" in error.error_type:
            root_cause = stack_analysis.get("root_cause", {})
            method = root_cause.get("method", "")

            suggestions.append(
                f"NullPointerException in {method}. "
                f"Add null checks before accessing object fields. "
                f"Consider using Optional or defensive programming."
            )

        # Rule 3: FileNotFoundException
        elif "FileNotFoundException" in error.error_type:
            suggestions.append(
                "File not found. Verify file path, check if file exists before accessing, "
                "or handle the exception with a fallback mechanism."
            )

        # Rule 4: OutOfMemoryError
        elif "OutOfMemoryError" in error.error_type:
            suggestions.append(
                "Out of memory. Check for memory leaks, reduce data structures, "
                "or increase heap size. Profile the application to find the leak."
            )

        # Rule 5: Similar patterns
        if similar_errors:
            suggestions.append(
                f"Found {len(similar_errors)} similar errors. "
                f"Review their fixes for patterns."
            )

        # Default suggestion
        if not suggestions:
            root_cause = stack_analysis.get("root_cause", {})
            method = root_cause.get("method", "Unknown method")

            suggestions.append(
                f"The error occurs in {method}. "
                f"Review the stack trace and add appropriate error handling. "
                f"Consider adding logging to debug the issue."
            )

        return " ".join(suggestions)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Get tool schemas for Claude API.

        Returns:
            List of tool schema dictionaries compatible with Anthropic API
        """
        return [
            {
                "name": "search_similar_errors",
                "description": (
                    "Search for errors similar to a query using vector similarity. "
                    "Use this when you need to find related errors or patterns."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query_text": {
                            "type": "string",
                            "description": "Error message, exception type, or stack trace to search for",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of similar errors to return",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 20,
                        },
                    },
                    "required": ["query_text"],
                },
            },
            {
                "name": "analyze_stack_trace",
                "description": (
                    "Parse a stack trace to extract structured information. "
                    "Use this to understand the error flow and identify root cause."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "stack_trace": {
                            "type": "string",
                            "description": "Full stack trace text to analyze",
                        },
                    },
                    "required": ["stack_trace"],
                },
            },
            {
                "name": "suggest_fix",
                "description": (
                    "Generate fix recommendations for a known error. "
                    "Use this when you have an error ID and want to understand how to fix it."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "error_id": {
                            "type": "string",
                            "description": "ID of the error to generate a fix for",
                        },
                    },
                    "required": ["error_id"],
                },
            },
        ]
