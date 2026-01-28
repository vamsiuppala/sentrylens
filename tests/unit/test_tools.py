"""
Unit tests for ReAct agent tools.

Tests the three main tools: search_similar_errors, analyze_stack_trace, suggest_fix
"""
import pytest
import json
from unittest.mock import Mock, MagicMock

from sentrylens.agent.tools import TriageTools
from sentrylens.core.models import (
    AERIErrorRecord,
    ClusterAssignment,
    ErrorEmbedding,
)


@pytest.fixture
def sample_error():
    """Create a sample error record."""
    return AERIErrorRecord(
        error_id="abc123def456",
        error_type="java.lang.NullPointerException",
        error_message="Cannot invoke method getValue() on null object reference",
        stack_trace="""at com.example.service.UserService.getValue(UserService.java:42)
at com.example.controller.UserController.handleRequest(UserController.java:28)
at javax.servlet.http.HttpServlet.service(HttpServlet.java:750)
at org.apache.catalina.connector.CoyoteAdapter.service(CoyoteAdapter.java:298)""",
    )


@pytest.fixture
def sample_error2():
    """Create another sample error for clustering."""
    return AERIErrorRecord(
        error_id="def789ghi012",
        error_type="java.lang.NullPointerException",
        error_message="Null pointer in request processing",
        stack_trace="""at com.example.processor.RequestProcessor.process(RequestProcessor.java:101)
at com.example.handler.RequestHandler.handle(RequestHandler.java:55)
at java.base/java.lang.Thread.run(Thread.java:833)""",
    )


@pytest.fixture
def sample_error3():
    """Create a third error in different cluster."""
    return AERIErrorRecord(
        error_id="ghi345jkl678",
        error_type="java.io.FileNotFoundException",
        error_message="File not found: /path/to/missing/file.txt",
        stack_trace="""at java.base/java.io.FileInputStream.open(FileInputStream.java:195)
at java.base/java.io.FileInputStream.<init>(FileInputStream.java:153)
at com.example.file.FileReader.readFile(FileReader.java:42)""",
    )


@pytest.fixture
def cluster_assignment1(sample_error):
    """Create cluster assignment for first error."""
    return ClusterAssignment(
        error_id=sample_error.error_id,
        cluster_id=0,
        distance_to_centroid=0.123,
        cluster_size=5,
    )


@pytest.fixture
def cluster_assignment2(sample_error2):
    """Create cluster assignment for second error (same cluster)."""
    return ClusterAssignment(
        error_id=sample_error2.error_id,
        cluster_id=0,
        distance_to_centroid=0.145,
        cluster_size=5,
    )


@pytest.fixture
def cluster_assignment3(sample_error3):
    """Create cluster assignment for third error (different cluster)."""
    return ClusterAssignment(
        error_id=sample_error3.error_id,
        cluster_id=1,
        distance_to_centroid=0.089,
        cluster_size=3,
    )


@pytest.fixture
def mock_vector_store(sample_error, sample_error2, sample_error3):
    """Create mock vector store."""
    mock_store = Mock()
    mock_store.search.return_value = [
        (sample_error2.error_id, 0.92),
        (sample_error3.error_id, 0.78),
    ]
    mock_store.search_by_error_id.return_value = [
        (sample_error2.error_id, 0.92),
    ]
    return mock_store


@pytest.fixture
def mock_embedder(sample_error):
    """Create mock embedder."""
    mock_emb = Mock()

    # Create a mock ErrorEmbedding
    mock_embedding_obj = Mock(spec=ErrorEmbedding)
    mock_embedding_obj.embedding = [0.1] * 384  # Mock 384-dim embedding

    mock_emb.embed_single.return_value = mock_embedding_obj
    return mock_emb


@pytest.fixture
def triage_tools(
    mock_vector_store,
    mock_embedder,
    sample_error,
    sample_error2,
    sample_error3,
    cluster_assignment1,
    cluster_assignment2,
    cluster_assignment3,
):
    """Create TriageTools instance with mocked dependencies."""
    errors_dict = {
        sample_error.error_id: sample_error,
        sample_error2.error_id: sample_error2,
        sample_error3.error_id: sample_error3,
    }

    clusters_dict = {
        cluster_assignment1.error_id: cluster_assignment1,
        cluster_assignment2.error_id: cluster_assignment2,
        cluster_assignment3.error_id: cluster_assignment3,
    }

    return TriageTools(
        vector_store=mock_vector_store,
        embedder=mock_embedder,
        errors_dict=errors_dict,
        clusters_dict=clusters_dict,
    )


class TestSearchSimilarErrors:
    """Tests for search_similar_errors tool."""

    def test_search_returns_json(self, triage_tools):
        """Test that search returns valid JSON."""
        result = triage_tools.search_similar_errors("NullPointerException")

        # Should be valid JSON
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    def test_search_includes_error_details(self, triage_tools):
        """Test that results include error details."""
        result = triage_tools.search_similar_errors("test query")
        parsed = json.loads(result)

        # Check first result has expected fields
        assert len(parsed) > 0
        first_result = parsed[0]
        assert "error_id" in first_result
        assert "error_type" in first_result
        assert "error_message" in first_result
        assert "similarity_score" in first_result
        assert "cluster_id" in first_result

    def test_search_respects_top_k(self, triage_tools):
        """Test that search respects top_k parameter."""
        # Mock vector store to return many results
        triage_tools.vector_store.search.return_value = [
            ("id1", 0.95),
            ("id2", 0.92),
            ("id3", 0.89),
            ("id4", 0.85),
            ("id5", 0.81),
        ]

        result = triage_tools.search_similar_errors("query", top_k=3)
        parsed = json.loads(result)

        # Should return up to top_k results
        assert len(parsed) <= 3

    def test_search_calls_embedder(self, triage_tools):
        """Test that search uses embedder for query."""
        triage_tools.search_similar_errors("test query")

        # Embedder should be called once
        assert triage_tools.embedder.embed_single.called

    def test_search_handles_invalid_error_ids(self, triage_tools):
        """Test that search handles errors not in dict."""
        # Mock vector store to return invalid IDs
        triage_tools.vector_store.search.return_value = [
            ("nonexistent_id", 0.95),
        ]

        result = triage_tools.search_similar_errors("query")
        parsed = json.loads(result)

        # Should handle gracefully
        assert isinstance(parsed, list)


class TestAnalyzeStackTrace:
    """Tests for analyze_stack_trace tool."""

    def test_analyze_returns_json(self, triage_tools, sample_error):
        """Test that analysis returns valid JSON."""
        result = triage_tools.analyze_stack_trace(sample_error.stack_trace)

        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_analyze_extracts_exception_type(self, triage_tools):
        """Test that exception type is extracted."""
        stack_trace = """java.lang.NullPointerException: Cannot invoke method
at com.example.Service.process(Service.java:42)"""

        result = triage_tools.analyze_stack_trace(stack_trace)
        parsed = json.loads(result)

        assert "exception_type" in parsed
        assert "NullPointerException" in parsed["exception_type"]

    def test_analyze_extracts_frames(self, triage_tools, sample_error):
        """Test that stack frames are extracted."""
        result = triage_tools.analyze_stack_trace(sample_error.stack_trace)
        parsed = json.loads(result)

        assert "key_frames" in parsed
        assert len(parsed["key_frames"]) > 0

        # First frame should have expected fields
        frame = parsed["key_frames"][0]
        assert "method" in frame
        assert "file" in frame
        assert "line_number" in frame

    def test_analyze_identifies_root_cause(self, triage_tools, sample_error):
        """Test that root cause (non-library frame) is identified."""
        result = triage_tools.analyze_stack_trace(sample_error.stack_trace)
        parsed = json.loads(result)

        assert "root_cause" in parsed
        if parsed["root_cause"]:
            # Root cause should be non-library
            method = parsed["root_cause"].get("method", "")
            assert "UserService" in method or "com.example" in method

    def test_analyze_counts_frames(self, triage_tools, sample_error):
        """Test that frame count is correct."""
        result = triage_tools.analyze_stack_trace(sample_error.stack_trace)
        parsed = json.loads(result)

        assert "total_frames" in parsed
        assert parsed["total_frames"] > 0

    def test_analyze_handles_empty_trace(self, triage_tools):
        """Test that empty stack trace is handled."""
        result = triage_tools.analyze_stack_trace("")
        parsed = json.loads(result)

        assert "exception_type" in parsed
        assert parsed["total_frames"] == 0

    def test_analyze_with_multiple_frames(self, triage_tools):
        """Test analysis with many stack frames."""
        long_trace = "\n".join([
            f"at com.example.Class{i}.method{i}(Class{i}.java:{100+i})"
            for i in range(20)
        ])

        result = triage_tools.analyze_stack_trace(long_trace)
        parsed = json.loads(result)

        assert parsed["total_frames"] == 20


class TestSuggestFix:
    """Tests for suggest_fix tool."""

    def test_suggest_fix_returns_json(self, triage_tools, sample_error):
        """Test that suggestion returns valid JSON."""
        result = triage_tools.suggest_fix(sample_error.error_id)

        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_suggest_fix_includes_error_info(self, triage_tools, sample_error):
        """Test that suggestion includes error information."""
        result = triage_tools.suggest_fix(sample_error.error_id)
        parsed = json.loads(result)

        assert parsed["error_id"] == sample_error.error_id
        assert parsed["error_type"] == sample_error.error_type
        assert "suggestion" in parsed

    def test_suggest_fix_includes_cluster_context(self, triage_tools, sample_error):
        """Test that suggestion includes cluster context."""
        result = triage_tools.suggest_fix(sample_error.error_id)
        parsed = json.loads(result)

        assert "cluster_context" in parsed
        cluster_ctx = parsed["cluster_context"]

        if cluster_ctx:
            assert "cluster_id" in cluster_ctx
            assert "cluster_size" in cluster_ctx

    def test_suggest_fix_for_nonexistent_error(self, triage_tools):
        """Test handling of nonexistent error ID."""
        result = triage_tools.suggest_fix("nonexistent_id")
        parsed = json.loads(result)

        # Should return error message
        assert "error" in parsed

    def test_suggest_fix_for_null_pointer_exception(self, triage_tools, sample_error):
        """Test that NullPointerException gets appropriate suggestion."""
        result = triage_tools.suggest_fix(sample_error.error_id)
        parsed = json.loads(result)

        suggestion = parsed.get("suggestion", "").lower()

        # Should mention null checks or defensive programming
        assert any(
            phrase in suggestion
            for phrase in ["null", "check", "defensive"]
        )

    def test_suggest_fix_common_pattern(self, triage_tools, sample_error):
        """Test that common patterns are prioritized."""
        # sample_error has cluster_size=5, so is_common_pattern should be False
        result = triage_tools.suggest_fix(sample_error.error_id)
        parsed = json.loads(result)

        cluster_ctx = parsed.get("cluster_context", {})
        is_common = cluster_ctx.get("is_common_pattern", False)

        # For cluster_size=5, should not be common (threshold is >10)
        assert is_common == False


class TestToolSchemas:
    """Tests for tool schema generation."""

    def test_get_tool_schemas_returns_list(self, triage_tools):
        """Test that get_tool_schemas returns list."""
        schemas = triage_tools.get_tool_schemas()

        assert isinstance(schemas, list)
        assert len(schemas) == 3

    def test_tool_schemas_have_required_fields(self, triage_tools):
        """Test that each schema has required fields."""
        schemas = triage_tools.get_tool_schemas()

        for schema in schemas:
            assert "name" in schema
            assert "description" in schema
            assert "input_schema" in schema

    def test_search_similar_errors_schema(self, triage_tools):
        """Test search_similar_errors schema."""
        schemas = triage_tools.get_tool_schemas()
        search_schema = next(s for s in schemas if s["name"] == "search_similar_errors")

        props = search_schema["input_schema"]["properties"]
        assert "query_text" in props
        assert "top_k" in props

    def test_analyze_stack_trace_schema(self, triage_tools):
        """Test analyze_stack_trace schema."""
        schemas = triage_tools.get_tool_schemas()
        analyze_schema = next(s for s in schemas if s["name"] == "analyze_stack_trace")

        props = analyze_schema["input_schema"]["properties"]
        assert "stack_trace" in props

    def test_suggest_fix_schema(self, triage_tools):
        """Test suggest_fix schema."""
        schemas = triage_tools.get_tool_schemas()
        suggest_schema = next(s for s in schemas if s["name"] == "suggest_fix")

        props = suggest_schema["input_schema"]["properties"]
        assert "error_id" in props
