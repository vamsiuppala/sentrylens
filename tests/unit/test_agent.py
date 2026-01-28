"""
Unit tests for ReAct TriageAgent.

Tests agent initialization, message handling, and reasoning loop.
"""
import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from sentrylens.agent.triage_agent import TriageAgent
from sentrylens.core.models import (
    AERIErrorRecord,
    ClusterAssignment,
    ErrorEmbedding,
)


@pytest.fixture
def sample_errors():
    """Create sample error records."""
    return [
        AERIErrorRecord(
            error_id="error_001",
            error_type="java.lang.NullPointerException",
            error_message="Cannot invoke method getValue() on null",
            stack_trace="at com.example.Service.process(Service.java:42)",
        ),
        AERIErrorRecord(
            error_id="error_002",
            error_type="java.lang.NullPointerException",
            error_message="Null reference in request handler",
            stack_trace="at com.example.Handler.handle(Handler.java:55)",
        ),
        AERIErrorRecord(
            error_id="error_003",
            error_type="java.io.FileNotFoundException",
            error_message="File not found",
            stack_trace="at java.io.FileInputStream.open(FileInputStream.java:195)",
        ),
    ]


@pytest.fixture
def sample_clusters(sample_errors):
    """Create cluster assignments."""
    return [
        ClusterAssignment(
            error_id=sample_errors[0].error_id,
            cluster_id=0,
            distance_to_centroid=0.123,
            cluster_size=5,
        ),
        ClusterAssignment(
            error_id=sample_errors[1].error_id,
            cluster_id=0,
            distance_to_centroid=0.145,
            cluster_size=5,
        ),
        ClusterAssignment(
            error_id=sample_errors[2].error_id,
            cluster_id=1,
            distance_to_centroid=0.089,
            cluster_size=3,
        ),
    ]


@pytest.fixture
def mock_data_dir(tmp_path, sample_errors, sample_clusters):
    """Create mock data directory structure."""
    # Create indexes directory
    indexes_dir = tmp_path / "indexes"
    indexes_dir.mkdir()
    vector_store_dir = indexes_dir / "hnswlib_index_20260120_132919"
    vector_store_dir.mkdir()

    # Create dummy Hnswlib files
    (vector_store_dir / "index.hnsw").write_text("dummy")
    (vector_store_dir / "metadata.pkl").write_text("dummy")

    # Create processed directory with cluster data
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    cluster_data = {
        "errors": [e.model_dump() for e in sample_errors],
        "clusters": [c.model_dump() for c in sample_clusters],
    }

    cluster_file = processed_dir / "clusters_20260120_213215.json"
    with open(cluster_file, 'w') as f:
        json.dump(cluster_data, f)

    return tmp_path


@pytest.fixture
def mock_client():
    """Create mock Anthropic client."""
    return Mock()


class TestTriageAgentInitialization:
    """Tests for TriageAgent initialization."""

    @patch('sentrylens.agent.triage_agent.HnswlibVectorStore')
    @patch('sentrylens.agent.triage_agent.ErrorEmbedder')
    @patch('sentrylens.agent.triage_agent.Anthropic')
    def test_agent_initialization(
        self,
        mock_anthropic,
        mock_embedder_class,
        mock_vector_store_class,
        mock_data_dir,
    ):
        """Test that agent initializes correctly."""
        mock_store = Mock()
        mock_vector_store_class.load.return_value = mock_store

        mock_emb = Mock()
        mock_embedder_class.return_value = mock_emb

        agent = TriageAgent(
            data_dir=mock_data_dir,
            vector_store_path=mock_data_dir / "indexes" / "hnswlib_index_20260120_132919",
        )

        assert agent.model == "claude-3-5-sonnet-20241022"
        assert agent.max_turns == 10
        assert agent.vector_store is not None
        assert agent.embedder is not None

    @patch('sentrylens.agent.triage_agent.HnswlibVectorStore')
    @patch('sentrylens.agent.triage_agent.ErrorEmbedder')
    @patch('sentrylens.agent.triage_agent.Anthropic')
    def test_agent_loads_data(
        self,
        mock_anthropic,
        mock_embedder_class,
        mock_vector_store_class,
        mock_data_dir,
    ):
        """Test that agent loads error data and clusters."""
        mock_store = Mock()
        mock_vector_store_class.load.return_value = mock_store

        mock_emb = Mock()
        mock_embedder_class.return_value = mock_emb

        agent = TriageAgent(
            data_dir=mock_data_dir,
            vector_store_path=mock_data_dir / "indexes" / "hnswlib_index_20260120_132919",
        )

        # Check that data was loaded
        assert len(agent.errors_dict) == 3
        assert len(agent.clusters_dict) == 3

    @patch('sentrylens.agent.triage_agent.HnswlibVectorStore')
    @patch('sentrylens.agent.triage_agent.ErrorEmbedder')
    @patch('sentrylens.agent.triage_agent.Anthropic')
    def test_agent_initializes_tools(
        self,
        mock_anthropic,
        mock_embedder_class,
        mock_vector_store_class,
        mock_data_dir,
    ):
        """Test that agent initializes tools correctly."""
        mock_store = Mock()
        mock_vector_store_class.load.return_value = mock_store

        mock_emb = Mock()
        mock_embedder_class.return_value = mock_emb

        agent = TriageAgent(
            data_dir=mock_data_dir,
            vector_store_path=mock_data_dir / "indexes" / "hnswlib_index_20260120_132919",
        )

        assert agent.tools is not None
        assert hasattr(agent.tools, 'search_similar_errors')
        assert hasattr(agent.tools, 'analyze_stack_trace')
        assert hasattr(agent.tools, 'suggest_fix')


class TestTriageAgentRun:
    """Tests for TriageAgent.run() method."""

    @patch('sentrylens.agent.triage_agent.HnswlibVectorStore')
    @patch('sentrylens.agent.triage_agent.ErrorEmbedder')
    @patch('sentrylens.agent.triage_agent.Anthropic')
    def test_run_returns_string(
        self,
        mock_anthropic_class,
        mock_embedder_class,
        mock_vector_store_class,
        mock_data_dir,
    ):
        """Test that run() returns a string response."""
        mock_store = Mock()
        mock_vector_store_class.load.return_value = mock_store

        mock_emb = Mock()
        mock_embedder_class.return_value = mock_emb

        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Mock Claude response (simple end_turn without tool calls)
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(type="text", text="This is the answer")]

        mock_client.messages.create.return_value = mock_response

        agent = TriageAgent(
            data_dir=mock_data_dir,
            vector_store_path=mock_data_dir / "indexes" / "hnswlib_index_20260120_132919",
        )

        result = agent.run("Help me understand error_001")

        assert isinstance(result, str)
        assert len(result) > 0

    @patch('sentrylens.agent.triage_agent.HnswlibVectorStore')
    @patch('sentrylens.agent.triage_agent.ErrorEmbedder')
    @patch('sentrylens.agent.triage_agent.Anthropic')
    def test_run_handles_tool_calls(
        self,
        mock_anthropic_class,
        mock_embedder_class,
        mock_vector_store_class,
        mock_data_dir,
    ):
        """Test that run() handles tool calls correctly."""
        mock_store = Mock()
        mock_vector_store_class.load.return_value = mock_store

        mock_emb = Mock()
        mock_emb.embed_single.return_value = Mock(embedding=[0.1] * 384)
        mock_embedder_class.return_value = mock_emb

        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # First response: tool call
        tool_call = Mock()
        tool_call.type = "tool_use"
        tool_call.name = "search_similar_errors"
        tool_call.id = "tool_call_001"
        tool_call.input = {"query_text": "NullPointerException"}

        mock_response_1 = Mock()
        mock_response_1.stop_reason = "tool_use"
        mock_response_1.content = [tool_call]

        # Second response: final answer
        mock_response_2 = Mock()
        mock_response_2.stop_reason = "end_turn"
        mock_response_2.content = [Mock(type="text", text="Final answer")]

        mock_client.messages.create.side_effect = [mock_response_1, mock_response_2]

        agent = TriageAgent(
            data_dir=mock_data_dir,
            vector_store_path=mock_data_dir / "indexes" / "hnswlib_index_20260120_132919",
        )

        result = agent.run("Find similar errors")

        assert isinstance(result, str)
        assert "Final answer" in result

    @patch('sentrylens.agent.triage_agent.HnswlibVectorStore')
    @patch('sentrylens.agent.triage_agent.ErrorEmbedder')
    @patch('sentrylens.agent.triage_agent.Anthropic')
    def test_run_max_turns(
        self,
        mock_anthropic_class,
        mock_embedder_class,
        mock_vector_store_class,
        mock_data_dir,
    ):
        """Test that run() respects max_turns limit."""
        mock_store = Mock()
        mock_vector_store_class.load.return_value = mock_store

        mock_emb = Mock()
        mock_emb.embed_single.return_value = Mock(embedding=[0.1] * 384)
        mock_embedder_class.return_value = mock_emb

        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Always return tool calls (never end_turn)
        tool_call = Mock()
        tool_call.type = "tool_use"
        tool_call.name = "search_similar_errors"
        tool_call.id = "tool_call_001"
        tool_call.input = {"query_text": "test"}

        mock_response = Mock()
        mock_response.stop_reason = "tool_use"
        mock_response.content = [tool_call]

        mock_client.messages.create.return_value = mock_response

        agent = TriageAgent(
            data_dir=mock_data_dir,
            vector_store_path=mock_data_dir / "indexes" / "hnswlib_index_20260120_132919",
            max_turns=2,
        )

        result = agent.run("test query")

        # Should hit max turns and return appropriate message
        assert "Max reasoning turns reached" in result


class TestToolExecution:
    """Tests for tool execution in the agent."""

    @patch('sentrylens.agent.triage_agent.HnswlibVectorStore')
    @patch('sentrylens.agent.triage_agent.ErrorEmbedder')
    @patch('sentrylens.agent.triage_agent.Anthropic')
    def test_execute_search_similar_errors(
        self,
        mock_anthropic_class,
        mock_embedder_class,
        mock_vector_store_class,
        mock_data_dir,
    ):
        """Test execution of search_similar_errors tool."""
        mock_store = Mock()
        mock_store.search.return_value = [("error_002", 0.92)]
        mock_vector_store_class.load.return_value = mock_store

        mock_emb = Mock()
        mock_emb.embed_single.return_value = Mock(embedding=[0.1] * 384)
        mock_embedder_class.return_value = mock_emb

        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        agent = TriageAgent(
            data_dir=mock_data_dir,
            vector_store_path=mock_data_dir / "indexes" / "hnswlib_index_20260120_132919",
        )

        result = agent._execute_tool(
            "search_similar_errors",
            {"query_text": "NullPointerException", "top_k": 5},
        )

        # Should return JSON
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    @patch('sentrylens.agent.triage_agent.HnswlibVectorStore')
    @patch('sentrylens.agent.triage_agent.ErrorEmbedder')
    @patch('sentrylens.agent.triage_agent.Anthropic')
    def test_execute_analyze_stack_trace(
        self,
        mock_anthropic_class,
        mock_embedder_class,
        mock_vector_store_class,
        mock_data_dir,
    ):
        """Test execution of analyze_stack_trace tool."""
        mock_store = Mock()
        mock_vector_store_class.load.return_value = mock_store

        mock_emb = Mock()
        mock_embedder_class.return_value = mock_emb

        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        agent = TriageAgent(
            data_dir=mock_data_dir,
            vector_store_path=mock_data_dir / "indexes" / "hnswlib_index_20260120_132919",
        )

        stack_trace = "at com.example.Service.process(Service.java:42)"

        result = agent._execute_tool(
            "analyze_stack_trace",
            {"stack_trace": stack_trace},
        )

        # Should return JSON with analysis
        parsed = json.loads(result)
        assert "exception_type" in parsed

    @patch('sentrylens.agent.triage_agent.HnswlibVectorStore')
    @patch('sentrylens.agent.triage_agent.ErrorEmbedder')
    @patch('sentrylens.agent.triage_agent.Anthropic')
    def test_execute_suggest_fix(
        self,
        mock_anthropic_class,
        mock_embedder_class,
        mock_vector_store_class,
        mock_data_dir,
    ):
        """Test execution of suggest_fix tool."""
        mock_store = Mock()
        mock_store.search.return_value = [("error_002", 0.92)]
        mock_vector_store_class.load.return_value = mock_store

        mock_emb = Mock()
        mock_emb.embed_single.return_value = Mock(embedding=[0.1] * 384)
        mock_embedder_class.return_value = mock_emb

        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        agent = TriageAgent(
            data_dir=mock_data_dir,
            vector_store_path=mock_data_dir / "indexes" / "hnswlib_index_20260120_132919",
        )

        result = agent._execute_tool("suggest_fix", {"error_id": "error_001"})

        # Should return JSON with suggestion
        parsed = json.loads(result)
        assert "error_id" in parsed
        assert "suggestion" in parsed

    @patch('sentrylens.agent.triage_agent.HnswlibVectorStore')
    @patch('sentrylens.agent.triage_agent.ErrorEmbedder')
    @patch('sentrylens.agent.triage_agent.Anthropic')
    def test_execute_unknown_tool(
        self,
        mock_anthropic_class,
        mock_embedder_class,
        mock_vector_store_class,
        mock_data_dir,
    ):
        """Test handling of unknown tool."""
        mock_store = Mock()
        mock_vector_store_class.load.return_value = mock_store

        mock_emb = Mock()
        mock_embedder_class.return_value = mock_emb

        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        agent = TriageAgent(
            data_dir=mock_data_dir,
            vector_store_path=mock_data_dir / "indexes" / "hnswlib_index_20260120_132919",
        )

        result = agent._execute_tool("unknown_tool", {})

        # Should return error message
        parsed = json.loads(result)
        assert "error" in parsed
