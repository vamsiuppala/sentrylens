"""
Unit tests for Pydantic models.
"""
import pytest
from datetime import datetime
from pydantic import ValidationError

from sentrylens.core.models import (
    AERIErrorRecord,
    SeverityLevel,
    ErrorEmbedding,
    ClusterAssignment,
)
from sentrylens.config import settings


class TestAERIErrorRecord:
    """Tests for AERIErrorRecord model."""
    
    def test_valid_error_record(self):
        """Test creating a valid error record."""
        record = AERIErrorRecord(
            error_id="test-123",
            error_type="NullPointerException",
            error_message="Object reference not set",
            stack_trace="at com.example.Main.main(Main.java:42)\nat java.base/..."
        )
        
        assert record.error_id == "test-123"
        assert record.error_type == "NullPointerException"
        assert record.severity == SeverityLevel.UNKNOWN
    
    def test_empty_stack_trace_fails(self):
        """Test that empty stack trace is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            AERIErrorRecord(
                error_id="test-456",
                error_type="TestError",
                error_message="Test",
                stack_trace=""
            )
        
        assert "String should have at least 1 character" in str(exc_info.value)
    
    def test_stack_trace_too_long_fails(self):
        """Test that excessively long stack traces are rejected."""
        with pytest.raises(ValidationError):
            AERIErrorRecord(
                error_id="test-789",
                error_type="TestError",
                error_message="Test",
                stack_trace="x" * 51000  # Exceeds 50k limit
            )
    
    def test_severity_enum(self):
        """Test severity level enumeration."""
        record = AERIErrorRecord(
            error_id="test-critical",
            error_type="OutOfMemoryError",
            error_message="Heap space",
            stack_trace="at ...",
            severity=SeverityLevel.CRITICAL
        )
        
        assert record.severity == SeverityLevel.CRITICAL


class TestErrorEmbedding:
    """Tests for ErrorEmbedding model."""
    
    def test_valid_embedding(self):
        """Test creating a valid embedding."""
        embedding = ErrorEmbedding(
            error_id="test-123",
            embedding=[0.1] * settings.EMBEDDING_DIMENSION,
            model_name="all-MiniLM-L6-v2"
        )
        
        assert len(embedding.embedding) == settings.EMBEDDING_DIMENSION
        assert embedding.model_name == "all-MiniLM-L6-v2"
    
    def test_wrong_dimension_fails(self):
        """Test that wrong embedding dimension is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ErrorEmbedding(
                error_id="test-456",
                embedding=[0.1] * 128,  # Wrong dimension
                model_name="test-model"
            )
        
        assert "Embedding dimension mismatch" in str(exc_info.value)


class TestClusterAssignment:
    """Tests for ClusterAssignment model."""
    
    def test_noise_detection(self):
        """Test noise cluster detection."""
        noise_assignment = ClusterAssignment(
            error_id="test-123",
            cluster_id=-1
        )
        
        assert noise_assignment.is_noise is True
        
        regular_assignment = ClusterAssignment(
            error_id="test-456",
            cluster_id=0
        )
        
        assert regular_assignment.is_noise is False