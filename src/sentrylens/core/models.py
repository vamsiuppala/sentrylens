"""
Pydantic models for type-safe data handling.
Defines the contract for AERI error data and internal representations.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum


class SeverityLevel(str, Enum):
    """Error severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class StackFrame(BaseModel):
    """Individual stack frame in a stack trace."""
    
    file_path: Optional[str] = Field(None, description="Source file path")
    line_number: Optional[int] = Field(None, ge=0, description="Line number")
    method_name: Optional[str] = Field(None, description="Method/function name")
    class_name: Optional[str] = Field(None, description="Class name if applicable")
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class AERIErrorRecord(BaseModel):
    """
    Schema for AERI (Automated Error Reporting Initiative) error records.
    Validates incoming error data from the Eclipse AERI dataset.
    """
    
    # Required fields
    error_id: str = Field(..., description="Unique identifier for the error")
    error_type: str = Field(..., description="Exception class name")
    error_message: str = Field(..., description="Error message")
    stack_trace: str = Field(..., min_length=1, description="Full stack trace")
    
    # Optional metadata
    timestamp: Optional[datetime] = Field(None, description="When error occurred")
    plugin_id: Optional[str] = Field(None, description="Eclipse plugin ID")
    plugin_version: Optional[str] = Field(None, description="Plugin version")
    eclipse_version: Optional[str] = Field(None, description="Eclipse version")
    os_name: Optional[str] = Field(None, description="Operating system")
    java_version: Optional[str] = Field(None, description="Java runtime version")
    
    # Additional context
    severity: SeverityLevel = Field(
        default=SeverityLevel.UNKNOWN,
        description="Error severity level"
    )
    tags: List[str] = Field(default_factory=list, description="Classification tags")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
    )
    
    @field_validator("stack_trace")
    @classmethod
    def validate_stack_trace(cls, v: str) -> str:
        """Ensure stack trace is not empty and has reasonable length."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Stack trace cannot be empty")
        if len(v) > 50000:  # Sanity check
            raise ValueError("Stack trace exceeds maximum length (50k chars)")
        return v.strip()
    
    @field_validator("error_type")
    @classmethod
    def validate_error_type(cls, v: str) -> str:
        """Ensure error type is a valid class name format."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Error type cannot be empty")
        return v.strip()
    
    def get_root_cause_file(self) -> Optional[str]:
        """Extract likely root cause file from stack trace."""
        # Simple heuristic: first non-library frame
        lines = self.stack_trace.split('\n')
        for line in lines:
            if '.java:' in line and not any(
                lib in line.lower() 
                for lib in ['java.', 'javax.', 'sun.', 'org.eclipse.']
            ):
                try:
                    file_path = line.split('(')[1].split(':')[0]
                    return file_path
                except (IndexError, AttributeError):
                    continue
        return None


class ErrorEmbedding(BaseModel):
    """
    Pydantic model representing the embedding of an error.

    This model links an error to its dense vector representation,
    along with metadata such as the embedding model used and the creation timestamp.
    """
    
    error_id: str = Field(..., description="Reference to AERIErrorRecord.error_id")
    embedding: List[float] = Field(..., description="Dense vector representation")
    model_name: str = Field(..., description="Embedding model used")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(validate_assignment=True)
    
    @field_validator("embedding")
    @classmethod
    def validate_embedding_dimension(cls, v: List[float]) -> List[float]:
        """Ensure embedding has expected dimension."""
        from configs.settings import settings
        if len(v) != settings.EMBEDDING_DIMENSION:
            raise ValueError(
                f"Embedding dimension mismatch: expected {settings.EMBEDDING_DIMENSION}, "
                f"got {len(v)}"
            )
        return v


class ClusterAssignment(BaseModel):
    """
    Assignment of an error to a cluster.
    """
    
    error_id: str = Field(..., description="Reference to AERIErrorRecord.error_id")
    cluster_id: int = Field(..., description="Cluster label (-1 for noise)")
    distance_to_centroid: Optional[float] = Field(
        None,
        ge=0.0,
        description="Distance to cluster center"
    )
    cluster_size: Optional[int] = Field(None, ge=1, description="Total errors in cluster")
    
    model_config = ConfigDict(validate_assignment=True)
    
    @property
    def is_noise(self) -> bool:
        """Check if this error is classified as noise."""
        return self.cluster_id == -1


class ProcessedDataset(BaseModel):
    """
    Complete processed dataset with metadata.
    Used for saving/loading processed data with validation.
    """
    
    errors: List[AERIErrorRecord] = Field(..., description="List of error records")
    embeddings: Optional[List[ErrorEmbedding]] = Field(
        None,
        description="Embeddings if computed"
    )
    clusters: Optional[List[ClusterAssignment]] = Field(
        None,
        description="Cluster assignments if computed"
    )
    
    # Metadata
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    source_file: Optional[str] = Field(None, description="Original data source")
    processing_version: str = Field(default="1.0.0", description="Processing pipeline version")
    
    model_config = ConfigDict(validate_assignment=True)
    
    @property
    def total_errors(self) -> int:
        """Total number of errors in dataset."""
        return len(self.errors)
    
    @property
    def has_embeddings(self) -> bool:
        """Check if embeddings are computed."""
        return self.embeddings is not None and len(self.embeddings) > 0
    
    @property
    def has_clusters(self) -> bool:
        """Check if clustering is done."""
        return self.clusters is not None and len(self.clusters) > 0


class SimilarityResult(BaseModel):
    """
    Result from similarity search.
    """
    
    error: AERIErrorRecord = Field(..., description="The similar error")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity")
    rank: int = Field(..., ge=1, description="Rank in results (1 = most similar)")
    
    model_config = ConfigDict(validate_assignment=True)