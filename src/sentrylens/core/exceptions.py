"""
Custom exceptions for the SentryLens application.
"""


class SentryLensError(Exception):
    """Base exception for all SentryLens errors."""
    pass


class DataValidationError(SentryLensError):
    """Raised when data validation fails."""
    pass


class EmbeddingError(SentryLensError):
    """Raised when embedding generation fails."""
    pass


class VectorStoreError(SentryLensError):
    """Raised when vector store operations fail."""
    pass


class ClusteringError(SentryLensError):
    """Raised when clustering operations fail."""
    pass


class DataLoadError(SentryLensError):
    """Raised when data loading fails."""
    pass