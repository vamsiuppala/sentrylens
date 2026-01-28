"""
Centralized configuration management.
Uses environment variables with sensible defaults.
"""
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


def _find_project_root() -> Path:
    """Find project root by looking for pyproject.toml."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parent.parent  # Fallback: src/sentrylens -> project root


class Settings(BaseSettings):
    """Application settings with validation."""

    # Project paths
    PROJECT_ROOT: Path = _find_project_root()
    DATA_DIR: Path = PROJECT_ROOT / "data"
    AERI_DATA_DIR: Path = DATA_DIR / "aeri" / "output_problems"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    EMBEDDINGS_DIR: Path = DATA_DIR / "embeddings"
    INDEXES_DIR: Path = DATA_DIR / "indexes"

    # Embedding configuration
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    BATCH_SIZE: int = 32

    # Clustering configuration
    MIN_CLUSTER_SIZE: int = 5
    MIN_SAMPLES: int = 3

    # Data processing
    MAX_STACK_TRACE_LENGTH: int = 5000
    SAMPLE_SIZE: Optional[int] = None  # None for full dataset

    # Logging
    LOG_LEVEL: str = "WARNING"
    LOG_FORMAT: str = "text"

    # API keys (for later agent work)
    ANTHROPIC_API_KEY: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()

# Ensure directories exist
settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
settings.INDEXES_DIR.mkdir(parents=True, exist_ok=True)
