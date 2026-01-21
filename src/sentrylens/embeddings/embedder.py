"""
Error embedding generation using Sentence-Transformers.
Implements batch processing, caching, and error handling.
"""
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.sentrylens.core.models import AERIErrorRecord, ErrorEmbedding
from src.sentrylens.core.exceptions import EmbeddingError
from src.sentrylens.utils.logger import logger
from configs.settings import settings


class ErrorEmbedder:
    """
    Generate embeddings for error records using pre-trained models.
    Supports batching, GPU acceleration, and progress tracking.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None
    ):
        """
        Initialize embedder with specified model.
        
        Args:
            model_name: Name of sentence-transformers model (default: from settings)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            batch_size: Batch size for encoding (default: from settings)
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.batch_size = batch_size or settings.BATCH_SIZE
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(
            "Initializing ErrorEmbedder",
            model_name=self.model_name,
            device=self.device,
            batch_size=self.batch_size
        )
        
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            logger.info(
                "Model loaded successfully",
                embedding_dimension=self.embedding_dim
            )
            
            # Verify dimension matches settings
            if self.embedding_dim != settings.EMBEDDING_DIMENSION:
                logger.warning(
                    "Embedding dimension mismatch",
                    expected=settings.EMBEDDING_DIMENSION,
                    actual=self.embedding_dim
                )
        
        except Exception as e:
            raise EmbeddingError(f"Failed to load model {self.model_name}: {e}")
    
    def prepare_text(self, error: AERIErrorRecord) -> str:
        """
        Convert error record to text suitable for embedding.
        
        Strategy: Combine error type, message, and key stack trace info
        into a structured representation that captures semantic meaning.
        
        Args:
            error: AERIErrorRecord to convert
        
        Returns:
            Formatted text string
        """
        # Extract key information
        error_type = error.error_type
        message = error.error_message
        
        # Get first few frames of stack trace (most relevant)
        stack_lines = error.stack_trace.split('\n')
        # Filter out empty lines and take first 10 meaningful frames
        meaningful_frames = [
            line.strip() 
            for line in stack_lines 
            if line.strip() and not line.strip().startswith('...')
        ][:10]
        
        stack_summary = '\n'.join(meaningful_frames)
        
        # Truncate if needed
        max_length = settings.MAX_STACK_TRACE_LENGTH
        if len(stack_summary) > max_length:
            stack_summary = stack_summary[:max_length] + "..."
        
        # Format as structured text
        text = f"""Error Type: {error_type}
Message: {message}
Stack Trace:
{stack_summary}"""
        
        return text
    
    def embed_single(self, error: AERIErrorRecord) -> ErrorEmbedding:
        """
        Generate embedding for a single error.
        
        Args:
            error: Error record to embed
        
        Returns:
            ErrorEmbedding object
        """
        text = self.prepare_text(error)
        
        try:
            # Generate embedding
            embedding_array = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Convert to list for Pydantic
            embedding_list = embedding_array.tolist()
            
            return ErrorEmbedding(
                error_id=error.error_id,
                embedding=embedding_list,
                model_name=self.model_name
            )
        
        except Exception as e:
            raise EmbeddingError(f"Failed to embed error {error.error_id}: {e}")
    
    def embed_batch(
        self,
        errors: List[AERIErrorRecord],
        show_progress: bool = True
    ) -> List[ErrorEmbedding]:
        """
        Generate embeddings for a batch of errors efficiently.
        
        Args:
            errors: List of error records
            show_progress: Whether to show progress bar
        
        Returns:
            List of ErrorEmbedding objects
        """
        if not errors:
            logger.warning("Empty error list provided to embed_batch")
            return []
        
        logger.info(
            "Starting batch embedding",
            num_errors=len(errors),
            batch_size=self.batch_size
        )
        
        try:
            # Prepare all texts
            texts = [self.prepare_text(error) for error in errors]
            
            # Generate embeddings with batching
            embedding_arrays = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
                device=self.device
            )
            
            # Create ErrorEmbedding objects
            embeddings = []
            for error, embedding_array in zip(errors, embedding_arrays):
                embeddings.append(
                    ErrorEmbedding(
                        error_id=error.error_id,
                        embedding=embedding_array.tolist(),
                        model_name=self.model_name
                    )
                )
            
            logger.info(
                "Batch embedding complete",
                num_embeddings=len(embeddings),
                embedding_dimension=len(embeddings[0].embedding)
            )
            
            return embeddings
        
        except Exception as e:
            raise EmbeddingError(f"Batch embedding failed: {e}")
    
    def get_cache_key(self, error: AERIErrorRecord) -> str:
        """
        Generate a cache key for an error record.
        
        Args:
            error: Error record
        
        Returns:
            Cache key (hash of text + model name)
        """
        text = self.prepare_text(error)
        content = f"{self.model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()


class EmbeddingCache:
    """
    Simple disk-based cache for embeddings to avoid recomputation.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize cache.
        
        Args:
            cache_dir: Directory to store cache files (default: from settings)
        """
        self.cache_dir = cache_dir or settings.EMBEDDINGS_DIR / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized embedding cache", cache_dir=str(self.cache_dir))
    
    def get_cache_path(self, cache_key: str) -> Path:
        """Get file path for a cache key."""
        return self.cache_dir / f"{cache_key}.npy"
    
    def get(self, cache_key: str) -> Optional[np.ndarray]:
        """
        Retrieve embedding from cache.
        
        Args:
            cache_key: Cache key
        
        Returns:
            Embedding array if found, None otherwise
        """
        cache_path = self.get_cache_path(cache_key)
        
        if cache_path.exists():
            try:
                return np.load(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cache for {cache_key}: {e}")
                return None
        
        return None
    
    def set(self, cache_key: str, embedding: np.ndarray) -> None:
        """
        Store embedding in cache.
        
        Args:
            cache_key: Cache key
            embedding: Embedding array
        """
        cache_path = self.get_cache_path(cache_key)
        
        try:
            np.save(cache_path, embedding)
        except Exception as e:
            logger.warning(f"Failed to save cache for {cache_key}: {e}")
    
    def clear(self) -> int:
        """
        Clear all cached embeddings.
        
        Returns:
            Number of cache files deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.npy"):
            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Failed to delete {cache_file}: {e}")
        
        logger.info(f"Cleared {count} cache files")
        return count