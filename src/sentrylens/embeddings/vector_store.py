"""
FAISS-based vector store for efficient similarity search.
Implements indexing, search, and persistence operations.
"""
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import faiss

from src.sentrylens.core.models import (
    ErrorEmbedding,
    AERIErrorRecord,
    SimilarityResult
)
from src.sentrylens.core.exceptions import VectorStoreError
from src.sentrylens.utils.logger import logger
from configs.settings import settings


class FAISSVectorStore:
    """
    FAISS-based vector store for error embeddings.
    Supports efficient similarity search and persistence.
    """
    
    def __init__(
        self,
        dimension: Optional[int] = None,
        index_type: str = "IndexFlatL2"
    ):
        """
        Initialize vector store.
        
        Args:
            dimension: Embedding dimension (default: from settings)
            index_type: FAISS index type ('IndexFlatL2', 'IndexIVFFlat', etc.)
        """
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self.index_type = index_type
        
        logger.info(
            "Initializing FAISSVectorStore",
            dimension=self.dimension,
            index_type=index_type
        )
        
        # Initialize FAISS index
        self.index = self._create_index()
        
        # Metadata storage
        self.error_ids: List[str] = []  # Maps index position to error_id
        self.id_to_index: Dict[str, int] = {}  # Maps error_id to index position
        self.errors_metadata: Dict[str, Dict[str, Any]] = {}  # Additional metadata
    
    def _create_index(self) -> faiss.Index:
        """
        Create FAISS index based on type.
        
        Returns:
            Initialized FAISS index
        """
        try:
            if self.index_type == "IndexFlatL2":
                # Exact search using L2 distance
                index = faiss.IndexFlatL2(self.dimension)
            
            elif self.index_type == "IndexFlatIP":
                # Exact search using inner product (cosine similarity)
                index = faiss.IndexFlatIP(self.dimension)
            
            elif self.index_type == "IndexIVFFlat":
                # Approximate search with inverted file index
                # Good for larger datasets (>10k vectors)
                quantizer = faiss.IndexFlatL2(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
            
            logger.info(f"Created FAISS index: {self.index_type}")
            return index
        
        except Exception as e:
            raise VectorStoreError(f"Failed to create FAISS index: {e}")
    
    def add_embeddings(
        self,
        embeddings: List[ErrorEmbedding],
        errors: Optional[List[AERIErrorRecord]] = None
    ) -> None:
        """
        Add embeddings to the vector store.
        
        Args:
            embeddings: List of ErrorEmbedding objects
            errors: Optional list of corresponding error records for metadata
        """
        if not embeddings:
            logger.warning("No embeddings provided to add_embeddings")
            return
        
        logger.info(f"Adding {len(embeddings)} embeddings to vector store")
        
        try:
            # Convert embeddings to numpy array
            embedding_matrix = np.array(
                [emb.embedding for emb in embeddings],
                dtype=np.float32
            )
            
            # Normalize for cosine similarity if using IndexFlatIP
            if self.index_type == "IndexFlatIP":
                faiss.normalize_L2(embedding_matrix)
            
            # Add to index
            start_idx = len(self.error_ids)
            self.index.add(embedding_matrix)
            
            # Update metadata
            for idx, embedding in enumerate(embeddings):
                error_id = embedding.error_id
                index_position = start_idx + idx
                
                self.error_ids.append(error_id)
                self.id_to_index[error_id] = index_position
                
                # Store additional metadata if errors provided
                if errors and idx < len(errors):
                    self.errors_metadata[error_id] = {
                        'error_type': errors[idx].error_type,
                        'error_message': errors[idx].error_message,
                        'severity': errors[idx].severity,
                    }
            
            logger.info(
                "Embeddings added successfully",
                total_vectors=len(self.error_ids),
                new_vectors=len(embeddings)
            )
        
        except Exception as e:
            raise VectorStoreError(f"Failed to add embeddings: {e}")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        return_distances: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Search for most similar vectors.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            return_distances: Whether to return distance scores
        
        Returns:
            List of (error_id, distance) tuples, sorted by similarity
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty, cannot search")
            return []
        
        try:
            # Convert to numpy array and normalize if needed
            query_array = np.array([query_embedding], dtype=np.float32)
            
            if self.index_type == "IndexFlatIP":
                faiss.normalize_L2(query_array)
            
            # Search
            distances, indices = self.index.search(query_array, min(top_k, self.index.ntotal))
            
            # Convert results
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx < len(self.error_ids):  # Valid index
                    error_id = self.error_ids[idx]
                    
                    # Convert distance to similarity score (0-1)
                    if self.index_type == "IndexFlatIP":
                        # Inner product is already similarity
                        similarity = float(dist)
                    else:
                        # Convert L2 distance to similarity
                        # Using exponential decay: sim = exp(-distance)
                        similarity = float(np.exp(-dist))
                    
                    results.append((error_id, similarity))
            
            logger.debug(
                "Search complete",
                top_k=top_k,
                results_found=len(results)
            )
            
            return results
        
        except Exception as e:
            raise VectorStoreError(f"Search failed: {e}")
    
    def search_by_error_id(
        self,
        error_id: str,
        top_k: int = 5,
        exclude_self: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Find similar errors to a given error ID.
        
        Args:
            error_id: Error ID to find similar errors for
            top_k: Number of results to return
            exclude_self: Whether to exclude the query error from results
        
        Returns:
            List of (error_id, similarity) tuples
        """
        if error_id not in self.id_to_index:
            raise VectorStoreError(f"Error ID not found in vector store: {error_id}")
        
        # Get the embedding for this error
        idx = self.id_to_index[error_id]
        embedding = self.index.reconstruct(int(idx))
        
        # Search (request k+1 if excluding self)
        search_k = top_k + 1 if exclude_self else top_k
        results = self.search(embedding.tolist(), top_k=search_k)
        
        # Remove self if needed
        if exclude_self:
            results = [(eid, score) for eid, score in results if eid != error_id]
        
        return results[:top_k]
    
    def get_similar_errors(
        self,
        query_embedding: List[float],
        errors_dict: Dict[str, AERIErrorRecord],
        top_k: int = 5
    ) -> List[SimilarityResult]:
        """
        Search and return full SimilarityResult objects.
        
        Args:
            query_embedding: Query vector
            errors_dict: Dictionary mapping error_id to AERIErrorRecord
            top_k: Number of results
        
        Returns:
            List of SimilarityResult objects
        """
        search_results = self.search(query_embedding, top_k=top_k)
        
        similarity_results = []
        for rank, (error_id, similarity) in enumerate(search_results, start=1):
            if error_id in errors_dict:
                similarity_results.append(
                    SimilarityResult(
                        error=errors_dict[error_id],
                        similarity_score=similarity,
                        rank=rank
                    )
                )
        
        return similarity_results
    
    def save(self, filepath: Optional[Path] = None) -> Path:
        """
        Save vector store to disk.
        
        Args:
            filepath: Where to save (default: auto-generated in indexes_dir)
        
        Returns:
            Path where store was saved
        """
        if filepath is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = settings.INDEXES_DIR / f"faiss_index_{timestamp}"
        
        filepath = Path(filepath)
        filepath.mkdir(parents=True, exist_ok=True)
        
        logger.info("Saving vector store", filepath=str(filepath))
        
        try:
            # Save FAISS index
            index_path = filepath / "index.faiss"
            faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            metadata = {
                'error_ids': self.error_ids,
                'id_to_index': self.id_to_index,
                'errors_metadata': self.errors_metadata,
                'dimension': self.dimension,
                'index_type': self.index_type,
            }
            
            metadata_path = filepath / "metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(
                "Vector store saved successfully",
                filepath=str(filepath),
                total_vectors=len(self.error_ids)
            )
            
            return filepath
        
        except Exception as e:
            raise VectorStoreError(f"Failed to save vector store: {e}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'FAISSVectorStore':
        """
        Load vector store from disk.
        
        Args:
            filepath: Directory containing saved vector store
        
        Returns:
            Loaded FAISSVectorStore instance
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise VectorStoreError(f"Vector store not found: {filepath}")
        
        logger.info("Loading vector store", filepath=str(filepath))
        
        try:
            # Load metadata
            metadata_path = filepath / "metadata.pkl"
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # Create instance
            store = cls(
                dimension=metadata['dimension'],
                index_type=metadata['index_type']
            )
            
            # Load FAISS index
            index_path = filepath / "index.faiss"
            store.index = faiss.read_index(str(index_path))
            
            # Restore metadata
            store.error_ids = metadata['error_ids']
            store.id_to_index = metadata['id_to_index']
            store.errors_metadata = metadata['errors_metadata']
            
            logger.info(
                "Vector store loaded successfully",
                total_vectors=len(store.error_ids)
            )
            
            return store
        
        except Exception as e:
            raise VectorStoreError(f"Failed to load vector store: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'unique_error_ids': len(set(self.error_ids)),
            'has_metadata': len(self.errors_metadata) > 0,
        }