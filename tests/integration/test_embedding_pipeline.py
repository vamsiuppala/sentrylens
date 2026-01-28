"""
Integration test for complete embedding pipeline.
"""
import pytest
from pathlib import Path
import tempfile

from sentrylens.data.loader import AERIDataLoader
from sentrylens.embeddings.embedder import ErrorEmbedder
from sentrylens.embeddings.vector_store import HnswlibVectorStore
from sentrylens.core.models import ProcessedDataset


@pytest.mark.integration
class TestEmbeddingPipeline:
    """Integration tests for full embedding pipeline."""

    def test_end_to_end_pipeline(self):
        """Test complete pipeline from data loading to vector search."""

        # 1. Create sample data
        from sentrylens.core.models import AERIErrorRecord

        errors = [
            AERIErrorRecord(
                error_id=f"integration-test-{i}",
                error_type="NullPointerException" if i % 2 == 0 else "IndexOutOfBoundsException",
                error_message=f"Test error {i}",
                stack_trace=f"at com.test.Class.method(Class.java:{i*10})\nat com.test.Main.main(Main.java:1)"
            )
            for i in range(20)
        ]

        # 2. Generate embeddings
        embedder = ErrorEmbedder(device="cpu", batch_size=8)
        embeddings = embedder.embed_batch(errors, show_progress=False)

        assert len(embeddings) == 20

        # 3. Create vector store
        store = HnswlibVectorStore()
        store.add_embeddings(embeddings, errors)

        assert len(store.error_ids) == 20

        # 4. Test search functionality
        # Errors with same type should be similar
        results = store.search_by_error_id("integration-test-0", top_k=5, exclude_self=True)

        # Get error types of similar errors
        similar_error_ids = [eid for eid, _ in results]
        similar_errors = [e for e in errors if e.error_id in similar_error_ids]

        # At least some should be NullPointerException (same as query)
        same_type_count = sum(1 for e in similar_errors if e.error_type == "NullPointerException")
        assert same_type_count > 0, "Similar errors should include same error type"

        # 5. Test persistence
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_store"
            store.save(save_path)

            loaded_store = HnswlibVectorStore.load(save_path)

            # Search on loaded store should give same results
            loaded_results = loaded_store.search_by_error_id(
                "integration-test-0",
                top_k=5,
                exclude_self=True
            )

            assert len(loaded_results) == len(results)
