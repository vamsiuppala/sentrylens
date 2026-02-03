---
layout: post
title: "Building SentryLens Part 1: Data Pipeline and Semantic Embeddings"
date: 2026-02-02
categories: [ml-engineering, embeddings, error-triage]
---

# Building SentryLens Part 1: Data Pipeline and Semantic Embeddings

**TL;DR**: Built a production-grade data pipeline that transforms raw Eclipse error reports into semantic embeddings, enabling fast similarity search with Hnswlib. The foundation for intelligent error triage.

## The Problem

Error logs are noisy. When building an AI-powered triage system, we faced three immediate challenges:

1. **Inconsistent data formats** - Eclipse AERI error reports have deeply nested structures with optional fields
2. **Scale** - Need to process thousands of errors efficiently without excessive memory usage
3. **Searchability** - Raw text search isn't enough; we need semantic understanding to find similar errors

We needed a pipeline that could validate, normalize, and embed error data while maintaining type safety and performance.

## The Approach

Our pipeline follows three phases:
1. **Load and validate** error records using Pydantic schemas
2. **Generate embeddings** using sentence-transformers
3. **Build a vector index** with Hnswlib for fast similarity search

The key insight: treat this as a contract-first problem. Define strict schemas upfront, fail fast on validation errors, and leverage type safety throughout.

## Implementation

### Phase 1: Schema-First Data Loading

We start with Pydantic models that capture the essence of an error:

```python
class AERIErrorRecord(BaseModel):
    # Required fields
    error_id: str
    error_type: str  # Exception class
    error_message: str
    stack_trace: str

    # Optional metadata
    timestamp: Optional[datetime] = None
    plugin_id: Optional[str] = None
    severity: SeverityLevel = SeverityLevel.UNKNOWN

    @field_validator("stack_trace")
    @classmethod
    def validate_stack_trace(cls, v: str) -> str:
        if not v or len(v.strip()) == 0:
            raise ValueError("Stack trace cannot be empty")
        if len(v) > 50000:
            raise ValueError("Stack trace exceeds maximum length")
        return v.strip()
```

This model enforces invariants at the boundary. Invalid data is rejected immediately, not propagated through the system.

The loader implements graceful degradation:

```python
def load_from_json_file(self, file_path: Path) -> List[AERIErrorRecord]:
    validated_records = []
    validation_errors = []

    for idx, record in enumerate(raw_records):
        try:
            normalized = self._normalize_record(record)
            validated = AERIErrorRecord(**normalized)
            validated_records.append(validated)
        except ValidationError as e:
            validation_errors.append((idx, record, str(e)))

    # Fail if > 10% validation errors
    if len(validation_errors) > len(raw_records) * 0.1:
        raise DataValidationError("Too many validation errors")

    return validated_records
```

This 10% threshold is critical. We accept that real-world data is messy, but we don't proceed if something is fundamentally broken.

### Phase 2: Semantic Embeddings

Raw error text isn't directly comparable. We need dense vector representations that capture semantic meaning. Enter sentence-transformers:

```python
class ErrorEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def prepare_text(self, error: AERIErrorRecord) -> str:
        # Extract key information
        stack_lines = error.stack_trace.split('\n')
        meaningful_frames = [
            line.strip()
            for line in stack_lines
            if line.strip() and not line.strip().startswith('...')
        ][:10]  # First 10 frames are most relevant

        stack_summary = '\n'.join(meaningful_frames)

        return f"""Error Type: {error.error_type}
Message: {error.error_message}
Stack Trace:
{stack_summary}"""
```

The `prepare_text` method is where domain knowledge lives. We don't embed the entire stack trace - just the first 10 meaningful frames. This reduces noise and focuses on the actual error location.

Batching is essential for performance:

```python
def embed_batch(self, errors: List[AERIErrorRecord]) -> List[ErrorEmbedding]:
    texts = [self.prepare_text(error) for error in errors]

    # Generate embeddings with batching and progress tracking
    embedding_arrays = self.model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=True,
        device=self.device
    )

    return [
        ErrorEmbedding(
            error_id=error.error_id,
            embedding=array.tolist(),
            model_name=self.model_name
        )
        for error, array in zip(errors, embedding_arrays)
    ]
```

On a modern CPU, this processes ~100 errors per second. With GPU acceleration, we hit 500+.

### Phase 3: Fast Similarity Search with Hnswlib

Embeddings are useless without fast search. We use Hnswlib, which implements Hierarchical Navigable Small World graphs for approximate nearest neighbor search:

```python
class HnswlibVectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

        # Initialize HNSW index with cosine similarity
        self.index = hnswlib.Index(space='cosine', dim=dimension)
        self.index.init_index(
            max_elements=100000,
            ef_construction=200,  # Quality vs speed tradeoff
            M=16  # Number of bi-directional links per node
        )

        # Map index positions to error IDs
        self.error_ids: List[str] = []
        self.id_to_index: Dict[str, int] = {}
```

The parameters matter:
- `ef_construction=200` - Higher values = better index quality but slower build
- `M=16` - More links = better recall but more memory

For our use case (errors, not images), these defaults work well.

Adding vectors is straightforward:

```python
def add_embeddings(self, embeddings: List[ErrorEmbedding]) -> None:
    embedding_matrix = np.array(
        [emb.embedding for emb in embeddings],
        dtype=np.float32
    )

    start_idx = len(self.error_ids)
    ids = np.arange(start_idx, start_idx + len(embeddings), dtype=np.int32)

    self.index.add_items(embedding_matrix, ids)

    # Update metadata
    for embedding in embeddings:
        self.error_ids.append(embedding.error_id)
        self.id_to_index[embedding.error_id] = start_idx
        start_idx += 1
```

Search returns the most similar errors in milliseconds:

```python
def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
    query_array = np.array([query_embedding], dtype=np.float32)

    # Returns (indices, distances) in cosine space
    indices, distances = self.index.knn_query(query_array, k=top_k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        error_id = self.error_ids[int(idx)]
        similarity = 1.0 - float(dist)  # Convert distance to similarity
        results.append((error_id, similarity))

    return results
```

On a dataset of 10,000 errors, search takes ~2ms. This is fast enough for interactive use.

## Challenges and Solutions

### Challenge 1: Nested AERI Format

Eclipse AERI data has stacktraces as nested arrays of frame objects. We need to flatten this into readable text:

```python
def _normalize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
    # Convert stacktraces array to string
    if 'stack_trace' in normalized and isinstance(normalized['stack_trace'], list):
        stacktraces = normalized['stack_trace']
        formatted_traces = []

        for i, trace in enumerate(stacktraces, 1):
            if isinstance(trace, list):
                trace_str = '\n'.join(
                    f"  at {frame.get('cN')}.{frame.get('mN')} "
                    f"({frame.get('fN')}:{frame.get('lN')})"
                    for frame in trace if isinstance(frame, dict)
                )
                formatted_traces.append(f"Stack trace {i}:\n{trace_str}")

        normalized['stack_trace'] = '\n\n'.join(formatted_traces)
```

This preserves Java stack trace formatting while handling the AERI structure.

### Challenge 2: Missing Error IDs

Some records lack unique identifiers. We generate deterministic IDs:

```python
if 'error_id' not in normalized:
    content = f"{normalized.get('error_type', '')}:{normalized.get('error_message', '')}"
    normalized['error_id'] = hashlib.md5(content.encode()).hexdigest()
```

Using MD5 of error type and message ensures the same error gets the same ID.

### Challenge 3: Embedding Dimension Mismatches

Different sentence-transformer models have different dimensions. We validate eagerly:

```python
@field_validator("embedding")
@classmethod
def validate_embedding_dimension(cls, v: List[float]) -> List[float]:
    from sentrylens.config import settings
    if len(v) != settings.EMBEDDING_DIMENSION:
        raise ValueError(
            f"Embedding dimension mismatch: expected {settings.EMBEDDING_DIMENSION}, "
            f"got {len(v)}"
        )
    return v
```

This catches issues at validation time, not during search.

## Results

The complete pipeline processes 1,000 errors in under 30 seconds:
- **Loading**: 1-2 seconds (with validation)
- **Embedding**: 20-25 seconds (CPU, batch_size=32)
- **Index building**: 2-3 seconds

We achieve:
- 99.8% validation success rate on AERI data
- Sub-3ms search latency for top-5 results
- 384-dimensional embeddings (all-MiniLM-L6-v2)

The index persists to disk efficiently:

```python
def save(self, filepath: Path) -> Path:
    # Save HNSW index
    self.index.save_index(str(filepath / "index.hnsw"))

    # Save metadata separately
    metadata = {
        'error_ids': self.error_ids,
        'id_to_index': self.id_to_index,
        'dimension': self.dimension
    }
    with open(filepath / "metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
```

A 10K error index is ~15MB on disk and loads in <1 second.

## What's Next

With data ingestion and embeddings working, we have a searchable knowledge base of errors. In Part 2, we'll add HDBSCAN clustering to automatically group similar errors and build a ReAct agent with Claude to reason about error patterns and suggest fixes.

**Key Takeaways:**
- Start with schemas and validation - fail fast on bad data
- Batch operations for performance (32-64 works well for embeddings)
- Choose the right index structure (HNSW for speed, exact search for accuracy)
- Preserve domain semantics when converting text (stack trace format matters)

---
