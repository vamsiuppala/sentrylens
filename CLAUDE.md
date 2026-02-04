# SentryLens

Agentic AI system for error triage. Learning project for building production-grade ML pipelines with ReAct agents.

## Status

**Completed:**
- Data loading (Eclipse AERI dataset)
- Embeddings (sentence-transformers + Hnswlib)
- Clustering (HDBSCAN)
- ReAct agent (Claude API with native tool_use)
- CLI interface (Typer)
- FastAPI Backend (REST API)
- Sentry Webhook (`POST /webhooks/sentry` with auto-embed and cluster assignment)
- Web UI (chat interface + error browser with cluster visualization)

## Architecture

```
AERI JSON → DataLoader → Embedder → Hnswlib → Clusterer → ReAct Agent
                                                              ↑
                                                         FastAPI ← Web UI
                                                              ↑
                                                         Webhooks (Sentry)
```

## Structure

See `structure.txt` for full layout.

```
src/sentrylens/
├── agent/          # ReAct agent + tools (Claude API)
├── api/            # FastAPI backend + static web UI
├── cli/            # Typer CLI commands
├── clustering/     # HDBSCAN clustering
├── core/           # Pydantic models
├── data/           # Data loading
├── embeddings/     # Embedder + Hnswlib vector store
└── utils/          # Logger
```

## Usage

```bash
# Setup
export ANTHROPIC_API_KEY=sk-ant-...
pip install -e .

# Run full pipeline
sentrylens pipeline -i data/aeri/output_problems -n 1000

# Start interactive agent
sentrylens agent data/indexes/hnswlib_index_* data/processed/clusters_*.json
```

## Key Files

| File | Purpose |
|------|---------|
| `src/sentrylens/agent/triage_agent.py` | ReAct loop with Claude |
| `src/sentrylens/agent/tools.py` | search_similar, analyze_stack, suggest_fix |
| `src/sentrylens/api/main.py` | FastAPI REST endpoints + webhook |
| `src/sentrylens/api/static/index.html` | Web UI (chat + error browser) |
| `src/sentrylens/embeddings/vector_store.py` | Hnswlib vector store operations |
| `src/sentrylens/clustering/clusterer.py` | HDBSCAN clustering |

## Dependencies

- `anthropic` - Claude API
- `sentence-transformers` - Embeddings
- `hnswlib` - Vector search
- `hdbscan` - Clustering
- `pydantic` - Data models

## Planned: Live Triage Tools

**Status: Not started**

New tools for `src/sentrylens/agent/tools.py` to help diagnose incoming errors against the existing cluster knowledge base.

### Essential Tools

1. **`get_cluster_context(cluster_id: int)`**
   - When error is assigned to a cluster, retrieve cluster label, size, common patterns
   - Return frequent error types in cluster, typical stack trace patterns, and recommendations
   - Use `clusters_dict` to find all errors in cluster, analyze `error_type` distribution

2. **`find_best_known_fix(error_id: str)`**
   - Search cluster history for similar resolved errors
   - Return known fixes with confidence score based on cluster membership
   - Leverage similarity search + cluster context

3. **`assess_error_prevalence(error_id: str)`**
   - Check if error is part of large cluster (systemic issue) or isolated/noise (one-off)
   - Return prevalence score, cluster size, noise flag
   - Flag potential regression indicators (sudden cluster growth)

4. **`get_root_component(cluster_id: int)`**
   - Analyze cluster's stack traces to find common methods/classes
   - Filter out library code (java.*, javax.*, org.eclipse.*, etc.)
   - Return top components with frequency counts

### Nice-to-Have

5. **`compare_against_cluster(error_id: str, cluster_id: int)`**
   - Compare new error to cluster's typical profile
   - Flag if error is an outlier within its cluster
   - Return similarity score to cluster centroid

### Implementation Notes

- Each tool returns JSON string (consistent with existing tools)
- Add tool schemas to `get_tool_schemas()` method
- Use existing infrastructure: `errors_dict`, `clusters_dict`, `vector_store`, `embedder`
- Include logging and error handling following existing patterns
