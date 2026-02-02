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

**Next Steps:**
1. **Simple Web UI** (optional) - Basic interface for the agent
   - Chat interface for agent queries
   - Error browser with cluster visualization

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
| `src/sentrylens/embeddings/vector_store.py` | Hnswlib vector store operations |
| `src/sentrylens/clustering/clusterer.py` | HDBSCAN clustering |
| `scripts/demo_agent.py` | Interactive agent demo |

## Dependencies

- `anthropic` - Claude API
- `sentence-transformers` - Embeddings
- `hnswlib` - Vector search
- `hdbscan` - Clustering
- `pydantic` - Data models
