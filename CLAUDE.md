# SentryLens

Agentic AI system for error triage. Learning project for building production-grade ML pipelines with ReAct agents.

## Status

**Completed:**
- Data loading (Eclipse AERI dataset)
- Embeddings (sentence-transformers + FAISS)
- Clustering (HDBSCAN)
- ReAct agent (Claude API with native tool_use)

**TODO:**
- CLI interface
- FastAPI backend

## Architecture

```
AERI JSON → DataLoader → Embedder → FAISS → Clusterer → ReAct Agent
```

## Structure

See `structure.txt` for full layout.

```
src/sentrylens/
├── agent/          # ReAct agent + tools (Claude API)
├── clustering/     # HDBSCAN clustering
├── core/           # Pydantic models
├── data/           # Data loading
├── embeddings/     # Embedder + FAISS vector store
└── utils/          # Logger
```

## Usage

```bash
# Setup
export ANTHROPIC_API_KEY=sk-ant-...
pip install -r requirements.txt

# Run agent demo
python scripts/demo_agent.py
```

## Key Files

| File | Purpose |
|------|---------|
| `src/sentrylens/agent/triage_agent.py` | ReAct loop with Claude |
| `src/sentrylens/agent/tools.py` | search_similar, analyze_stack, suggest_fix |
| `src/sentrylens/embeddings/vector_store.py` | FAISS operations |
| `src/sentrylens/clustering/clusterer.py` | HDBSCAN clustering |
| `scripts/demo_agent.py` | Interactive agent demo |

## Dependencies

- `anthropic` - Claude API
- `sentence-transformers` - Embeddings
- `faiss-cpu` - Vector search
- `hdbscan` - Clustering
- `pydantic` - Data models
