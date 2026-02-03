# SentryLens Blog Posts

Technical blog posts documenting the development of SentryLens, an agentic AI system for error triage.

## Series Overview

This three-part series covers the complete journey of building a production-grade ML pipeline that combines semantic embeddings, clustering, and LLM agents to help developers triage errors at scale.

### Part 1: Data Pipeline and Semantic Embeddings
**File**: `2026-02-02-data-pipeline-embeddings.md`

**Topics Covered:**
- Loading and validating Eclipse AERI error data with Pydantic
- Generating semantic embeddings with sentence-transformers
- Building a fast vector index with Hnswlib for similarity search
- Handling nested JSON formats and data normalization

**Key Takeaways:**
- Contract-first approach with Pydantic schemas
- Batch processing for performance (32-64 batch size)
- HNSW graphs for sub-3ms search latency
- Domain-aware text preparation (first 10 stack frames)

**Metrics:**
- 1,000 errors processed in ~30 seconds
- 99.8% validation success rate
- Sub-3ms search for top-5 similar errors

---

### Part 2: Clustering and the ReAct Agent
**File**: `2026-02-02-clustering-react-agent.md`

**Topics Covered:**
- HDBSCAN density-based clustering for automatic error grouping
- Building a ReAct agent with Claude's native tool_use
- Implementing three tools: search, analyze, suggest
- Combining cluster context with LLM reasoning

**Key Takeaways:**
- HDBSCAN automatically detects cluster count
- ReAct pattern for agentic reasoning
- Tool responses should be concise and structured
- Cluster size indicates fix priority

**Metrics:**
- 1,000 errors → 32 clusters + 8% noise
- Average query: 2-3 tool calls
- 92% success rate on user queries
- 3-5 second response time

---

### Part 3: FastAPI Backend and Web UI
**File**: `2026-02-02-api-web-ui.md`

**Topics Covered:**
- FastAPI REST endpoints for errors, clusters, and agent queries
- Single-page web UI with chat and browse modes
- Sentry webhook integration with auto-clustering
- Cluster visualization with CSS-only bar charts

**Key Takeaways:**
- FastAPI for ML system deployment
- Vanilla JavaScript for simple UIs
- Webhook patterns for real-time inference
- Nearest-neighbor for cluster assignment

**Metrics:**
- API latency: 5-10ms for CRUD, 3-5s for agent
- Webhook processing: 100-150ms per error
- 85% clustering accuracy on new errors
- UI load time: <500ms

---

## Project Links

- **Repository**: [github.com/vamsiuppala/sentrylens](https://github.com/vamsiuppala/sentrylens)
- **Documentation**: See main README.md in project root
- **Live Demo**: (Add deployment URL here if available)

## Stack

- **ML/AI**: sentence-transformers, Hnswlib, HDBSCAN, Claude API
- **Backend**: FastAPI, Uvicorn, Pydantic
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **Data**: Eclipse AERI dataset, Sentry webhooks

## Quick Start

```bash
# Clone and setup
git clone https://github.com/vamsiuppala/sentrylens.git
cd sentrylens
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...

# Run full pipeline
sentrylens pipeline -i data/aeri/output_problems -n 1000

# Start web server
sentrylens serve data/indexes/hnswlib_index_* data/processed/clusters_*.json

# Open http://localhost:8000
```
