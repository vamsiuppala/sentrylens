# SentryLens

An agentic AI system for automated error triage. Uses semantic embeddings, clustering, and a Claude-powered ReAct agent to help developers understand, categorize, and resolve errors at scale.

## Features

- **Semantic Search**: Find similar errors using sentence-transformer embeddings and Hnswlib vector search
- **Automatic Clustering**: Group related errors using HDBSCAN density-based clustering
- **AI-Powered Triage**: Interactive ReAct agent powered by Claude for intelligent error analysis
- **Stack Trace Analysis**: Parse and analyze stack traces to identify root causes
- **Fix Suggestions**: Generate actionable fix recommendations based on error patterns
- **Web UI**: Chat interface for agent queries + error browser with cluster visualization
- **REST API**: FastAPI backend with endpoints for errors, clusters, and search
- **Sentry Integration**: Webhook endpoint to receive errors from Sentry with auto-clustering

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│  AERI JSON  │───▶│   Embedder   │───▶│   Hnswlib   │───▶│  Clusterer  │
│    Data     │    │ (MiniLM-L6)  │    │   Index     │    │  (HDBSCAN)  │
└─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘
                                              │                   │
                                              ▼                   ▼
┌─────────────┐                        ┌─────────────────────────────────┐
│   Sentry    │───webhook─────────────▶│         FastAPI Backend         │
│  (errors)   │                        ├─────────────────────────────────┤
└─────────────┘                        │       ReAct Agent (Claude)      │
                                       │  • search_similar_errors        │
      ┌────────────────────────────────│  • analyze_stack_trace          │
      │                                │  • suggest_fix                  │
      ▼                                └─────────────────────────────────┘
┌─────────────┐                                       │
│   Web UI    │◀──────────────────────────────────────┘
│ Chat/Browse │
└─────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/vamsiuppala/sentrylens.git
cd sentrylens

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...
```

## Quick Start

### Run the Full Pipeline

Process your error data end-to-end:

```bash
sentrylens pipeline -i data/aeri/output_problems -n 1000
```

This will:
1. **Ingest** JSON error files into normalized JSONL format
2. **Embed** errors using sentence-transformers and build an Hnswlib index
3. **Cluster** similar errors using HDBSCAN

### Start the Interactive Agent

After running the pipeline:

```bash
sentrylens agent data/indexes/hnswlib_index_* data/processed/clusters_*.json
```

Example queries:
- "Find errors similar to NullPointerException in UserService"
- "Help me understand error abc123"
- "What are the most common error patterns?"

## CLI Commands

### `sentrylens ingest`

Load and normalize error data:

```bash
sentrylens ingest -i <input_dir> [-n <sample_size>] [-o <output_file>]
```

| Option | Description |
|--------|-------------|
| `-i, --input` | Input directory containing JSON files |
| `-n, --sample-size` | Limit number of records to process |
| `-o, --output` | Output JSONL file path |

### `sentrylens embed`

Generate embeddings and create vector index:

```bash
sentrylens embed <input_file> [-m <model>] [-b <batch_size>] [--use-gpu]
```

| Option | Description |
|--------|-------------|
| `-m, --model` | Sentence-transformers model (default: all-MiniLM-L6-v2) |
| `-b, --batch-size` | Batch size for embedding (default: 32) |
| `--use-gpu` | Use GPU for embedding generation |

### `sentrylens cluster`

Run HDBSCAN clustering:

```bash
sentrylens cluster <embeddings_file> <errors_file> [--min-cluster-size <n>]
```

| Option | Description |
|--------|-------------|
| `--min-cluster-size` | Minimum cluster size (default: 5) |
| `--min-samples` | Minimum samples in neighborhood |
| `--metric` | Distance metric: euclidean, cosine, manhattan |

### `sentrylens agent`

Start the interactive triage agent:

```bash
sentrylens agent <vector_store> <cluster_data> [-m <model>] [--max-turns <n>]
```

| Option | Description |
|--------|-------------|
| `-m, --model` | Claude model (default: claude-3-5-haiku-20241022) |
| `--max-turns` | Maximum ReAct turns per query (default: 10) |
| `--example` | Run programmatic example instead of interactive mode |

### `sentrylens pipeline`

Run the complete pipeline:

```bash
sentrylens pipeline -i <input_dir> [-n <sample_size>] [--min-cluster-size <n>]
```

### `sentrylens serve`

Start the web server:

```bash
sentrylens serve <vector_store> <cluster_data> [--host <host>] [--port <port>]
```

| Option | Description |
|--------|-------------|
| `--host` | Host to bind (default: 0.0.0.0) |
| `--port` | Port to bind (default: 8000) |

## Web UI

After starting the server, open http://localhost:8000 in your browser.

### Chat Tab
- Query the AI agent in natural language
- Ask about error patterns, similar errors, or get fix suggestions
- Example: "What are the most common NullPointerException errors?"

### Browse Tab
- View cluster visualization as horizontal bar chart
- Click a cluster to filter errors by that cluster
- Click an error to see full details including stack trace
- Paginated error list with "Load more" support

## REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with loaded data stats |
| `/errors` | GET | List errors (supports `limit`, `offset`) |
| `/errors/{id}` | GET | Get error details by ID |
| `/clusters` | GET | List all clusters with sizes |
| `/clusters/{id}` | GET | Get errors in a specific cluster |
| `/errors/search` | POST | Semantic search for similar errors |
| `/query` | POST | Send a query to the triage agent |
| `/webhooks/sentry` | POST | Receive errors from Sentry |

### Sentry Webhook

Configure Sentry to send error events to `POST /webhooks/sentry`. The endpoint will:
1. Parse the Sentry event payload
2. Generate an embedding for the error
3. Find the nearest cluster and assign the error
4. Store the error for browsing and searching

## Project Structure

```
src/sentrylens/
├── agent/              # ReAct agent and tools
│   ├── tools.py        # search_similar, analyze_stack, suggest_fix
│   └── triage_agent.py # ReAct loop with Claude API
├── api/                # FastAPI backend
│   ├── main.py         # REST endpoints and webhook
│   └── static/         # Web UI (HTML/CSS/JS)
├── cli/                # Typer CLI commands
├── clustering/         # HDBSCAN clustering
├── core/               # Pydantic models and exceptions
├── data/               # Data loading utilities
├── embeddings/         # Embedder and Hnswlib vector store
└── utils/              # Logging configuration
```

## Data Format

SentryLens works with Eclipse AERI (Automated Error Reporting Initiative) data format:

```json
{
  "kind": "java.lang.NullPointerException",
  "summary": "Cannot invoke method on null object",
  "stacktraces": [[
    {"cN": "com.example.Service", "mN": "process", "fN": "Service.java", "lN": 42}
  ]],
  "javaRuntimeVersion": "17.0.1",
  "osgiOs": "linux"
}
```

## Agent Tools

The ReAct agent has access to three tools:

| Tool | Description |
|------|-------------|
| `search_similar_errors` | Find semantically similar errors using vector search |
| `analyze_stack_trace` | Parse stack trace to extract structured information |
| `suggest_fix` | Generate fix recommendations based on error context and patterns |

## Configuration

Configuration is managed via environment variables and `src/sentrylens/config.py`:

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Claude API key | Required |
| `AERI_DATA_DIR` | Default input data directory | `data/aeri/output_problems` |
| `EMBEDDING_DIMENSION` | Embedding vector size | 384 |
| `SAMPLE_SIZE` | Default sample size | 1000 |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=sentrylens
```

## Dependencies

- **anthropic** - Claude API client
- **sentence-transformers** - Text embeddings
- **hnswlib** - Approximate nearest neighbor search
- **hdbscan** - Density-based clustering
- **pydantic** - Data validation
- **typer** - CLI framework
- **fastapi** - REST API framework
- **uvicorn** - ASGI server

## Acknowledgments

- [Eclipse AERI](https://www.eclipse.org/epp/error-reporting/) for the error dataset
- [Anthropic](https://www.anthropic.com/) for Claude API
