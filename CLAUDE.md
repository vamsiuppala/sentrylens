# CLAUDE.md - SentryLens Project Context

> **Purpose**: This file provides context for Claude Code (or any AI assistant) to understand the SentryLens project, what's been accomplished, architectural decisions, and how to continue development effectively.

---

## 🎯 Project Goal

**SentryLens** is a portfolio project demonstrating qualifications for the **Staff Machine Learning Engineer, AI** position at Sentry. The project showcases:

- Production-grade agentic AI systems for error triage
- Embedding models and vector search for code/errors
- ReAct pattern implementation with multi-step reasoning
- End-to-end ML system design (not just notebooks)

### Target Job Requirements (Sentry Staff ML Engineer)

| Requirement | How SentryLens Demonstrates It |
|-------------|-------------------------------|
| Build state-of-the-art agentic AI systems to triage, debug, and solve production issues | ReAct agent with tools for error analysis, similarity search, fix suggestions |
| Leverage massive datasets of errors, spans, and profiles | Eclipse AERI dataset (100k+ Java exceptions) |
| Production-grade agentic systems and tools | Modular architecture, proper error handling, logging, tests |
| Python + PyTorch expertise | Core implementation in Python, embeddings with sentence-transformers |
| Deploy ML models at scale | FAISS vector store, batched inference, async API |
| Technical documentation | Comprehensive README, architecture diagrams, blog post |

---

## 📊 Current State

### ✅ Completed (Steps 1-2)

**Step 1: Setup & Data Exploration**
- [x] Project structure created
- [x] Eclipse AERI dataset downloaded and explored (`explore_data.ipynb`)
- [x] Stack trace parsing implemented (`parse_stacktrace.py`, `explore_stacktrace.py`)
- [x] Data schemas defined in `src/sentrylens/core/models.py`
- [x] Download scripts (`download_aeri.sh`, `download_aeri_json.sh`)

**Step 2: Embeddings Infrastructure**
- [x] Data loader implemented (`src/sentrylens/data/loader.py`)
- [x] Embedder class with sentence-transformers (`src/sentrylens/embeddings/embedder.py`)
- [x] Vector store with FAISS (`src/sentrylens/embeddings/vector_store.py`)
- [x] Logging infrastructure (`src/sentrylens/utils/logger.py`)
- [x] Unit tests with coverage (see `htmlcov/`)
- [x] Scripts: `scripts/ingest_data.py`, `scripts/generate_embeddings.py`

**Step 3: Clustering Infrastructure**
- [x] HDBSCAN clusterer (`src/sentrylens/clustering/clusterer.py`)
- [x] Cluster assignment model and statistics
- [x] Unit tests with 16 test cases
- [x] Script: `scripts/cluster_errors.py`
- [x] Documentation: CLUSTERING_DESIGN.md, SYSTEM_ARCHITECTURE.md

### 🔄 In Progress (Step 3.5: Vertical Slice Complete)

End-to-end pipeline fully implemented: data → embeddings → clustering.

**Architecture**:
```
Error Report → Embedding → Vector Search → Clustering → ReAct Agent → Analysis + Fix Suggestions
```

### ⏳ Remaining Steps

| Step | Description | Status |
|------|-------------|--------|
| Step 4 | ReAct agent with tools (analyze_stack, search_similar, suggest_fix) | Not started |
| Step 5 | CLI demo showing full flow | Not started |
| Step 6 | FastAPI backend + simple frontend | Not started |
| Step 7 | Polish: README, CI/CD, blog post | Not started |

---

## 🏗️ Architecture Decisions

### Why Vertical Slice First?
- Validates end-to-end architecture before investing in fine-tuning
- Identifies what embeddings actually need to capture
- Creates working demo quickly for recruiter conversations
- Interview talking point: "I built a vertical slice first to validate assumptions"

### Embedding Strategy
- **Current**: Pre-trained `sentence-transformers` (e.g., `all-mpnet-base-v2` or `microsoft/codebert-base`)
- **Future**: Fine-tune on AERI data with contrastive learning if baseline performance insufficient

### Agent Pattern
- **Pattern**: ReAct (Reasoning + Acting)
- **Framework**: LangChain with `langchain-anthropic`
- **LLM**: Claude Sonnet for agent reasoning
- **Tools**:
  1. `search_similar_errors` - Vector similarity search
  2. `analyze_stack_trace` - Parse stack traces for structured info
  3. `suggest_fix` - Generate fix recommendations based on error + similar cases

### Data Flow
```
AERI JSON files
    ↓
DataLoader (src/sentrylens/data/loader.py)
    ↓
ErrorReport models (src/sentrylens/core/models.py)
    ↓
Embedder (src/sentrylens/embeddings/embedder.py)
    ↓
VectorStore/FAISS (src/sentrylens/embeddings/vector_store.py)
    ↓
ReAct Agent (src/sentrylens/agent/) [TO BUILD]
    ↓
API/CLI (TO BUILD)
```

---

## 📁 Project Structure

```
sentrylens/
├── configs/
│   └── settings.py              # Configuration management
├── data/
│   ├── aeri/                    # Raw AERI JSON files
│   ├── embeddings/              # Generated embeddings (.npy)
│   ├── indexes/                 # FAISS indexes
│   ├── processed/               # Processed data
│   └── swebench/                # SWE-bench data (optional)
├── logs/                        # Application logs
├── scripts/
│   ├── ingest_data.py           # Load and process AERI data
│   └── generate_embeddings.py   # Generate and save embeddings
├── src/sentrylens/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── exceptions.py        # Custom exceptions
│   │   └── models.py            # Pydantic models (ErrorReport, etc.)
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py            # AERI data loading
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── embedder.py          # Sentence transformer wrapper
│   │   └── vector_store.py      # FAISS vector store
│   ├── clustering/              # [TO CREATE] HDBSCAN clustering
│   ├── agent/                   # [TO CREATE] ReAct agent + tools
│   ├── api/                     # [TO CREATE] FastAPI backend
│   └── utils/
│       ├── __init__.py
│       └── logger.py            # Logging configuration
├── tests/
│   ├── unit/                    # Unit tests
│   └── integration/             # Integration tests
├── requirements.txt             # Production dependencies
├── requirements-dev.txt         # Dev dependencies (pytest, etc.)
└── README.md                    # [TO ENHANCE]
```

---

## 🛠️ Development Conventions

### Python Style
- Python 3.10+
- Type hints required for all functions
- Pydantic for data models
- Docstrings for public functions

### Dependencies
```bash
# Core
sentence-transformers
faiss-cpu
langchain
langchain-anthropic
pydantic

# API (when building)
fastapi
uvicorn

# Dev
pytest
pytest-cov
```

### Testing
```bash
# Run tests with coverage
pytest tests/ --cov=src/sentrylens --cov-report=html

# Run specific test file
pytest tests/unit/test_embedder.py -v
```

### Logging
- Use the logger from `src/sentrylens/utils/logger.py`
- Logs go to `logs/sentrylens_<timestamp>.log`
- Import: `from sentrylens.utils.logger import get_logger`

### Environment Variables
```bash
ANTHROPIC_API_KEY=sk-ant-...  # For ReAct agent
SENTRYLENS_LOG_LEVEL=INFO     # Optional
```

---

## 🚀 Next Steps (For Claude Code)

### Immediate: Complete Step 4 - ReAct Agent

Create `src/sentrylens/agent/`:
```
agent/
├── __init__.py
├── tools.py          # search_similar, analyze_stack, suggest_fix
├── prompts.py        # ReAct prompt templates
└── triage_agent.py   # Main agent executor
```

**Tools to implement**:
1. `search_similar_errors`: Use FAISS to find similar errors
2. `analyze_stack_trace`: Parse stack trace for structure
3. `suggest_fix`: Use LLM to recommend fixes

### Then: Step 5 - CLI Demo

Create `sentrylens/cli.py` using Click:
```bash
# Target usage:
sentrylens analyze "NullPointerException at com.example.Service.process(Service.java:42)"
sentrylens cluster --show-stats
sentrylens similar --error-id 12345 --k 5
```

### Step 3 ✅ COMPLETED

See CLUSTERING_DESIGN.md and SYSTEM_ARCHITECTURE.md for detailed documentation.

---

## 💡 Prompting Tips for Claude Code

### When implementing new features:
```
@workspace Implement [feature] in src/sentrylens/[module]/
Follow the existing patterns in the codebase:
- Use Pydantic models from core/models.py
- Use logger from utils/logger.py  
- Add type hints
- Include docstrings
- Add unit tests in tests/unit/
```

### When debugging:
```
@workspace I'm getting [error] when running [command].
Check the logs in logs/ and help me fix it.
```

### When adding tests:
```
@workspace Add unit tests for src/sentrylens/[module]/[file].py
Follow the existing test patterns in tests/unit/
```

---

## 📝 Interview Talking Points

Use these when discussing the project:

1. **Vertical Slice Approach**: "I built a working end-to-end system first to validate architecture decisions before investing in fine-tuning."

2. **Production Mindset**: "The codebase has proper error handling, logging, configuration management, and test coverage—not just a Jupyter notebook."

3. **Agentic Design**: "The ReAct agent uses tool-augmented reasoning: it searches similar errors, analyzes stack traces, and suggests fixes through multi-step deliberation."

4. **Scalability Considerations**: "FAISS enables sub-linear similarity search, and the architecture supports batched inference and async API calls."

5. **Data Understanding**: "I explored the AERI dataset to understand error patterns, which informed my embedding strategy and clustering approach."

---

## 🔗 Resources

- [Eclipse AERI Dataset](https://eclipse.org/recommenders/)
- [Sentry Job Posting](https://sentry.io/careers/81f09568-da7d-4ed1-8283-614f846c9b00/)
- [LangChain ReAct Agents](https://python.langchain.com/docs/modules/agents/)
- [FAISS Documentation](https://faiss.ai/)
- [HDBSCAN Documentation](https://hdbscan.readthedocs.io/)

---

## 📅 Timeline

| Phase | Target | Description |
|-------|--------|-------------|
| Week 1 | ✅ Done | Data exploration, embeddings infrastructure |
| Week 2 | Current | Vertical slice: clustering + agent |
| Week 3 | Upcoming | CLI demo + FastAPI backend |
| Week 4 | Upcoming | Frontend, polish, blog post |

---

*Last updated: January 2026*