# Step 4: ReAct Agent for Error Triage

## Overview

This document describes the **ReAct (Reasoning + Acting) Agent** implementation for SentryLens, which provides intelligent error triage and fix recommendations using a local Ollama LLM.

## Architecture

### Integration with Previous Steps

The agent integrates all infrastructure from Steps 1-3:

```
Step 1: Error Data (500+ Java exceptions)
   ↓
Step 2: Embeddings & Vector Store (384-dim vectors, FAISS)
   ↓
Step 3: Clustering (HDBSCAN, ~10 clusters)
   ↓
Step 4: ReAct Agent (Ollama LLM + Tool Calling)
   ↓
User → Interactive Chat Interface
```

### Component Structure

**Step 4 Files:**
- `src/sentrylens/agent/tools.py` - Three tools for error analysis
- `src/sentrylens/agent/prompts.py` - System prompts for Ollama
- `src/sentrylens/agent/triage_agent.py` - Main agent class with ReAct loop
- `tests/unit/test_tools.py` - Tool unit tests
- `tests/unit/test_agent.py` - Agent unit tests
- `scripts/demo_agent.py` - Interactive demo script

## Key Design: Ollama Instead of Claude API

The original plan mentioned using Claude API, but for a **portfolio project**, using **Ollama** is better because:

✓ **Free & Private** - No API costs, no data sent to cloud
✓ **Shows ML Deployment Skills** - Local LLM integration is production-relevant
✓ **Reproducible** - Anyone can run it without API keys
✓ **Educational** - Demonstrates open-source LLM integration patterns

### How Ollama Tool Calling Works

Unlike Claude's native `tool_use` feature, Ollama doesn't support structured tool calls. Instead:

1. **System Prompt** guides the model to output actions in a specific format
2. **Regex Parsing** extracts tool calls from text responses
3. **Tool Execution** runs the appropriate function
4. **Observation Format** returns results back to the model

**Tool Call Format (Ollama):**
```
THOUGHT: I need to find similar errors to understand this pattern
ACTION: search_similar_errors(query_text="NullPointerException", top_k=5)
```

**Parsing with Regex:**
```python
action_pattern = r'ACTION:\s*(\w+)\s*\((.*?)\)\s*(?:\n|$)'
# Extracts: tool_name="search_similar_errors", args="query_text=..., top_k=..."
```

## The Three Tools

### 1. search_similar_errors

**Purpose:** Find semantically similar errors using vector similarity

**Integration with Step 2:**
- Uses `ErrorEmbedder` to embed query text
- Uses `FAISSVectorStore.search()` for similarity search
- Returns top-k similar errors with scores (0-1)

**Input:** `query_text` (error type, message, or partial stack trace), `top_k` (optional, default 5)

**Output:** JSON with similar errors including:
- Error ID, type, message
- Similarity score (0-1)
- Cluster ID and cluster size
- Whether it's a noise point

**Example:**
```
ACTION: search_similar_errors(query_text="NullPointerException", top_k=5)
```

### 2. analyze_stack_trace

**Purpose:** Parse stack traces into structured information

**Implementation:**
- Regex extraction of exception type and frames
- Identification of root cause (first non-library frame)
- Counts frames and extracts key methods/files/line numbers

**Input:** `stack_trace` (full Java stack trace text)

**Output:** JSON with:
- Exception type
- Total frame count
- Root cause frame (method, file, line number)
- Top 5 frames for context
- Stack depth

**Example:**
```
ACTION: analyze_stack_trace(stack_trace="at com.example.Service.process(Service.java:42)\n...")
```

### 3. suggest_fix

**Purpose:** Generate fix recommendations based on full error context

**Integration with All Steps:**
- **Step 1:** Gets full error details and stack trace
- **Step 2:** Finds similar errors using vector search
- **Step 3:** Gets cluster context and pattern frequency

**Input:** `error_id` (MD5 hash of error)

**Output:** JSON with:
- Error type and message
- Root cause analysis
- Cluster context (size, is_common_pattern)
- Textual fix recommendation

**Rules Applied:**
- NullPointerException → Add null checks
- FileNotFoundException → Check file existence
- OutOfMemoryError → Find memory leaks
- Common patterns (cluster_size > 10) → Prioritize

**Example:**
```
ACTION: suggest_fix(error_id="3d0af1bba06c8bc40f9eb7c7a56da2c5")
```

## Setup & Usage

### Prerequisites

1. **Ollama Installation**
   ```bash
   # macOS/Linux
   curl -fsSL https://ollama.ai/install.sh | sh

   # Or download from https://ollama.ai/download
   ```

2. **Pull a Model**
   ```bash
   # Recommended: Llama 3.1 8B (good balance of speed/quality)
   ollama pull llama3.1:8b

   # Alternative options:
   ollama pull codellama:13b      # Better for code
   ollama pull mistral:7b         # Faster
   ```

3. **Start Ollama Server**
   ```bash
   ollama serve
   # Runs on http://localhost:11434
   ```

### Interactive Mode

```bash
python scripts/demo_agent.py
```

You'll see:
```
============================================================
SentryLens Error Triage Agent - Interactive Mode
============================================================

Model: llama3.1:8b (via Ollama at http://localhost:11434)
Knowledge base: 500 errors
Max reasoning turns: 10
Type 'help' for examples or 'exit' to quit

You: Help me fix error 3d0af1bba06c8bc40f9eb7c7a56da2c5

Agent thinking...

Agent: [Multi-turn reasoning with tool calls]
```

### Programmatic Usage

```python
from src.sentrylens.agent import TriageAgent

# Initialize agent
agent = TriageAgent(
    ollama_base_url="http://localhost:11434",
    model="llama3.1:8b",
    max_turns=10,
)

# Single query
response = agent.run("Find errors similar to NullPointerException")
print(response)

# Interactive session
agent.chat()
```

### Advanced: Custom Model

```python
agent = TriageAgent(
    model="codellama:13b",          # Use CodeLlama for better code understanding
    timeout=180,                    # Longer timeout for 13B model
    max_turns=15,
)
```

## ReAct Loop Execution

The agent implements the **ReAct pattern** (Reasoning + Acting):

```
Turn 1:
  Input: "Help me understand error 12345"
  ↓
  Thought: I need error details and similar errors
  Action: suggest_fix(error_id="12345")
  Observation: [Tool result with full context]

Turn 2:
  Thought: I have enough context to analyze
  FINAL ANSWER: [Comprehensive response]
```

**Max turns limit:** Prevents infinite loops (default 10)

**Termination conditions:**
1. Model outputs final answer (no more tool calls)
2. Tool execution fails → error message returned
3. Max turns reached → generic response

## System Prompts

### OLLAMA_AGENT_SYSTEM_PROMPT

Guides the model to:
- Use the ReAct pattern (THOUGHT → ACTION → OBSERVATION)
- Output tool calls in the expected format
- Focus on actionable recommendations
- Consider error patterns and clusters

**Key instruction:**
```
When you need to use a tool, output in this EXACT format:
THOUGHT: [Your reasoning]
ACTION: tool_name(param1="value1", param2="value2")
```

## Error Handling

### Ollama Connection Errors
```
✗ Cannot connect to Ollama at http://localhost:11434
Setup: ollama serve
```

### Request Timeouts
- Default: 120 seconds
- Adjustable: `agent = TriageAgent(timeout=180)`
- Occurs when model is slow to respond

### Tool Execution Failures
- Tool catches exceptions
- Returns JSON error response
- Agent continues with next thought

## Testing

### Tool Tests
```bash
pytest tests/unit/test_tools.py -v
```

Tests cover:
- Similarity search with mocked vector store
- Stack trace parsing (various formats)
- Fix suggestion generation
- Tool schema validation

### Agent Tests
```bash
pytest tests/unit/test_agent.py -v
```

Tests cover:
- Agent initialization with mocked data
- ReAct loop execution
- Tool call parsing from Ollama responses
- Error handling (connection failures, timeouts)
- Max turns enforcement

## Performance Notes

### Model Latency

- **Llama 3.1 8B:** 2-10 seconds per response (depends on prompt size)
- **CodeLlama 13B:** 5-15 seconds per response
- **Mistral 7B:** 2-5 seconds per response (fastest)

### Memory Requirements

- **8B models:** 8GB RAM minimum
- **13B models:** 16GB RAM recommended
- **GPU acceleration:** Speeds up 2-10x (CUDA/Metal)

### Optimization Tips

1. Use smaller model for faster responses: `ollama pull mistral:7b`
2. Increase timeout for longer prompts: `timeout=180`
3. Reduce max_turns if quick answers needed: `max_turns=5`
4. Enable GPU acceleration if available

## Architecture Decisions

### Why Ollama?

| Aspect | Ollama | Claude API |
|--------|--------|-----------|
| Cost | Free | $$ per request |
| Privacy | Local | Cloud |
| Reproducibility | Deterministic | API changes |
| Deployment | Local/self-hosted | Cloud-only |
| Portfolio Value | Shows ML deployment | Shows cloud integration |

### Why Regex Tool Calling?

Since Ollama doesn't support native tool use, regex parsing is:
- **Simple:** Minimal parsing logic
- **Robust:** Handles minor formatting variations
- **Observable:** Easy to debug model outputs
- **Aligned with Plan:** Matches the documented approach

### Why These Three Tools?

1. **search_similar_errors** - Core value: find patterns
2. **analyze_stack_trace** - Understand problem structure
3. **suggest_fix** - Actionable recommendation

Each tool maps to a key capability:
- Step 2 → Similarity search
- Step 1 → Stack trace analysis
- Step 3 → Pattern context

## Future Improvements

1. **Multi-turn Conversation** - Maintain context across multiple queries
2. **Tool Refinement** - Add more specialized tools (extract logs, check metrics)
3. **Fine-tuning** - Train Ollama model on Java error corpus
4. **Streaming Responses** - Use Ollama streaming API for real-time output
5. **Confidence Scores** - Add uncertainty quantification to recommendations
6. **Integration with IDEs** - VS Code plugin, JetBrains plugin

## Interview Talking Points

### 1. Local LLM Integration
"The agent uses Ollama for local LLM execution, showing understanding of deployment beyond cloud APIs. This is more reproducible and privacy-preserving."

### 2. Tool-Augmented Reasoning
"The ReAct pattern demonstrates multi-step reasoning: the model thinks through problems, decides what tools to use, and iteratively refines its understanding."

### 3. Regex-based Tool Calling
"Since Ollama doesn't have native tool support, I implemented regex parsing for tool calls. This shows how to work with model outputs that don't have built-in structure."

### 4. Integration with Previous Steps
"The agent seamlessly integrates embeddings, clustering, and vector search from Steps 1-3, showing understanding of end-to-end ML systems."

### 5. Production Mindset
"The agent has proper error handling, logging, timeouts, and connection checks—it's designed to fail gracefully rather than crash."

## Troubleshooting

### Issue: "Cannot connect to Ollama"
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve
```

### Issue: "Model not found"
```bash
# Pull the model
ollama pull llama3.1:8b

# List available models
ollama list
```

### Issue: "Timeout" errors
```python
# Increase timeout for slower hardware
agent = TriageAgent(timeout=300)  # 5 minutes
```

### Issue: "Out of Memory"
```bash
# Use smaller model
ollama pull mistral:7b

# Or increase system RAM/GPU memory
```

## References

- [Ollama Documentation](https://github.com/ollama/ollama)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [FAISS Documentation](https://faiss.ai/)
- [Sentence Transformers](https://www.sbert.net/)
- [HDBSCAN Clustering](https://hdbscan.readthedocs.io/)
