---
layout: post
title: "Building SentryLens Part 2: Clustering and the ReAct Agent"
date: 2026-02-02
categories: [ml-engineering, clustering, llm-agents]
---

# Building SentryLens Part 2: Clustering and the ReAct Agent

**TL;DR**: Used HDBSCAN to automatically group similar errors into clusters, then built a ReAct agent with Claude that leverages clustering, embeddings, and custom tools to intelligently triage errors and suggest fixes.

## The Problem

Having a searchable vector store is great, but developers need more:
1. **Pattern detection** - Which errors occur together? What are the high-impact issues?
2. **Intelligent triage** - How do we go beyond simple search to reasoning about errors?
3. **Actionable insights** - Can we suggest fixes based on similar error patterns?

We needed two things: automatic clustering to identify error patterns, and an agentic system that could reason about these patterns to help developers.

## The Approach

**Phase 1: HDBSCAN Clustering**
- Use density-based clustering to group semantically similar errors
- Automatically detect the number of clusters (no manual tuning)
- Mark outliers as noise points (rare, unique errors)

**Phase 2: ReAct Agent with Claude**
- Implement the ReAct (Reasoning + Acting) pattern
- Give the agent three tools: search similar errors, analyze stack traces, suggest fixes
- Leverage Claude's native tool_use capability for clean execution

The key insight: clustering provides structure (this error belongs to a known pattern), while the agent provides intelligence (here's what that pattern means and how to fix it).

## Implementation

### Phase 1: HDBSCAN Clustering

Why HDBSCAN over K-means or DBSCAN?
- **No need to specify K** - Automatically finds the optimal number of clusters
- **Handles varying densities** - Some error types are common, others are rare
- **Identifies noise** - Outliers are explicitly marked as cluster -1

Here's the core implementation:

```python
class HDBSCANClusterer:
    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        metric: str = "euclidean"
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples or min_cluster_size
        self.metric = metric

    def fit(self, embeddings: np.ndarray) -> np.ndarray:
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            prediction_data=True  # Enable prediction on new data
        )

        self.labels = self.clusterer.fit_predict(embeddings)
        return self.labels
```

The parameters matter:
- `min_cluster_size=5` - Minimum 5 errors to form a cluster (prevents over-fragmentation)
- `prediction_data=True` - Allows assigning new errors to existing clusters
- `metric="euclidean"` - Works well for normalized embeddings

Processing errors into clusters:

```python
def cluster_embeddings(
    self,
    embeddings: List[ErrorEmbedding],
    errors: Optional[List[AERIErrorRecord]] = None
) -> List[ClusterAssignment]:
    # Convert to numpy array
    embedding_vectors = np.array([e.embedding for e in embeddings])

    # Fit clustering
    labels = self.fit(embedding_vectors)

    # Create assignments with metadata
    assignments = []
    stats = self.get_stats()

    for embedding, label in zip(embeddings, labels):
        cluster_id = int(label)
        cluster_size = stats.cluster_sizes.get(cluster_id, 0) if cluster_id != -1 else None

        assignment = ClusterAssignment(
            error_id=embedding.error_id,
            cluster_id=cluster_id,
            cluster_size=cluster_size
        )
        assignments.append(assignment)

    return assignments
```

On 1,000 errors, HDBSCAN typically finds 20-40 meaningful clusters, with 5-15% noise points. This distribution makes sense - most errors fall into common patterns, but rare edge cases exist.

### Phase 2: ReAct Agent Architecture

The ReAct pattern combines reasoning (thinking through a problem) with acting (using tools to gather information). Our agent follows this loop:

1. Receive user query
2. Think about what information is needed
3. Use tools to get that information
4. Synthesize results into an answer
5. Repeat until confident

We use Claude's native tool_use capability, which is cleaner than parsing function calls from text:

```python
class TriageAgent:
    def __init__(
        self,
        vector_store_path: Path,
        cluster_data_path: Path,
        model: str = "claude-3-5-haiku-20241022",
        max_turns: int = 10
    ):
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model
        self.max_turns = max_turns

        # Load infrastructure from Phase 1
        self.vector_store = HnswlibVectorStore.load(vector_store_path)
        self.embedder = ErrorEmbedder()
        self.errors_dict, self.clusters_dict = self._load_data()

        # Initialize tools
        self.tools = TriageTools(
            vector_store=self.vector_store,
            embedder=self.embedder,
            errors_dict=self.errors_dict,
            clusters_dict=self.clusters_dict
        )
```

Notice how we compose all the Phase 1 components. The agent is the brain, but it needs eyes (embeddings), memory (vector store), and organization (clusters).

### The ReAct Loop

This is where the magic happens:

```python
def run(self, user_query: str) -> str:
    messages = [{"role": "user", "content": user_query}]

    for turn in range(self.max_turns):
        # Call Claude with tools
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=TRIAGE_AGENT_SYSTEM_PROMPT,
            tools=self.tools.get_tool_schemas(),
            messages=messages
        )

        # Check if Claude wants to use a tool
        if response.stop_reason == "tool_use":
            tool_use_block = None
            for block in response.content:
                if block.type == "tool_use":
                    tool_use_block = block
                    break

            # Add Claude's response to conversation
            messages.append({"role": "assistant", "content": response.content})

            # Execute the tool
            result = self._execute_tool(
                tool_name=tool_use_block.name,
                tool_input=tool_use_block.input
            )

            # Add tool result back to conversation
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use_block.id,
                    "content": result
                }]
            })

            continue

        # No tool use - Claude provided final answer
        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    return block.text

    return "Max reasoning turns reached. Please try a simpler query."
```

The beauty of this pattern: Claude decides when to use tools and when it has enough information. We just facilitate the loop.

### Tool Implementation

The agent has three tools that leverage our pipeline:

**Tool 1: Search Similar Errors**

```python
def search_similar_errors(self, query_text: str, top_k: int = 5) -> str:
    # Create temporary error record from query
    temp_error = AERIErrorRecord(
        error_id="query",
        error_type="Query",
        error_message=query_text[:200],
        stack_trace=query_text if "\n" in query_text else "Query stack trace"
    )

    # Generate embedding for query
    query_embedding_obj = self.embedder.embed_single(temp_error)

    # Search vector store
    results = self.vector_store.search(
        query_embedding_obj.embedding,
        top_k=top_k
    )

    # Build response with cluster context
    similar_errors = []
    for error_id, similarity_score in results:
        error = self.errors_dict.get(error_id)
        cluster_assignment = self.clusters_dict.get(error_id)

        similar_errors.append({
            "error_id": error_id,
            "error_type": error.error_type,
            "error_message": error.error_message[:150],
            "similarity_score": f"{similarity_score:.4f}",
            "cluster_id": cluster_assignment.cluster_id if cluster_assignment else None,
            "cluster_size": cluster_assignment.cluster_size if cluster_assignment else None
        })

    return json.dumps(similar_errors, indent=2)
```

This combines Phase 1 (embeddings + search) with Phase 2 (cluster context). The agent gets both similarity scores and cluster membership.

**Tool 2: Analyze Stack Trace**

```python
def analyze_stack_trace(self, stack_trace: str) -> str:
    frames = []
    exception_type = None

    # Parse stack trace for frames
    frame_pattern = r'\s*at\s+([^\(]+)\(([^:]+):(\d+)\)'

    for line in stack_trace.split('\n'):
        # Extract exception type from first line
        if not exception_type and ':' in line and ' at ' not in line:
            parts = line.split(':')
            if len(parts) >= 1:
                exception_type = parts[0].strip()

        # Extract frames
        match = re.match(frame_pattern, line)
        if match:
            frames.append({
                "method": match.group(1).strip(),
                "file": match.group(2).strip(),
                "line_number": int(match.group(3).strip())
            })

    # Identify root cause (first non-library frame)
    library_packages = ['java.', 'javax.', 'sun.', 'org.eclipse.']
    root_cause = None

    for frame in frames:
        is_library = any(lib in frame['method'] for lib in library_packages)
        if not is_library:
            root_cause = frame
            break

    return json.dumps({
        "exception_type": exception_type or "Unknown",
        "total_frames": len(frames),
        "root_cause": root_cause,
        "key_frames": frames[:5]
    }, indent=2)
```

This tool parses Java stack traces and identifies the likely root cause by filtering out library frames.

**Tool 3: Suggest Fix**

This is the most sophisticated tool - it combines information from all sources:

```python
def suggest_fix(self, error_id: str) -> str:
    # Step 1: Get error details
    error = self.errors_dict.get(error_id)

    # Step 2: Find similar errors
    similar_results = self.search_similar_errors(
        f"{error.error_type}: {error.error_message}",
        top_k=5
    )

    # Step 3: Get cluster context
    cluster_assignment = self.clusters_dict.get(error_id)
    cluster_context = {}

    if cluster_assignment and not cluster_assignment.is_noise:
        # Get all errors in same cluster
        cluster_members = [
            eid for eid, c in self.clusters_dict.items()
            if c.cluster_id == cluster_assignment.cluster_id
        ]

        # Analyze cluster composition
        error_types = [
            self.errors_dict[eid].error_type
            for eid in cluster_members[:10]
            if eid in self.errors_dict
        ]

        cluster_context = {
            "cluster_id": cluster_assignment.cluster_id,
            "cluster_size": cluster_assignment.cluster_size,
            "common_error_types": list(set(error_types)),
            "is_common_pattern": cluster_assignment.cluster_size > 10
        }

    # Step 4: Analyze stack trace
    stack_analysis = json.loads(self.analyze_stack_trace(error.stack_trace))

    # Step 5: Generate suggestion
    suggestion = self._generate_suggestion(
        error=error,
        stack_analysis=stack_analysis,
        similar_errors=json.loads(similar_results)[:3],
        cluster_context=cluster_context
    )

    return json.dumps({
        "error_id": error_id,
        "error_type": error.error_type,
        "root_cause": stack_analysis.get("root_cause"),
        "cluster_context": cluster_context,
        "suggestion": suggestion
    }, indent=2)
```

The suggestion logic uses rule-based heuristics:

```python
def _generate_suggestion(self, error, stack_analysis, similar_errors, cluster_context):
    suggestions = []

    # Rule 1: Common patterns in cluster
    if cluster_context.get("is_common_pattern"):
        cluster_size = cluster_context.get("cluster_size", 0)
        suggestions.append(
            f"This is a COMMON pattern ({cluster_size} similar errors). "
            f"Prioritize this fix as it affects multiple instances."
        )

    # Rule 2: NullPointerException
    if "NullPointerException" in error.error_type:
        root_cause = stack_analysis.get("root_cause", {})
        method = root_cause.get("method", "")
        suggestions.append(
            f"NullPointerException in {method}. "
            f"Add null checks before accessing object fields. "
            f"Consider using Optional or defensive programming."
        )

    # Rule 3: Similar patterns
    if similar_errors:
        suggestions.append(
            f"Found {len(similar_errors)} similar errors. "
            f"Review their fixes for patterns."
        )

    return " ".join(suggestions)
```

These rules leverage cluster context (is this a high-impact issue?) and error type patterns (what typically causes this?).

## Challenges and Solutions

### Challenge 1: Choosing min_cluster_size

Too small = fragmented clusters. Too large = everything is noise.

We settled on `min_cluster_size=5` after experimentation:
- Size 3: 60+ clusters, too granular
- Size 10: Only 15 clusters, missed patterns
- Size 5: 25-35 clusters, balanced

The key metric: do developers recognize clusters as meaningful patterns? Manual inspection validated size 5.

### Challenge 2: Agent Context Management

Early versions included full stack traces in tool responses. This overwhelmed Claude's context window.

Solution: Truncate and summarize:

```python
"error_message": error.error_message[:150],  # First 150 chars
"key_frames": frames[:5],  # Top 5 frames only
```

This reduced token usage by 70% while preserving essential information.

### Challenge 3: Tool Execution Failures

If a tool crashes, the ReAct loop breaks. We wrap all tools in try-catch:

```python
def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
    try:
        if tool_name == "search_similar_errors":
            return self.tools.search_similar_errors(
                query_text=str(tool_input.get("query_text", "")),
                top_k=int(tool_input.get("top_k", 5))
            )
        # ... other tools
    except Exception as e:
        logger.error("Tool execution failed", tool_name=tool_name, error=str(e))
        return json.dumps({
            "error": f"Tool execution failed: {str(e)}"
        })
```

The agent sees tool failures as results and can adapt (ask for clarification, use a different tool, etc.).

## Results

**Clustering Performance:**
- 1,000 errors → 32 clusters + 8% noise
- Largest cluster: 127 errors (NullPointerException in Eclipse UI)
- Average cluster size: 18 errors
- Silhouette score: 0.41 (indicates decent cluster cohesion)

**Agent Performance:**
- Average query: 2-3 tool calls before answer
- Response time: 3-5 seconds (including Claude API)
- Success rate: 92% (answers user query satisfactorily)

Example interaction:

```
User: "Find errors similar to NullPointerException in UserService"

Agent:
[Uses search_similar_errors tool]
[Receives 5 results with cluster context]

Response: "I found 5 similar NullPointerExceptions. The top match (similarity: 0.89)
is in cluster #12 with 23 similar errors. This suggests a common pattern where
UserService.getPreferences() is called on a null user object. I recommend adding
null checks before this method call and considering Optional<User> in your API."
```

The agent combines search results with cluster context to provide actionable advice.

## What's Next

We now have an intelligent error triage system with clustering and an agentic brain. In Part 3, we'll wrap this in a FastAPI backend, add a web UI for browsing clusters and chatting with the agent, and integrate with Sentry via webhooks for real-time error ingestion.

**Key Takeaways:**
- HDBSCAN is ideal for error clustering - no need to specify cluster count
- The ReAct pattern is powerful for building agentic systems with LLMs
- Tools should return concise, structured information (not raw dumps)
- Cluster context adds valuable signal for prioritization
- Rule-based heuristics complement LLM reasoning effectively

---