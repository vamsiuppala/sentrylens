"""FastAPI application for SentryLens."""
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(
    title="SentryLens API",
    description="Agentic AI system for error triage",
    version="0.1.0",
)

# Data loaded by init_app()
errors_dict = {}
clusters_dict = {}
cluster_labels = {}  # Maps cluster_id (int) to label (str)
vector_store = None
embedder = None
agent = None


def init_app(vector_store_path: Path, cluster_data_path: Path):
    """Load data into the app."""
    global errors_dict, clusters_dict, cluster_labels, vector_store, embedder, agent

    # Load cluster data
    with open(cluster_data_path) as f:
        data = json.load(f)

    for error in data.get("errors", []):
        errors_dict[error["error_id"]] = error

    for cluster in data.get("clusters", []):
        clusters_dict[cluster["error_id"]] = cluster

    # Load cluster labels (keys are strings in JSON, convert to int)
    raw_labels = data.get("cluster_labels", {})
    cluster_labels.clear()
    for cid, label in raw_labels.items():
        cluster_labels[int(cid)] = label

    # Load vector store and embedder
    from sentrylens.embeddings.vector_store import HnswlibVectorStore
    from sentrylens.embeddings.embedder import ErrorEmbedder

    vector_store = HnswlibVectorStore.load(vector_store_path)
    embedder = ErrorEmbedder()

    # Initialize agent
    from sentrylens.agent import TriageAgent
    agent = TriageAgent(
        vector_store_path=vector_store_path,
        cluster_data_path=cluster_data_path,
    )


# --- Sentry Webhook Models ---
# These model Sentry's error event payload structure.
# Sentry nests exception info: event.exception.values[0].stacktrace.frames

class SentryEvent(BaseModel):
    """Top-level Sentry error event."""
    event_id: str  # Required - unique error identifier
    message: Optional[str] = None  # Error message (sometimes empty)
    culprit: Optional[str] = None  # Location where error occurred
    platform: Optional[str] = None  # python, javascript, java, etc.
    timestamp: Optional[str] = None
    # exception contains the actual error details - we'll handle it as a dict
    # to keep things simple (Sentry's schema is deeply nested)
    exception: Optional[dict] = None


def sentry_to_aeri(event: SentryEvent):
    """
    Convert Sentry event to AERIErrorRecord.

    Sentry structure (simplified):
    {
        "event_id": "abc123",
        "exception": {
            "values": [{
                "type": "ValueError",
                "value": "invalid input",
                "stacktrace": {
                    "frames": [{"filename": "app.py", "function": "main", "lineno": 42}]
                }
            }]
        }
    }
    """
    from sentrylens.core.models import AERIErrorRecord

    # Default values
    error_type = "UnknownError"
    error_message = event.message or "No message"
    stack_lines = []

    # Extract from exception if present
    if event.exception and "values" in event.exception:
        values = event.exception["values"]
        if values:
            first = values[0]
            error_type = first.get("type", error_type)
            error_message = first.get("value", error_message) or error_message

            # Build stack trace from frames
            stacktrace = first.get("stacktrace", {})
            frames = stacktrace.get("frames", [])
            for frame in frames:
                fn = frame.get("function", "?")
                filename = frame.get("filename", "?")
                lineno = frame.get("lineno", "?")
                stack_lines.append(f"  at {fn}({filename}:{lineno})")

    # If no stack trace, create a minimal one
    stack_trace = "\n".join(stack_lines) if stack_lines else f"{error_type}: {error_message}"

    return AERIErrorRecord(
        error_id=event.event_id,
        error_type=error_type,
        error_message=error_message,
        stack_trace=stack_trace,
    )


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
def health_check():
    """Check if the server is running and data is loaded."""
    num_clusters = len(set(
        c["cluster_id"] for c in clusters_dict.values()
        if c["cluster_id"] != -1
    )) if clusters_dict else 0

    return {
        "status": "ok",
        "version": "0.1.0",
        "errors_loaded": len(errors_dict),
        "clusters_loaded": num_clusters,
    }


@app.get("/errors")
def list_errors(limit: int = 10, offset: int = 0):
    """List errors with pagination."""
    error_list = list(errors_dict.values())[offset:offset + limit]
    return {
        "errors": [
            {
                "error_id": e["error_id"],
                "error_type": e.get("error_type", ""),
                "error_message": e.get("error_message", "")[:200],
            }
            for e in error_list
        ],
        "total": len(errors_dict),
    }


@app.get("/errors/{error_id}")
def get_error(error_id: str):
    """Get a single error by ID."""
    error = errors_dict.get(error_id)
    if not error:
        raise HTTPException(status_code=404, detail=f"Error {error_id} not found")

    cluster = clusters_dict.get(error_id, {})
    return {
        **error,
        "cluster_id": cluster.get("cluster_id"),
    }


@app.get("/clusters")
def list_clusters():
    """List all clusters with their sizes and labels."""
    from collections import Counter

    # Count errors per cluster
    cluster_counts = Counter(
        c["cluster_id"] for c in clusters_dict.values()
        if c["cluster_id"] != -1
    )

    clusters = [
        {
            "cluster_id": cid,
            "size": count,
            "label": cluster_labels.get(cid, f"Cluster {cid}"),
        }
        for cid, count in cluster_counts.most_common()
    ]

    noise_count = sum(1 for c in clusters_dict.values() if c["cluster_id"] == -1)

    return {
        "clusters": clusters,
        "total_clusters": len(clusters),
        "noise_points": noise_count,
    }


@app.get("/clusters/{cluster_id}")
def get_cluster(cluster_id: int, limit: int = 10):
    """Get errors in a specific cluster."""
    error_ids = [
        eid for eid, c in clusters_dict.items()
        if c["cluster_id"] == cluster_id
    ]

    if not error_ids:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")

    errors = [
        {
            "error_id": eid,
            "error_type": errors_dict[eid].get("error_type", ""),
            "error_message": errors_dict[eid].get("error_message", "")[:200],
        }
        for eid in error_ids[:limit]
    ]

    return {
        "cluster_id": cluster_id,
        "label": cluster_labels.get(cluster_id, f"Cluster {cluster_id}"),
        "size": len(error_ids),
        "errors": errors,
    }


@app.post("/errors/search")
def search_errors(request: SearchRequest):
    """Search for similar errors using semantic similarity."""
    from sentrylens.core.models import AERIErrorRecord

    # Create a temporary error record for embedding
    temp_error = AERIErrorRecord(
        error_id="query",
        error_type="Query",
        error_message=request.query[:200],
        stack_trace=request.query if "\n" in request.query else "N/A",
    )

    # Generate embedding and search
    embedding = embedder.embed_single(temp_error)
    results = vector_store.search(embedding.embedding, top_k=request.top_k)

    # Build response
    matches = []
    for error_id, score in results:
        error = errors_dict.get(error_id, {})
        matches.append({
            "error_id": error_id,
            "score": round(score, 4),
            "error_type": error.get("error_type", ""),
            "error_message": error.get("error_message", "")[:200],
        })

    return {"results": matches}


@app.post("/query")
def query_agent(request: QueryRequest):
    """Send a query to the triage agent."""
    response = agent.run(request.query)
    return {"response": response}


@app.post("/webhooks/sentry")
def sentry_webhook(event: SentryEvent):
    """
    Receive error events from Sentry.

    Flow:
    1. Parse Sentry payload (FastAPI does this automatically via SentryEvent)
    2. Convert to AERI format
    3. Generate embedding (so it's searchable)
    4. Find nearest neighbor → use its cluster
    5. Store in memory (errors_dict, clusters_dict)

    Note: Data is lost on restart - this is intentional for simplicity.
    """
    # Check for duplicates
    if event.event_id in errors_dict:
        raise HTTPException(status_code=409, detail=f"Error {event.event_id} already exists")

    # Convert Sentry → AERI
    aeri = sentry_to_aeri(event)

    # Generate embedding and add to vector store
    error_embedding = embedder.embed_single(aeri)
    vector_store.add_embeddings([error_embedding])

    # Find nearest neighbor and use its cluster
    cluster_id = -1  # Default to noise
    results = vector_store.search(error_embedding.embedding, top_k=1)
    if results:
        nearest_id, similarity = results[0]
        # Only assign cluster if similarity is high enough (> 0.5)
        if similarity > 0.5 and nearest_id in clusters_dict:
            cluster_id = clusters_dict[nearest_id]["cluster_id"]

    # Store in memory
    errors_dict[aeri.error_id] = {
        "error_id": aeri.error_id,
        "error_type": aeri.error_type,
        "error_message": aeri.error_message,
        "stack_trace": aeri.stack_trace,
    }

    clusters_dict[aeri.error_id] = {
        "error_id": aeri.error_id,
        "cluster_id": cluster_id,
    }

    # Get cluster label if assigned to a cluster
    label = None
    if cluster_id != -1:
        label = cluster_labels.get(cluster_id, f"Cluster {cluster_id}")

    return {
        "status": "received",
        "error_id": aeri.error_id,
        "cluster_id": cluster_id,
        "cluster_label": label,
    }


# Mount static files for web UI
# Must be after all routes so API endpoints take precedence
static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
