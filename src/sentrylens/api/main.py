"""FastAPI application for SentryLens."""
import json
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="SentryLens API",
    description="Agentic AI system for error triage",
    version="0.1.0",
)

# Data loaded by init_app()
errors_dict = {}
clusters_dict = {}
vector_store = None
embedder = None
agent = None


def init_app(vector_store_path: Path, cluster_data_path: Path):
    """Load data into the app."""
    global errors_dict, clusters_dict, vector_store, embedder, agent

    # Load cluster data
    with open(cluster_data_path) as f:
        data = json.load(f)

    for error in data.get("errors", []):
        errors_dict[error["error_id"]] = error

    for cluster in data.get("clusters", []):
        clusters_dict[cluster["error_id"]] = cluster

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
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Error {error_id} not found")

    cluster = clusters_dict.get(error_id, {})
    return {
        **error,
        "cluster_id": cluster.get("cluster_id"),
    }


@app.get("/clusters")
def list_clusters():
    """List all clusters with their sizes."""
    from collections import Counter

    # Count errors per cluster
    cluster_counts = Counter(
        c["cluster_id"] for c in clusters_dict.values()
        if c["cluster_id"] != -1
    )

    clusters = [
        {"cluster_id": cid, "size": count}
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
    from fastapi import HTTPException

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
