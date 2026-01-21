#!/usr/bin/env python3
"""
Analyze clustering results to validate cluster quality.
"""
import json
from collections import defaultdict
from pathlib import Path

def analyze_clusters(clusters_file: str):
    """Analyze clustering results."""
    path = Path(clusters_file)
    if not path.exists():
        print(f"File not found: {clusters_file}")
        return

    with open(path) as f:
        data = json.load(f)

    # Extract cluster assignments from the data
    clusters = defaultdict(list)
    error_types = defaultdict(int)

    if "clusters" in data:
        assignments = data["clusters"]
    else:
        # Fallback: parse from errors if structure is different
        print("Warning: No 'clusters' key found in JSON")
        return

    for assignment in assignments:
        cluster_id = assignment.get("cluster_id")
        error_id = assignment.get("error_id")
        clusters[cluster_id].append(error_id)

    # Also count error types to see if clusters group similar exceptions
    if "errors" in data:
        for error in data["errors"]:
            error_types[error.get("error_type", "UNKNOWN")] += 1

    # Print summary
    print("\n" + "="*60)
    print("CLUSTERING QUALITY ANALYSIS")
    print("="*60)

    # Stats
    num_clusters = len([k for k in clusters.keys() if k != -1])
    num_noise = len(clusters.get(-1, []))
    total_points = sum(len(v) for v in clusters.values())

    print(f"\n📊 Clustering Statistics:")
    print(f"  Total errors: {total_points}")
    print(f"  Clusters found: {num_clusters}")
    print(f"  Noise points (-1): {num_noise} ({100*num_noise/total_points:.1f}%)")

    # Cluster size distribution
    print(f"\n📈 Cluster Sizes:")
    cluster_sizes = sorted([len(v) for k, v in clusters.items() if k != -1])
    if cluster_sizes:
        print(f"  Largest: {max(cluster_sizes)}")
        print(f"  Smallest: {min(cluster_sizes)}")
        print(f"  Average: {sum(cluster_sizes)/len(cluster_sizes):.1f}")

    # Error type distribution
    print(f"\n🔍 Error Types in Dataset:")
    sorted_types = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
    for error_type, count in sorted_types[:10]:
        print(f"  {error_type}: {count}")

    # Show a few cluster examples
    print(f"\n🎯 Sample Clusters (first 3):")
    for i, (cluster_id, members) in enumerate(sorted(clusters.items())[:3]):
        if cluster_id == -1:
            continue
        print(f"\n  Cluster {cluster_id} ({len(members)} members):")
        # Show first 3 error IDs in cluster
        for member_id in members[:3]:
            print(f"    - {member_id}")
        if len(members) > 3:
            print(f"    ... and {len(members) - 3} more")

    print("\n" + "="*60)
    print("✓ Clusters generated successfully!")
    print("="*60 + "\n")

if __name__ == "__main__":
    import sys

    # Use provided file or default
    clusters_file = sys.argv[1] if len(sys.argv) > 1 else "data/processed/clusters_20260120_213215.json"
    analyze_clusters(clusters_file)
