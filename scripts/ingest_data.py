#!/usr/bin/env python3
"""
Data ingestion script - loads AERI JSON files and saves as JSONL.
Uses configuration from settings (config.py).

Usage:
    python scripts/ingest_data.py
"""
import json
from pathlib import Path
from datetime import datetime

from sentrylens.config import settings


def ingest_aeri(data_dir: Path, limit: int = None, output_path: Path = None) -> int:
    """Load AERI data and save as JSONL."""

    # Find all JSON files
    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return 1

    if limit:
        json_files = json_files[:limit]
        print(f"📁 Loading {limit} files from {data_dir}")
    else:
        print(f"📁 Loading all {len(json_files)} files from {data_dir}")

    # Load errors
    errors = []
    skipped = 0

    for i, file_path in enumerate(json_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            error = {
                'error_id': file_path.stem,
                'summary': data.get('summary', ''),
                'kind': data.get('kind', ''),
                'stacktraces': data.get('stacktraces', []),
                'numberOfIncidents': data.get('numberOfIncidents', 0),
                'numberOfReporters': data.get('numberOfReporters', 0),
                'javaRuntimeVersion': data.get('javaRuntimeVersion', ''),
                'osgiOs': data.get('osgiOs', ''),
                'eclipseProduct': data.get('eclipseProduct', ''),
                'createdOn': data.get('createdOn', ''),
            }
            errors.append(error)

            if (i + 1) % 5000 == 0:
                print(f"  Loaded {i + 1} files...")

        except (json.JSONDecodeError, IOError):
            skipped += 1
            continue

    if not errors:
        print("❌ No errors loaded")
        return 1

    # Save output
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = settings.PROCESSED_DATA_DIR / f"aeri_{timestamp}.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for error in errors:
            f.write(json.dumps(error, default=str) + '\n')

    print(f"Loaded {len(errors)} errors (skipped {skipped})")
    print(f"Saved to {output_path}")

    return 0


def main():
    """Ingest AERI data using settings from config."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = settings.PROCESSED_DATA_DIR / f"aeri_{timestamp}.jsonl"

    return ingest_aeri(
        data_dir=settings.AERI_DATA_DIR,
        limit=settings.SAMPLE_SIZE,
        output_path=output_path
    )


if __name__ == "__main__":
    exit(main())