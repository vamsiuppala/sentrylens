"""
Data preparation for contrastive learning on error embeddings.

This script creates training pairs from AERI data:
- Positive pairs: errors with same exception type or similar stack traces
- Negative pairs: errors with different exception types

Usage:
    python -m sentrylens.training.data_preparation
"""
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from tqdm import tqdm
import hashlib


@dataclass
class ErrorSample:
    """A single error sample for training."""
    error_id: str
    text: str                    # Formatted for embedding model
    summary: str
    exception_type: str
    top_frame_class: str         # For grouping similar errors
    top_frame_method: str
    stacktrace_hash: str         # Hash of top 5 frames for similarity
    num_incidents: int
    num_reporters: int


@dataclass
class ContrastivePair:
    """A pair of errors for contrastive learning."""
    anchor: ErrorSample
    positive: ErrorSample        # Similar to anchor
    negative: ErrorSample        # Different from anchor
    similarity_reason: str       # Why anchor and positive are similar


def compute_stacktrace_hash(frames: List[Dict], top_n: int = 5) -> str:
    """
    Compute a hash of the top N stack frames.
    
    Errors with the same hash likely have the same root cause.
    """
    if not frames:
        return "empty"
    
    # Take top N frames and create a string representation
    frame_strs = []
    for frame in frames[:top_n]:
        if isinstance(frame, dict):
            cn = frame.get('cN', '')
            mn = frame.get('mN', '')
            frame_strs.append(f"{cn}.{mn}")
    
    combined = "|".join(frame_strs)
    return hashlib.md5(combined.encode()).hexdigest()[:12]


def format_error_for_embedding(summary: str, frames: List[Dict], max_frames: int = 15) -> str:
    """
    Format an error for the embedding model.
    
    Format:
        Error: <exception summary>
        Stack Trace:
            at class.method(file:line)
            ...
    """
    lines = [f"Error: {summary}", "Stack Trace:"]
    
    for frame in frames[:max_frames]:
        if isinstance(frame, dict):
            cn = frame.get('cN', 'Unknown')
            mn = frame.get('mN', 'unknown')
            fn = frame.get('fN', 'Unknown.java')
            ln = frame.get('lN', 0)
            lines.append(f"    at {cn}.{mn}({fn}:{ln})")
    
    return "\n".join(lines)


def extract_exception_type(summary: str) -> str:
    """Extract exception type from summary."""
    summary = summary.strip()
    
    # Handle "java.lang.NullPointerException: message" format
    if ':' in summary:
        exc_part = summary.split(':')[0].strip()
        # Get just the class name if it's fully qualified
        if '.' in exc_part:
            return exc_part.split('.')[-1]
        return exc_part
    
    # Handle "NullPointerException message" format
    first_word = summary.split()[0] if summary else "Unknown"
    return first_word


def load_and_process_aeri_data(
    data_dir: Path,
    limit: Optional[int] = None,
    min_incidents: int = 1
) -> List[ErrorSample]:
    """
    Load AERI JSON files and convert to ErrorSample objects.
    
    Args:
        data_dir: Directory containing JSON files
        limit: Maximum number of files to load
        min_incidents: Minimum incident count to include
        
    Returns:
        List of ErrorSample objects
    """
    json_files = list(data_dir.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files")
    
    if limit:
        json_files = json_files[:limit]
    
    samples = []
    skipped = 0
    
    for json_path in tqdm(json_files, desc="Loading AERI data"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Skip if too few incidents
            num_incidents = data.get('numberOfIncidents', 0)
            if num_incidents < min_incidents:
                skipped += 1
                continue
            
            # Get stacktraces
            stacktraces = data.get('stacktraces', [])
            if not stacktraces or not stacktraces[0]:
                skipped += 1
                continue
            
            primary_trace = stacktraces[0]
            if not isinstance(primary_trace, list) or len(primary_trace) == 0:
                skipped += 1
                continue
            
            summary = data.get('summary', 'Unknown error')
            
            # Extract metadata
            top_frame = primary_trace[0] if primary_trace else {}
            top_frame_class = top_frame.get('cN', 'Unknown') if isinstance(top_frame, dict) else 'Unknown'
            top_frame_method = top_frame.get('mN', 'unknown') if isinstance(top_frame, dict) else 'unknown'
            
            sample = ErrorSample(
                error_id=json_path.stem,
                text=format_error_for_embedding(summary, primary_trace),
                summary=summary,
                exception_type=extract_exception_type(summary),
                top_frame_class=top_frame_class,
                top_frame_method=top_frame_method,
                stacktrace_hash=compute_stacktrace_hash(primary_trace),
                num_incidents=num_incidents,
                num_reporters=data.get('numberOfReporters', 0)
            )
            samples.append(sample)
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            skipped += 1
            continue
    
    print(f"Loaded {len(samples)} samples, skipped {skipped}")
    return samples


def create_contrastive_pairs(
    samples: List[ErrorSample],
    pairs_per_anchor: int = 5,
    seed: int = 42
) -> List[ContrastivePair]:
    """
    Create contrastive learning pairs from error samples.
    
    Strategy:
    1. Group samples by exception type and stacktrace hash
    2. For each anchor, find positives (same group) and negatives (different group)
    
    Args:
        samples: List of ErrorSample objects
        pairs_per_anchor: Number of pairs to create per anchor
        seed: Random seed for reproducibility
        
    Returns:
        List of ContrastivePair objects
    """
    random.seed(seed)
    
    # Group by exception type
    by_exception_type = defaultdict(list)
    for s in samples:
        by_exception_type[s.exception_type].append(s)
    
    # Group by stacktrace hash (more precise similarity)
    by_stack_hash = defaultdict(list)
    for s in samples:
        by_stack_hash[s.stacktrace_hash].append(s)
    
    # Filter to groups with multiple samples
    exception_types_with_multiple = {
        k: v for k, v in by_exception_type.items() if len(v) >= 2
    }
    stack_hashes_with_multiple = {
        k: v for k, v in by_stack_hash.items() if len(v) >= 2
    }
    
    print(f"Exception types with 2+ samples: {len(exception_types_with_multiple)}")
    print(f"Stack hashes with 2+ samples: {len(stack_hashes_with_multiple)}")
    
    pairs = []
    all_samples = samples.copy()
    
    for anchor in tqdm(samples, desc="Creating pairs"):
        anchor_pairs = 0
        
        # Strategy 1: Same stacktrace hash (strongest similarity)
        if anchor.stacktrace_hash in stack_hashes_with_multiple:
            same_hash = [s for s in by_stack_hash[anchor.stacktrace_hash] if s.error_id != anchor.error_id]
            if same_hash:
                positive = random.choice(same_hash)
                
                # Find negative: different exception type
                different_types = [s for s in all_samples 
                                   if s.exception_type != anchor.exception_type]
                if different_types:
                    negative = random.choice(different_types)
                    pairs.append(ContrastivePair(
                        anchor=anchor,
                        positive=positive,
                        negative=negative,
                        similarity_reason="same_stacktrace_hash"
                    ))
                    anchor_pairs += 1
        
        # Strategy 2: Same exception type
        if anchor_pairs < pairs_per_anchor and anchor.exception_type in exception_types_with_multiple:
            same_type = [s for s in by_exception_type[anchor.exception_type] 
                         if s.error_id != anchor.error_id]
            
            for _ in range(min(pairs_per_anchor - anchor_pairs, len(same_type))):
                positive = random.choice(same_type)
                
                # Find negative: different exception type
                different_types = [s for s in all_samples 
                                   if s.exception_type != anchor.exception_type]
                if different_types:
                    negative = random.choice(different_types)
                    pairs.append(ContrastivePair(
                        anchor=anchor,
                        positive=positive,
                        negative=negative,
                        similarity_reason="same_exception_type"
                    ))
                    anchor_pairs += 1
        
        # Strategy 3: Same top frame class (if we still need more pairs)
        if anchor_pairs < pairs_per_anchor:
            same_class = [s for s in all_samples 
                          if s.top_frame_class == anchor.top_frame_class 
                          and s.error_id != anchor.error_id]
            
            if len(same_class) >= 1:
                positive = random.choice(same_class)
                
                # Find negative: different top frame class
                different_class = [s for s in all_samples 
                                   if s.top_frame_class != anchor.top_frame_class]
                if different_class:
                    negative = random.choice(different_class)
                    pairs.append(ContrastivePair(
                        anchor=anchor,
                        positive=positive,
                        negative=negative,
                        similarity_reason="same_top_frame_class"
                    ))
    
    print(f"Created {len(pairs)} contrastive pairs")
    return pairs


def save_training_data(
    samples: List[ErrorSample],
    pairs: List[ContrastivePair],
    output_dir: Path
):
    """
    Save processed data for training.
    
    Creates:
    - samples.jsonl: All error samples
    - pairs_train.jsonl: Training pairs (80%)
    - pairs_val.jsonl: Validation pairs (20%)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all samples
    samples_path = output_dir / "samples.jsonl"
    with open(samples_path, 'w') as f:
        for s in samples:
            f.write(json.dumps(asdict(s)) + '\n')
    print(f"Saved {len(samples)} samples to {samples_path}")
    
    # Shuffle and split pairs
    random.shuffle(pairs)
    split_idx = int(len(pairs) * 0.8)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    
    # Save training pairs
    train_path = output_dir / "pairs_train.jsonl"
    with open(train_path, 'w') as f:
        for p in train_pairs:
            record = {
                'anchor_text': p.anchor.text,
                'positive_text': p.positive.text,
                'negative_text': p.negative.text,
                'anchor_id': p.anchor.error_id,
                'positive_id': p.positive.error_id,
                'negative_id': p.negative.error_id,
                'similarity_reason': p.similarity_reason
            }
            f.write(json.dumps(record) + '\n')
    print(f"Saved {len(train_pairs)} training pairs to {train_path}")
    
    # Save validation pairs
    val_path = output_dir / "pairs_val.jsonl"
    with open(val_path, 'w') as f:
        for p in val_pairs:
            record = {
                'anchor_text': p.anchor.text,
                'positive_text': p.positive.text,
                'negative_text': p.negative.text,
                'anchor_id': p.anchor.error_id,
                'positive_id': p.positive.error_id,
                'negative_id': p.negative.error_id,
                'similarity_reason': p.similarity_reason
            }
            f.write(json.dumps(record) + '\n')
    print(f"Saved {len(val_pairs)} validation pairs to {val_path}")
    
    # Save statistics
    stats = {
        'num_samples': len(samples),
        'num_train_pairs': len(train_pairs),
        'num_val_pairs': len(val_pairs),
        'similarity_reasons': defaultdict(int)
    }
    for p in pairs:
        stats['similarity_reasons'][p.similarity_reason] += 1
    stats['similarity_reasons'] = dict(stats['similarity_reasons'])
    
    stats_path = output_dir / "stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_path}")


def main():
    """Main entry point for data preparation."""
    
    # Configuration
    AERI_DATA_DIR = Path("data/aeri")  # Adjust to your extracted JSON location
    OUTPUT_DIR = Path("data/processed")
    LIMIT = None  # Set to e.g. 10000 for testing, None for all data
    MIN_INCIDENTS = 2  # Minimum incidents to include a sample
    PAIRS_PER_ANCHOR = 3
    
    print("=" * 60)
    print("STEP 2.2: DATA PREPARATION FOR CONTRASTIVE LEARNING")
    print("=" * 60)
    
    # Find JSON files
    json_dirs = [
        AERI_DATA_DIR,
        AERI_DATA_DIR / "problems_full",
        AERI_DATA_DIR / "problems",
    ]
    
    data_dir = None
    for d in json_dirs:
        if d.exists() and list(d.rglob("*.json")):
            data_dir = d
            break
    
    if not data_dir:
        print("❌ Could not find AERI JSON files!")
        print("   Check that data/aeri/ contains extracted JSON files")
        return
    
    print(f"\n📁 Using data directory: {data_dir}")
    
    # Load and process data
    print("\n📥 Loading AERI data...")
    samples = load_and_process_aeri_data(data_dir, limit=LIMIT, min_incidents=MIN_INCIDENTS)
    
    if len(samples) < 100:
        print("⚠️  Warning: Very few samples loaded. Check your data directory.")
    
    # Create contrastive pairs
    print("\n🔗 Creating contrastive pairs...")
    pairs = create_contrastive_pairs(samples, pairs_per_anchor=PAIRS_PER_ANCHOR)
    
    # Save training data
    print("\n💾 Saving training data...")
    save_training_data(samples, pairs, OUTPUT_DIR)
    
    print("\n✅ Data preparation complete!")
    print(f"\nOutput files in {OUTPUT_DIR}:")
    print("  - samples.jsonl: All error samples")
    print("  - pairs_train.jsonl: Training pairs")
    print("  - pairs_val.jsonl: Validation pairs")
    print("  - stats.json: Dataset statistics")


if __name__ == "__main__":
    main()