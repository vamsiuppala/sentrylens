"""
Parse and format stack traces from AERI JSON data.
This prepares data for your embedding model (CodeBERT).

Usage:
    python parse_stacktrace.py
"""
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
import random


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class StackFrame:
    """A single frame in a stack trace."""
    class_name: str
    method_name: str
    file_name: str
    line_number: int
    
    def __str__(self) -> str:
        return f"    at {self.class_name}.{self.method_name}({self.file_name}:{self.line_number})"
    
    @classmethod
    def from_dict(cls, frame: Dict) -> "StackFrame":
        """Parse a frame from AERI JSON format."""
        return cls(
            class_name=frame.get('cN', 'Unknown'),
            method_name=frame.get('mN', 'unknown'),
            file_name=frame.get('fN', 'Unknown.java'),
            line_number=frame.get('lN', 0)
        )


@dataclass
class ParsedError:
    """A fully parsed error with metadata and stacktraces."""
    summary: str
    stacktraces: List[List[StackFrame]]
    num_incidents: int
    num_reporters: int
    os: str
    java_version: str
    eclipse_product: str
    file_path: Optional[str] = None
    
    def get_primary_stacktrace(self) -> List[StackFrame]:
        """Get the main stacktrace (first one)."""
        return self.stacktraces[0] if self.stacktraces else []
    
    def format_for_embedding(self, max_frames: int = 20) -> str:
        """
        Format this error for input to an embedding model.
        
        Returns a string like:
            Error: NullPointerException: Cannot invoke method on null
            Stack Trace:
                at org.eclipse.ui.WorkbenchPage.openEditor(WorkbenchPage.java:123)
                at org.eclipse.ui.EditorManager.open(EditorManager.java:456)
                ...
        """
        lines = [f"Error: {self.summary}", "Stack Trace:"]
        
        primary_trace = self.get_primary_stacktrace()
        for frame in primary_trace[:max_frames]:
            lines.append(str(frame))
        
        if len(primary_trace) > max_frames:
            lines.append(f"    ... and {len(primary_trace) - max_frames} more frames")
        
        return "\n".join(lines)
    
    def format_with_context(self, max_frames: int = 20) -> str:
        """
        Format with additional context (OS, Java version, etc.).
        Useful for severity prediction features.
        """
        base = self.format_for_embedding(max_frames)
        context = f"\nContext: OS={self.os}, Java={self.java_version}, Product={self.eclipse_product}"
        return base + context
    
    def get_exception_type(self) -> str:
        """Extract the exception type from the summary."""
        # Common patterns: "NullPointerException: message" or "java.lang.NPE: message"
        summary = self.summary.strip()
        if ':' in summary:
            return summary.split(':')[0].strip()
        return summary.split()[0] if summary else "Unknown"
    
    def get_top_frame(self) -> Optional[StackFrame]:
        """Get the top (most recent) stack frame."""
        trace = self.get_primary_stacktrace()
        return trace[0] if trace else None


# =============================================================================
# PARSING FUNCTIONS
# =============================================================================

def parse_stacktrace_frames(raw_frames: List[Dict]) -> List[StackFrame]:
    """Parse a list of raw frame dicts into StackFrame objects."""
    frames = []
    for frame in raw_frames:
        if isinstance(frame, dict):
            frames.append(StackFrame.from_dict(frame))
        elif isinstance(frame, list):
            # Nested stacktrace (e.g., caused by) - flatten for now
            frames.extend(parse_stacktrace_frames(frame))
    return frames


def parse_problem_json(json_path: Path) -> Optional[ParsedError]:
    """
    Parse a single AERI problem JSON file.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        ParsedError object or None if parsing fails
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Parse all stacktraces
        raw_stacktraces = data.get('stacktraces', [])
        stacktraces = []
        for raw_trace in raw_stacktraces:
            if isinstance(raw_trace, list):
                frames = parse_stacktrace_frames(raw_trace)
                if frames:
                    stacktraces.append(frames)
        
        return ParsedError(
            summary=data.get('summary', 'Unknown error'),
            stacktraces=stacktraces,
            num_incidents=data.get('numberOfIncidents', 0),
            num_reporters=data.get('numberOfReporters', 0),
            os=data.get('osgiOs', 'Unknown'),
            java_version=data.get('javaRuntimeVersion', 'Unknown'),
            eclipse_product=data.get('eclipseProduct', 'Unknown'),
            file_path=str(json_path)
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Error parsing {json_path}: {e}")
        return None


def load_all_problems(data_dir: Path, limit: Optional[int] = None) -> List[ParsedError]:
    """
    Load all problem JSON files from a directory.
    
    Args:
        data_dir: Directory containing JSON files
        limit: Maximum number of files to load (None for all)
        
    Returns:
        List of ParsedError objects
    """
    json_files = list(data_dir.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files")
    
    if limit:
        json_files = json_files[:limit]
        print(f"Loading first {limit} files...")
    
    problems = []
    for i, json_path in enumerate(json_files):
        if i % 10000 == 0 and i > 0:
            print(f"  Loaded {i} files...")
        
        parsed = parse_problem_json(json_path)
        if parsed and parsed.stacktraces:  # Only keep problems with stacktraces
            problems.append(parsed)
    
    print(f"Successfully parsed {len(problems)} problems with stacktraces")
    return problems


# =============================================================================
# DATA ANALYSIS FUNCTIONS
# =============================================================================

def analyze_problems(problems: List[ParsedError]) -> Dict:
    """
    Analyze the loaded problems for statistics.
    
    Useful for understanding your data before training.
    """
    stats = {
        'total_problems': len(problems),
        'total_incidents': sum(p.num_incidents for p in problems),
        'total_reporters': sum(p.num_reporters for p in problems),
        'avg_stacktrace_depth': 0,
        'exception_types': Counter(),
        'os_distribution': Counter(),
        'top_classes': Counter(),
    }
    
    depths = []
    for p in problems:
        # Exception types
        stats['exception_types'][p.get_exception_type()] += 1
        
        # OS distribution
        stats['os_distribution'][p.os] += 1
        
        # Stacktrace depth
        primary = p.get_primary_stacktrace()
        if primary:
            depths.append(len(primary))
            
            # Top frame classes
            top_frame = primary[0]
            stats['top_classes'][top_frame.class_name] += 1
    
    if depths:
        stats['avg_stacktrace_depth'] = sum(depths) / len(depths)
        stats['min_stacktrace_depth'] = min(depths)
        stats['max_stacktrace_depth'] = max(depths)
    
    return stats


def print_stats(stats: Dict):
    """Pretty print the statistics."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    
    print(f"\n📊 Overview:")
    print(f"   Total problems: {stats['total_problems']:,}")
    print(f"   Total incidents: {stats['total_incidents']:,}")
    print(f"   Total reporters: {stats['total_reporters']:,}")
    
    print(f"\n📚 Stacktrace Depth:")
    print(f"   Average: {stats['avg_stacktrace_depth']:.1f} frames")
    print(f"   Min: {stats.get('min_stacktrace_depth', 'N/A')}")
    print(f"   Max: {stats.get('max_stacktrace_depth', 'N/A')}")
    
    print(f"\n🐛 Top 10 Exception Types:")
    for exc_type, count in stats['exception_types'].most_common(10):
        print(f"   {exc_type}: {count:,}")
    
    print(f"\n💻 OS Distribution:")
    for os_name, count in stats['os_distribution'].most_common(5):
        pct = 100 * count / stats['total_problems']
        print(f"   {os_name}: {count:,} ({pct:.1f}%)")
    
    print(f"\n📍 Top 10 Error Locations (classes):")
    for class_name, count in stats['top_classes'].most_common(10):
        print(f"   {class_name}: {count:,}")


# =============================================================================
# EXPORT FUNCTIONS (for training)
# =============================================================================

def export_for_embedding_training(
    problems: List[ParsedError],
    output_path: Path,
    max_frames: int = 20
):
    """
    Export problems in a format ready for embedding model training.
    
    Creates a JSONL file with one error per line.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for p in problems:
            record = {
                'text': p.format_for_embedding(max_frames),
                'summary': p.summary,
                'exception_type': p.get_exception_type(),
                'num_incidents': p.num_incidents,
                'num_reporters': p.num_reporters,
                'stacktrace_depth': len(p.get_primary_stacktrace()),
                'os': p.os,
                'java_version': p.java_version,
            }
            f.write(json.dumps(record) + '\n')
    
    print(f"Exported {len(problems)} problems to {output_path}")


def create_severity_labels(problems: List[ParsedError]) -> List[Tuple[ParsedError, str]]:
    """
    Create severity labels based on incident count.
    
    This is a simple heuristic - you might want to refine this.
    
    Returns list of (problem, severity_label) tuples.
    """
    labeled = []
    for p in problems:
        incidents = p.num_incidents
        
        if incidents >= 1000:
            severity = "critical"
        elif incidents >= 100:
            severity = "high"
        elif incidents >= 10:
            severity = "medium"
        else:
            severity = "low"
        
        labeled.append((p, severity))
    
    return labeled


# =============================================================================
# MAIN - DEMO AND EXPLORATION
# =============================================================================

def main():
    # Find the AERI data directory
    # Adjust this path based on where your data extracted to
    possible_paths = [
        Path("data/aeri"),
        Path("data/aeri/problems_full"),
        Path("data/aeri/problems"),
    ]
    
    data_dir = None
    for p in possible_paths:
        if p.exists() and list(p.rglob("*.json")):
            data_dir = p
            break
    
    if not data_dir:
        print("❌ Could not find AERI JSON files!")
        print("   Make sure you extracted problems_full.tar.bz2")
        print("   Expected location: data/aeri/")
        return
    
    print(f"📁 Using data directory: {data_dir}")
    
    # Load a sample of problems (use limit=None for all)
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    problems = load_all_problems(data_dir, limit=1000)  # Start with 1000 for testing
    
    if not problems:
        print("❌ No problems loaded!")
        return
    
    # Analyze the data
    stats = analyze_problems(problems)
    print_stats(stats)
    
    # Show some examples
    print("\n" + "=" * 60)
    print("SAMPLE FORMATTED ERRORS (for embedding model)")
    print("=" * 60)
    
    # Pick a few random examples
    samples = random.sample(problems, min(3, len(problems)))
    
    for i, problem in enumerate(samples, 1):
        print(f"\n{'─' * 60}")
        print(f"EXAMPLE {i}")
        print(f"{'─' * 60}")
        print(f"Incidents: {problem.num_incidents:,} | Reporters: {problem.num_reporters:,}")
        print(f"Exception Type: {problem.get_exception_type()}")
        print()
        print(problem.format_for_embedding(max_frames=10))
    
    # Export a sample for training
    print("\n" + "=" * 60)
    print("EXPORTING TRAINING DATA")
    print("=" * 60)
    
    output_path = Path("data/processed/aeri_embedding_data.jsonl")
    export_for_embedding_training(problems, output_path)
    
    # Show severity distribution
    print("\n" + "=" * 60)
    print("SEVERITY LABEL DISTRIBUTION")
    print("=" * 60)
    
    labeled = create_severity_labels(problems)
    severity_counts = Counter(label for _, label in labeled)
    
    for severity, count in severity_counts.most_common():
        pct = 100 * count / len(labeled)
        print(f"   {severity.upper()}: {count:,} ({pct:.1f}%)")
    
    print("\n✅ Done! Check data/processed/aeri_embedding_data.jsonl")
    print("\nNext steps:")
    print("1. Review the exported JSONL file")
    print("2. Adjust severity thresholds if needed")
    print("3. Run with limit=None to process all ~125K problems")


if __name__ == "__main__":
    main()