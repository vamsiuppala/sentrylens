"""
Explore AERI JSON files (which contain actual stacktraces).
"""
import json
import os
from pathlib import Path

# Find where the JSON files are
aeri_dir = Path("data/aeri")

print("=" * 60)
print("SEARCHING FOR JSON FILES")
print("=" * 60)

# List what's in the directory after extraction
print("\nContents of data/aeri/:")
for item in aeri_dir.iterdir():
    if item.is_dir():
        # Count files in subdirectory
        files = list(item.iterdir())
        print(f"  📁 {item.name}/ ({len(files)} files)")
        # Show first few
        for f in files[:3]:
            print(f"      - {f.name}")
        if len(files) > 3:
            print(f"      ... and {len(files) - 3} more")
    else:
        size_mb = item.stat().st_size / (1024 * 1024)
        print(f"  📄 {item.name} ({size_mb:.1f} MB)")

# Try to find and load a JSON file
print("\n" + "=" * 60)
print("LOADING SAMPLE JSON FILE")
print("=" * 60)

json_files = list(aeri_dir.rglob("*.json"))
print(f"\nFound {len(json_files)} JSON files")

if json_files:
    # Load first JSON file
    sample_file = json_files[0]
    print(f"\nLoading: {sample_file}")
    
    with open(sample_file, 'r') as f:
        data = json.load(f)
    
    print(f"\n📋 KEYS IN JSON:")
    for key in data.keys():
        print(f"   - {key}")
    
    print(f"\n🔍 FULL CONTENT:")
    print("-" * 60)
    print(json.dumps(data, indent=2)[:3000])
    if len(json.dumps(data)) > 3000:
        print(f"\n... [truncated, {len(json.dumps(data))} total chars]")
    
    # Specifically look at stacktraces
    if 'stacktraces' in data:
        print(f"\n📚 STACKTRACE STRUCTURE:")
        print("-" * 60)
        stacktraces = data['stacktraces']
        print(f"Number of stacktraces: {len(stacktraces)}")
        
        if stacktraces:
            print(f"\nFirst stacktrace ({len(stacktraces[0])} frames):")
            for i, frame in enumerate(stacktraces[0][:5]):
                print(f"  Frame {i}: {json.dumps(frame)}")
            if len(stacktraces[0]) > 5:
                print(f"  ... and {len(stacktraces[0]) - 5} more frames")
else:
    print("\nNo JSON files found. The tar might extract to a different structure.")
    print("Check what was created by the extraction.")