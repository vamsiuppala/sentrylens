"""
Download SWE-bench dataset from HuggingFace.
"""
from datasets import load_dataset
import json
from pathlib import Path

def main():
    output_dir = Path("data/swebench")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download SWE-bench Lite (300 instances - good for development)
    print("Downloading SWE-bench Lite (300 instances)...")
    swebench_lite = load_dataset('SWE-bench/SWE-bench_Lite', split='test')
    
    # Save to JSON for easy inspection
    lite_path = output_dir / "swebench_lite.json"
    with open(lite_path, 'w') as f:
        # Convert to list of dicts
        data = [dict(item) for item in swebench_lite]
        json.dump(data, f, indent=2, default=str)
    print(f"Saved {len(data)} instances to {lite_path}")
    
    # Download SWE-bench Verified (500 human-validated instances)
    print("\nDownloading SWE-bench Verified (500 instances)...")
    swebench_verified = load_dataset('SWE-bench/SWE-bench_Verified', split='test')
    
    verified_path = output_dir / "swebench_verified.json"
    with open(verified_path, 'w') as f:
        data = [dict(item) for item in swebench_verified]
        json.dump(data, f, indent=2, default=str)
    print(f"Saved {len(data)} instances to {verified_path}")
    
    print("\nDone! Check data/swebench/ for the downloaded files.")

if __name__ == "__main__":
    main()
    