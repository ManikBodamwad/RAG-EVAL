"""
scripts/push_dataset_to_hf.py — One-Time HF Hub Dataset Push Script

Run this locally once to push your golden_dataset.jsonl to Hugging Face Hub.
After pushing, CI will load it via datasets.load_dataset() — no re-upload needed.

Requirements:
  pip install datasets huggingface_hub
  huggingface-cli login  (or set HF_TOKEN env var)

Usage:
  python scripts/push_dataset_to_hf.py
  python scripts/push_dataset_to_hf.py --repo your-username/rag-eval-golden
  python scripts/push_dataset_to_hf.py --repo your-username/rag-eval-golden --private
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def push_to_hub(repo_id: str, dataset_path: Path, private: bool = False) -> None:
    """Push the golden dataset to Hugging Face Hub."""
    try:
        from datasets import Dataset
        from huggingface_hub import HfApi
    except ImportError:
        print("❌ Missing dependencies. Run: pip install datasets huggingface_hub")
        sys.exit(1)

    # Check authentication
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("⚠️  HF_TOKEN environment variable not set.")
        print("   Trying huggingface-cli login credentials...")

    # Verify dataset file exists
    if not dataset_path.exists():
        print(f"❌ Dataset file not found: {dataset_path}")
        print("   Make sure golden_dataset.jsonl exists in the project root.")
        sys.exit(1)

    # Load dataset from JSONL
    print(f"📂 Loading dataset from: {dataset_path}")
    samples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Skipping malformed JSON on line {i}: {e}")

    print(f"✅ Loaded {len(samples)} samples")

    # Validate required fields
    required_fields = {"question", "ground_truth", "reference_context"}
    for i, sample in enumerate(samples):
        missing = required_fields - set(sample.keys())
        if missing:
            print(f"⚠️  Sample {i+1} missing fields: {missing}")

    # Create HF Dataset object
    hf_dataset = Dataset.from_list(samples)
    print(f"\nDataset schema:")
    print(f"   Features: {list(hf_dataset.features.keys())}")
    print(f"   Samples: {len(hf_dataset)}")

    # Push to Hub
    visibility = "private" if private else "public"
    print(f"\nPushing to HF Hub: {repo_id} ({visibility})...")

    try:
        hf_dataset.push_to_hub(
            repo_id=repo_id,
            private=private,
            token=hf_token,
        )
        print(f"\n✅ Dataset successfully pushed!")
        print(f"   URL: https://huggingface.co/datasets/{repo_id}")
        print(f"\n📋 Load in Python:")
        print(f"   from datasets import load_dataset")
        print(f"   dataset = load_dataset('{repo_id}', split='train')")
        print(f"\n📋 Update eval_config.yaml:")
        print(f"   dataset:")
        print(f"     hf_repo: \"{repo_id}\"")
        print(f"     split: \"train\"")

    except Exception as e:
        if "401" in str(e) or "authentication" in str(e).lower():
            print("\n❌ Authentication failed!")
            print("   Solution: Run 'huggingface-cli login' or set HF_TOKEN env var")
            print("   Get a token at: https://huggingface.co/settings/tokens")
        elif "403" in str(e) or "permission" in str(e).lower():
            print(f"\n❌ Permission denied for repo: {repo_id}")
            print("   Make sure you're using your own username in the repo ID")
        else:
            print(f"\n❌ Push failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Push golden_dataset.jsonl to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/push_dataset_to_hf.py
  python scripts/push_dataset_to_hf.py --repo myusername/rag-eval-golden
  python scripts/push_dataset_to_hf.py --repo myusername/rag-eval-golden --private
        """
    )
    parser.add_argument(
        "--repo",
        default="manikbodamwad/rag-eval-golden",
        help="HF Hub repository ID (username/dataset-name)",
    )
    parser.add_argument(
        "--dataset",
        default="golden_dataset.jsonl",
        help="Path to the JSONL dataset file",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Create as private dataset (public by default — CI needs public dataset)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  Hugging Face Hub Dataset Push Script")
    print("  rag-eval golden dataset uploader")
    print("=" * 60)
    print(f"\n  Target repo: {args.repo}")
    print(f"  Dataset file: {args.dataset}")
    print(f"  Visibility: {'private' if args.private else 'public'}\n")

    if args.private:
        print("⚠️  WARNING: Private datasets require HF_TOKEN in GitHub Secrets")
        print("   CI will fail unless HF_TOKEN is set in repository secrets.")
        print()

    push_to_hub(
        repo_id=args.repo,
        dataset_path=Path(args.dataset),
        private=args.private,
    )


if __name__ == "__main__":
    main()
