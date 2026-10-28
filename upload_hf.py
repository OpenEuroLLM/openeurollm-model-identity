#!/usr/bin/env python3
"""Upload the validated dataset repository to Hugging Face."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


ROOT = Path(__file__).resolve().parent
DEFAULT_REPO_ID = "openeurollm/openeurollm-model-identity"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    subprocess.run([sys.executable, str(ROOT / "validate_dataset.py")], check=True)
    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    release_files = [
        "README.md",
        "identity_facts.json",
        "manifest.json",
        "validation_report.json",
        "data/train.jsonl",
        "data/validation.jsonl",
    ]
    api.create_commit(
        repo_id=args.repo_id,
        repo_type="dataset",
        operations=[
            CommitOperationAdd(path_in_repo=relative, path_or_fileobj=ROOT / relative)
            for relative in release_files
        ],
        commit_message=f"Publish OpenEuroLLM multilingual model identity dataset v{json.loads((ROOT / 'manifest.json').read_text())['version']}",
    )
    print(f"Uploaded dataset to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
