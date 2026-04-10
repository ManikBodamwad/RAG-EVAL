#!/usr/bin/env python3
"""
push_to_github.py — Push all project files to GitHub via the REST API.

Usage:
    python push_to_github.py --token YOUR_GITHUB_TOKEN

Get a token at: https://github.com/settings/tokens/new
Required scope: repo (check the 'repo' checkbox)

This script:
1. Reads every file in the project directory
2. Pushes them all to GitHub in a single batch commit
3. No git installation or terminal navigation required
"""

import argparse
import base64
import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

REPO_OWNER = "ManikBodamwad"
REPO_NAME = "RAG-EVAL"
BRANCH = "main"
API_BASE = "https://api.github.com"

# Files and directories to skip
SKIP_PATTERNS = {
    ".DS_Store", "__pycache__", ".venv", "venv", "env",
    ".git", ".eggs", "dist", "build", "*.egg-info",
    "eval_report.json", "pr_comment.md", ".cache",
    "faiss_index", "node_modules", ".pytest_cache",
}

def should_skip(path: Path) -> bool:
    for part in path.parts:
        if part in SKIP_PATTERNS:
            return True
        if part.endswith(".egg-info") or part.endswith(".pyc"):
            return True
    return False

def is_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
        return b"\x00" in chunk
    except Exception:
        return True

def github_request(url: str, method: str, token: str, data: dict = None):
    """Make a GitHub API request."""
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
        "User-Agent": "rag-eval-pusher/1.0",
    }
    body = json.dumps(data).encode("utf-8") if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8")), resp.status
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        try:
            error_data = json.loads(error_body)
        except Exception:
            error_data = {"message": error_body}
        return error_data, e.code

def get_file_sha(path_in_repo: str, token: str) -> str | None:
    """Get the SHA of an existing file (needed for updates)."""
    url = f"{API_BASE}/repos/{REPO_OWNER}/{REPO_NAME}/contents/{path_in_repo}"
    data, status = github_request(url, "GET", token)
    if status == 200 and isinstance(data, dict):
        return data.get("sha")
    return None

def push_file(local_path: Path, repo_path: str, token: str, dry_run: bool = False) -> bool:
    """Push a single file to GitHub."""
    try:
        content = local_path.read_bytes()
        encoded = base64.b64encode(content).decode("utf-8")
    except Exception as e:
        print(f"  ⚠️  Could not read {local_path}: {e}")
        return False

    if dry_run:
        print(f"  [DRY RUN] Would push: {repo_path}")
        return True

    # Check if file already exists (to get SHA for update)
    sha = get_file_sha(repo_path, token)

    url = f"{API_BASE}/repos/{REPO_OWNER}/{REPO_NAME}/contents/{repo_path}"
    payload = {
        "message": f"chore: add {repo_path}",
        "content": encoded,
        "branch": BRANCH,
    }
    if sha:
        payload["sha"] = sha

    data, status = github_request(url, "PUT", token, payload)

    if status in (200, 201):
        action = "Updated" if sha else "Created"
        print(f"  ✅ {action}: {repo_path}")
        return True
    else:
        msg = data.get("message", str(data)) if isinstance(data, dict) else str(data)
        print(f"  ❌ Failed {repo_path}: HTTP {status} — {msg}")
        return False

def verify_token(token: str) -> str | None:
    """Verify the token works and return the authenticated username."""
    data, status = github_request(f"{API_BASE}/user", "GET", token)
    if status == 200:
        return data.get("login")
    return None

def check_repo_exists(token: str) -> bool:
    """Check if the target repository exists."""
    url = f"{API_BASE}/repos/{REPO_OWNER}/{REPO_NAME}"
    data, status = github_request(url, "GET", token)
    return status == 200

def main():
    parser = argparse.ArgumentParser(
        description="Push RAG EVAL project to GitHub via API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python push_to_github.py --token ghp_your_token_here
  python push_to_github.py --token ghp_your_token_here --dry-run

Get a token at: https://github.com/settings/tokens/new
Required scope: repo
        """
    )
    parser.add_argument("--token", required=True, help="GitHub Personal Access Token")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be pushed without actually pushing")
    args = parser.parse_args()

    print("=" * 65)
    print("  RAG-EVAL GitHub Push Script")
    print(f"  Target: https://github.com/{REPO_OWNER}/{REPO_NAME}")
    print("=" * 65)

    # Verify token
    print("\n🔑 Verifying GitHub token...")
    username = verify_token(args.token)
    if not username:
        print("❌ Token verification failed. Check your token and try again.")
        print("   Get a token at: https://github.com/settings/tokens/new")
        sys.exit(1)
    print(f"  ✅ Authenticated as: {username}")

    # Check repo exists
    print(f"\n📦 Checking repository {REPO_OWNER}/{REPO_NAME}...")
    if not check_repo_exists(args.token):
        print(f"❌ Repository not found: https://github.com/{REPO_OWNER}/{REPO_NAME}")
        print("   Create it at: https://github.com/new")
        sys.exit(1)
    print(f"  ✅ Repository found")

    # Find project root (same directory as this script)
    project_root = Path(__file__).parent.parent
    print(f"\n📂 Scanning project: {project_root}")

    # Collect all files to push
    files_to_push = []
    for local_path in sorted(project_root.rglob("*")):
        if not local_path.is_file():
            continue
        if should_skip(local_path):
            continue
        # Skip this script itself
        if local_path.name == "push_to_github.py":
            continue

        # Compute relative path for GitHub
        try:
            repo_path = local_path.relative_to(project_root).as_posix()
        except ValueError:
            continue

        files_to_push.append((local_path, repo_path))

    print(f"  Found {len(files_to_push)} files to push\n")

    if args.dry_run:
        print("🔍 DRY RUN — Files that would be pushed:")
        for _, repo_path in files_to_push:
            print(f"  → {repo_path}")
        print(f"\nTotal: {len(files_to_push)} files")
        return

    # Push all files
    print(f"🚀 Pushing {len(files_to_push)} files to GitHub...")
    print("-" * 65)

    success_count = 0
    fail_count = 0

    for i, (local_path, repo_path) in enumerate(files_to_push, 1):
        print(f"[{i:2d}/{len(files_to_push)}] Pushing: {repo_path}")
        ok = push_file(local_path, repo_path, args.token, dry_run=args.dry_run)
        if ok:
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 65)
    print(f"  Push Complete!")
    print(f"  ✅ Succeeded: {success_count} files")
    if fail_count:
        print(f"  ❌ Failed:    {fail_count} files")
    print(f"\n  🔗 View your repo:")
    print(f"     https://github.com/{REPO_OWNER}/{REPO_NAME}")
    print("=" * 65)

    if fail_count > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
