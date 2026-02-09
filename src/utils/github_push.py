"""Push model files to a new GitHub branch via GitHub API (e.g. after finetune/retrain)."""

import base64
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

from .model_version import MODEL_VERSION_FILENAME

# Model files to push (only those present in models_dir). model.pt is always included when changed; others skipped if unchanged.
MODEL_FILES = ("model.pt", "config.json", "normalize_stats.json", MODEL_VERSION_FILENAME)
# Binary files: never skip based on content comparison (size doesn't reflect real changes)
BINARY_FILES = {"model.pt"}
GITHUB_API = "https://api.github.com"


def push_models_to_github(
    models_dir: Optional[Path] = None,
    commit_message: str = "Update model (finetune/retrain)",
    branch_prefix: str = "revision/model-update",
) -> tuple[bool, str]:
    """
    Create a new branch and push model files in a single commit using the Git Data API.
    Skips text/JSON files that are unchanged; always includes model.pt when present (binary diff unreliable).
    Branch name: {branch_prefix}-YYYYMMDD-HHMMSS (e.g. revision/model-update-20250204-143022).
    Requires env: GITHUB_TOKEN (repo scope), GITHUB_REPOSITORY (owner/repo, e.g. Uvira007/stock-predictor).
    Optional: MODELS_REPO_PATH (path in repo for model files, default "models");
              GIT_USER_NAME, GIT_EMAIL (or GIT_USER_EMAIL) for commit author/committer (e.g. Render Bot).
    Returns (success, message).
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return False, "GITHUB_TOKEN not set; skip push to GitHub."

    repo_spec = os.environ.get("GITHUB_REPOSITORY")
    if not repo_spec or "/" not in repo_spec:
        return False, "GITHUB_REPOSITORY not set or invalid (use owner/repo, e.g. Uvira007/stock-predictor)."

    owner, repo = repo_spec.strip().split("/", 1)
    repo_path = (os.environ.get("MODELS_REPO_PATH") or "models").strip().rstrip("/")

    git_name = os.environ.get("GIT_USER_NAME", "").strip()
    git_email = (os.environ.get("GIT_EMAIL") or os.environ.get("GIT_USER_EMAIL") or "").strip()
    author_committer = None
    if git_name and git_email:
        author_committer = {"name": git_name, "email": git_email}

    if models_dir is None:
        from ..config import get_settings
        models_dir = Path(get_settings().models_dir)
    models_dir = Path(models_dir).resolve()
    if not models_dir.is_dir():
        return False, f"models_dir does not exist: {models_dir}"

    def _headers() -> dict[str, str]:
        return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"}

    headers = _headers()
    repo_resp = requests.get(
        f"{GITHUB_API}/repos/{owner}/{repo}",
        headers=headers,
        timeout=30,
    )
    if repo_resp.status_code == 401:
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
        repo_resp = requests.get(
            f"{GITHUB_API}/repos/{owner}/{repo}",
            headers=headers,
            timeout=30,
        )
    if repo_resp.status_code == 404:
        return False, (
            "Repo not found or no access (404). Check GITHUB_REPOSITORY (e.g. Uvira007/stock-predictor) "
            "and that GITHUB_TOKEN has repo access."
        )
    if repo_resp.status_code != 200:
        return False, f"Could not get repo: {repo_resp.status_code} {repo_resp.text[:200]}"
    default_branch = repo_resp.json().get("default_branch") or "main"

    refs_resp = requests.get(
        f"{GITHUB_API}/repos/{owner}/{repo}/git/ref/heads/{default_branch}",
        headers=headers,
        timeout=30,
    )
    if refs_resp.status_code != 200:
        return False, (
            f"Could not get branch '{default_branch}': {refs_resp.status_code} {refs_resp.text[:200]}. "
            "Ensure token has Contents read permission."
        )
    main_sha = refs_resp.json()["object"]["sha"]

    branch_name = f"{branch_prefix}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    create_ref_resp = requests.post(
        f"{GITHUB_API}/repos/{owner}/{repo}/git/refs",
        headers=headers,
        json={"ref": f"refs/heads/{branch_name}", "sha": main_sha},
        timeout=30,
    )
    if create_ref_resp.status_code not in (200, 201):
        return False, f"Could not create branch {branch_name}: {create_ref_resp.status_code} {create_ref_resp.text[:200]}"

    # Decide which files to include: always include binary (model.pt); skip text files if unchanged
    files_to_commit: list[tuple[str, Path]] = []
    for filename in MODEL_FILES:
        path = models_dir / filename
        if not path.is_file():
            continue
        if filename in BINARY_FILES:
            files_to_commit.append((filename, path))
            continue
        # Text file: compare with current content on branch
        file_path_in_repo = f"{repo_path}/{filename}"
        get_resp = requests.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/contents/{file_path_in_repo}",
            headers=headers,
            params={"ref": branch_name},
            timeout=30,
        )
        if get_resp.status_code == 200:
            try:
                remote_content_b64 = get_resp.json().get("content") or ""
                remote_content_b64 = remote_content_b64.replace("\n", "")
                remote_bytes = base64.b64decode(remote_content_b64)
                local_bytes = path.read_bytes()
                if remote_bytes == local_bytes:
                    continue
            except (Exception, TypeError):
                pass
        files_to_commit.append((filename, path))

    if not files_to_commit:
        return False, "No model file changes to push (all files unchanged or missing)."

    # Get base tree SHA from the commit we branched from
    commit_resp = requests.get(
        f"{GITHUB_API}/repos/{owner}/{repo}/git/commits/{main_sha}",
        headers=headers,
        timeout=30,
    )
    if commit_resp.status_code != 200:
        return False, f"Could not get commit: {commit_resp.status_code} {commit_resp.text[:200]}"
    base_tree_sha = commit_resp.json()["tree"]["sha"]

    # Create blobs and build tree entries
    tree_entries: list[dict[str, Any]] = []
    for filename, path in files_to_commit:
        content = path.read_bytes()
        content_b64 = base64.b64encode(content).decode("ascii")
        blob_resp = requests.post(
            f"{GITHUB_API}/repos/{owner}/{repo}/git/blobs",
            headers=headers,
            json={"content": content_b64, "encoding": "base64"},
            timeout=60,
        )
        if blob_resp.status_code not in (200, 201):
            return False, f"Could not create blob for {filename}: {blob_resp.status_code} {blob_resp.text[:200]}"
        blob_sha = blob_resp.json()["sha"]
        file_path_in_repo = f"{repo_path}/{filename}"
        tree_entries.append({
            "path": file_path_in_repo,
            "mode": "100644",
            "type": "blob",
            "sha": blob_sha,
        })

    # Create tree
    tree_resp = requests.post(
        f"{GITHUB_API}/repos/{owner}/{repo}/git/trees",
        headers=headers,
        json={"base_tree": base_tree_sha, "tree": tree_entries},
        timeout=30,
    )
    if tree_resp.status_code not in (200, 201):
        return False, f"Could not create tree: {tree_resp.status_code} {tree_resp.text[:200]}"
    new_tree_sha = tree_resp.json()["sha"]

    # Create single commit
    commit_payload: dict[str, Any] = {
        "message": commit_message,
        "tree": new_tree_sha,
        "parents": [main_sha],
    }
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if author_committer:
        commit_payload["author"] = {**author_committer, "date": now_iso}
        commit_payload["committer"] = {**author_committer, "date": now_iso}

    commit_create_resp = requests.post(
        f"{GITHUB_API}/repos/{owner}/{repo}/git/commits",
        headers=headers,
        json=commit_payload,
        timeout=30,
    )
    if commit_create_resp.status_code not in (200, 201):
        return False, f"Could not create commit: {commit_create_resp.status_code} {commit_create_resp.text[:200]}"
    new_commit_sha = commit_create_resp.json()["sha"]

    # Update branch ref to new commit
    update_ref_resp = requests.patch(
        f"{GITHUB_API}/repos/{owner}/{repo}/git/refs/heads/{branch_name}",
        headers=headers,
        json={"sha": new_commit_sha},
        timeout=30,
    )
    if update_ref_resp.status_code != 200:
        return False, f"Could not update branch ref: {update_ref_resp.status_code} {update_ref_resp.text[:200]}"

    return True, f"Pushed {len(files_to_commit)} file(s) in one commit to branch {branch_name}"
