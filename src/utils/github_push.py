"""Push model files to a new GitHub branch via GitHub API (e.g. after finetune/retrain)."""

import base64
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

# Model files to push (only those present in models_dir)
MODEL_FILES = ("model.pt", "config.json", "normalize_stats.json")
GITHUB_API = "https://api.github.com"


def push_models_to_github(
    models_dir: Optional[Path] = None,
    commit_message: str = "Update model (finetune/retrain)",
    branch_prefix: str = "revision/model-update",
) -> tuple[bool, str]:
    """
    Create a new branch and push model files using the GitHub Contents API.
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

    # Optional: commit author/committer (e.g. GIT_USER_NAME=render_bot, GIT_EMAIL=render@users.noreply.github.com)
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

    def _headers(auth: str) -> dict[str, str]:
        return {"Authorization": auth, "Accept": "application/vnd.github.v3+json"}

    # Try Bearer first (fine-grained PAT / GitHub App), then token (classic PAT)
    headers = _headers(f"Bearer {token}")
    repo_resp = requests.get(
        f"{GITHUB_API}/repos/{owner}/{repo}",
        headers=headers,
        timeout=30,
    )
    if repo_resp.status_code == 401:
        headers = _headers(f"token {token}")
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

    # Get commit SHA for the default branch. API uses git/ref/{ref} (singular).
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

    # Create new branch from main
    create_ref_resp = requests.post(
        f"{GITHUB_API}/repos/{owner}/{repo}/git/refs",
        headers=headers,
        json={"ref": f"refs/heads/{branch_name}", "sha": main_sha},
        timeout=30,
    )
    if create_ref_resp.status_code not in (200, 201):
        return False, f"Could not create branch {branch_name}: {create_ref_resp.status_code} {create_ref_resp.text[:200]}"

    # Upload each model file that exists
    uploaded = 0
    for filename in MODEL_FILES:
        path = models_dir / filename
        if not path.is_file():
            continue
        content = path.read_bytes()
        content_b64 = base64.b64encode(content).decode("ascii")
        file_path_in_repo = f"{repo_path}/{filename}"

        # Get current file sha on the new branch (for update); 404 means create
        get_resp = requests.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/contents/{file_path_in_repo}",
            headers=headers,
            params={"ref": branch_name},
            timeout=30,
        )
        sha = get_resp.json().get("sha") if get_resp.status_code == 200 else None

        put_payload: dict[str, Any] = {
            "message": commit_message,
            "content": content_b64,
            "branch": branch_name,
        }
        if sha:
            put_payload["sha"] = sha
        if author_committer:
            put_payload["author"] = author_committer
            put_payload["committer"] = author_committer

        put_resp = requests.put(
            f"{GITHUB_API}/repos/{owner}/{repo}/contents/{file_path_in_repo}",
            headers=headers,
            json=put_payload,
            timeout=60,
        )
        if put_resp.status_code not in (200, 201):
            return False, f"Could not upload {filename}: {put_resp.status_code} {put_resp.text[:200]}"
        uploaded += 1

    if uploaded == 0:
        return False, "No model files to push (none of model.pt, config.json, normalize_stats.json found)."
    return True, f"Pushed {uploaded} file(s) to branch {branch_name}"
