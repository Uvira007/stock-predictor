"""Push model files to a new GitHub Branch after fine-tune or retrain."""

import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from typing import Optional

def push_models_to_github(
        models_dir: Optional[Path] = None,
        commit_message: str = "model update (finetune/retrain)",
        branch_prefix: str = "revision/model-update",
) -> tuple[bool, str]:
    """
    Add model files, create a new branch, commit, and push to origin.
    Branch name: {branch_prefix}-YYYYMMDD-HHMMSS (e.g. revision/model-update-20260205-143022)
    Requires env: GITHUB_TOKEN (write access). Optional: GIT_USER_NAME, GIT_EMAIL.
    Returns (success, message).
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return False, "GITHUB TOKEN not set, skip model update to github"
    
    if models_dir is None:
        from ..config import get_settings
        models_dir = Path(get_settings().models_dir)
    models_dir = Path(models_dir).resolve()
    if not models_dir.is_dir():
        return False, f"models_dir does not exist: {models_dir}"
    
    try:
        repo_root = Path(
            subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output= True,
                text=True,
                check=True,
                cwd=models_dir.parent,
            ).stdout.strip()
        )
    except(subprocess.CalledProcessError, FileNotFoundError) as e:
        return False, f"Not a git repo or git unavailable: {e}"
    
    try:
        models_rel = models_dir.relative_to(repo_root)
    except ValueError:
        return False, f"models_dir {models_dir} is not under repo root {repo_root}"
    
    # Git config for commit
    for key, env_key in (("user.name", "GIT_USER_NAME"), ("user.email", "GIT_EMAIL")):
        val = os.environ.get(env_key)
        if val:
            subprocess.run(
                ["git", "config", key, val],
                capture_output=True,
                check=True,
                cwd=repo_root,
            )
    
    add_path = str(models_rel).replace("\\", "/")
    subprocess.run(
        ["git", "add", add_path],
        capture_output=True,
        check=True,
        cwd=repo_root,
    )
    status = subprocess.run(
        ["git","status", "--short"],
        capture_output=True,
        check=True,
        cwd=repo_root,
    )
    if not status.stdout.strip():
        return True, "No model file changes; nothing to push"
    
    branch_name = f"{branch_prefix}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    subprocess.run(
        ["git", "checkout", "-b", "branch_name"],
        capture_output=True,
        check=True,
        cwd=repo_root,
    )
    subprocess.run(
        ["git", "commit", "-m", commit_message],
        capture_output=True,
        check=True,
        cwd=repo_root,
    )

    # Build push URL with token (supports both https and @git origins)
    raw_url = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        capture_output=True,
        check=True,
        cwd=repo_root,
    ).stdout
    if not raw_url or not raw_url.strip():
        return False, "Could not get origin remote URL"
    stripped_url = raw_url.strip()
    remote_url: str = stripped_url if isinstance(stripped_url, str) else bytes(stripped_url).decode()
    # https:/github.com/owner/repo.git or git@github.com:owner/repo.git
    match = re.match(r"(?:https://github\.com/|git@github\.com:)([^/]+/[^/\s]+?)(?:\.git)?$", remote_url)
    if not match:
        return False, f"Could not parse the origin URL: {remote_url}"
    repo_path = match.group(1).rstrip(".git")
    push_url = f"https://x-access-token:{token}@github.com/{repo_path}.git"

    push_result = subprocess.run(
        ["git", "push", push_url, "HEAD", branch_name],
        capture_output=True,
        check=True,
        cwd=repo_root,
    )

    if push_result.returncode !=0:
        return False, f"git push failed: {push_result.stderr or push_result.stdout}"
    
    return True, f"Pushed to Branch {branch_name}"