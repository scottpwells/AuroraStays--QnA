"""Prepare an allowlisted deployment; --publish opens a draft PR in the existing repo."""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / ".cache/notebook/codespaces_preparation"
REPOSITORY = "scottpwells/AuroraStays--QnA"
BRANCH = "codespaces-deployment"
FILES = (
    ".gitignore", ".devcontainer/devcontainer.json", ".github/workflows/deployment-check.yml",
    "compose.yaml", "DEPLOYMENT.txt", "Full_Code_Notebook.ipynb", "SETUP_REPORT.txt",
    "requirements-notebook.txt", "requirements-notebook.lock.txt",
    "backend_files/app.py", "backend_files/rag_utils.py", "backend_files/policy_retrieval.py",
    "backend_files/deployment_manifest.json", "backend_files/.dockerignore",
    "backend_files/requirements.txt", "backend_files/Dockerfile",
    "backend_files/faiss_index/index.faiss", "backend_files/faiss_index/index.pkl",
    "frontend_files/app.py", "frontend_files/requirements.txt", "frontend_files/Dockerfile",
    "frontend_files/.dockerignore", "scripts/check-deployment.py", "scripts/start-codespaces.py",
    "scripts/smoke-deployment.py", "scripts/prepare-github.py", "scripts/run-frontend.py",
    "scripts/start-frontend.py", "scripts/verify-container.py", "scripts/verify-frontend.py",
    "tests/test_frontend.py",
)


def token():
    """Capture credentials privately. Never persist, print, or put them in command arguments."""
    for name in ("GH_TOKEN", "GITHUB_TOKEN"):
        if os.environ.get(name):
            return os.environ[name]
    gh = shutil.which("gh")
    if gh:
        result = subprocess.run([gh, "auth", "token"], capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    git = shutil.which("git")
    if git:
        result = subprocess.run([git, "credential", "fill"],
                                input="protocol=https\nhost=github.com\n\n", capture_output=True,
                                text=True, timeout=30, env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "Never"})
        fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        if fields.get("password"):
            return fields["password"]
    raise RuntimeError("GitHub sign-in unavailable. Sign in with Git credential manager or run gh auth login, then retry. Do not paste a token into the notebook.")


def github_access():
    from github import Auth, Github

    client = Github(auth=Auth.Token(token()), timeout=30, retry=0)
    try:
        repo = client.get_repo(REPOSITORY)
        if not repo.permissions.push:
            raise RuntimeError("The signed-in account cannot push to the configured repository.")
        result = {"account": client.get_user().login, "repository": repo.full_name,
                  "private": repo.private, "default_branch": repo.default_branch, "can_push": True}
        print(json.dumps(result, indent=2))
        return result
    finally:
        client.close()


def prepare():
    runpy.run_path(str(ROOT / "scripts/check-deployment.py"))["check"]()
    bundle = OUTPUT / "repository"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    known_secrets = []
    if (ROOT / "config.json").is_file():
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))
        known_secrets.extend(str(value).encode() for key, value in config.items() if "KEY" in key or "TOKEN" in key)
    known_secrets.extend(os.environ[name].encode() for name in ("GH_TOKEN", "GITHUB_TOKEN", "OPENAI_API_KEY") if os.environ.get(name))
    pattern = re.compile(rb"(?:github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|sk-(?:proj-)?[A-Za-z0-9_-]{24,})")
    artifacts = {}
    for name in FILES:
        data = (ROOT / name).read_bytes()
        if any(secret and secret in data for secret in known_secrets) or pattern.search(data):
            raise ValueError(f"Credential scan failed for {name}; file was not published.")
        artifacts[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    # Write only after the complete allowlist has passed the scan.
    for name in FILES:
        target = bundle / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    manifest = {"repository": REPOSITORY, "branch": BRANCH, "files": artifacts,
                "file_count": len(artifacts), "credential_scan_passed": True,
                "frozen_artifact_hashes_checked": True}
    (OUTPUT / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(artifacts)} scanned files for {REPOSITORY}, branch {BRANCH}.")
    print("Bundle: .cache/notebook/codespaces_preparation/repository")
    return manifest


def publish(manifest):
    from github import Auth, Github, InputGitTreeElement, UnknownObjectException

    client = Github(auth=Auth.Token(token()), timeout=60, retry=0)
    try:
        repo = client.get_repo(REPOSITORY)
        if not repo.permissions.push:
            raise RuntimeError("Push permission is required for the configured repository.")
        try:
            branch_ref = repo.get_git_ref("heads/" + BRANCH)
        except UnknownObjectException:
            branch_ref = None
        parent = repo.get_git_commit((branch_ref or repo.get_git_ref("heads/" + repo.default_branch)).object.sha)
        existing = {entry.path: entry.sha for entry in repo.get_git_tree(parent.tree.sha, recursive=True).tree if entry.type == "blob"}
        changes = []
        for name, metadata in manifest["files"].items():
            data = (OUTPUT / "repository" / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != metadata["sha256"]:
                raise ValueError(f"Prepared file changed after review: {name}")
            git_sha = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            if existing.get(name) == git_sha:
                continue
            blob = repo.create_git_blob(base64.b64encode(data).decode("ascii"), "base64")
            changes.append(InputGitTreeElement(path=name, mode="100644", type="blob", sha=blob.sha))
        if changes:
            tree = repo.create_git_tree(changes, base_tree=parent.tree)
            commit = repo.create_git_commit("Set up the verified AuroraStays application for Codespaces", tree, [parent])
            if branch_ref:
                branch_ref.edit(commit.sha, force=False)
            else:
                repo.create_git_ref("refs/heads/" + BRANCH, commit.sha)
            commit_sha = commit.sha
        else:
            commit_sha = parent.sha
        pulls = repo.get_pulls(state="open", head=f"{repo.owner.login}:{BRANCH}", base=repo.default_branch)
        pull = next(iter(pulls), None)
        if pull is None:
            pull = repo.create_pull(
                title="Run the verified policy application in GitHub Codespaces",
                body=("The repository could not launch the prepared application in Codespaces: it lacked a devcontainer, service orchestration, and the frozen FAISS files. "
                      "This adds a devcontainer that starts the backend and frontend after runtime secrets are available, includes the evaluated index, and forwards the frontend privately.\n\n"
                      "Includes the completed frontend files and notebook guidance for the existing repository. Credentials stay outside source control; no model settings, retrieval logic, or scored results change. "
                      "See DEPLOYMENT.txt for secrets, startup, diagnostics, and private notebook access.\n\n"
                      "Validation: frozen artifact SHA-256 checks, credential scan of the explicit upload list, local Compose service health and real policy retrieval without LLM calls. "
                      "The deployment workflow builds both images and repeats health/retrieval checks with a dummy API key. Codespaces cloud verification is tracked separately from local Docker verification."),
                head=BRANCH, base=repo.default_branch, draft=True)
        result = {"repository": REPOSITORY, "branch": BRANCH, "commit": commit_sha,
                  "changed_files": len(changes), "pull_request": pull.html_url, "draft": pull.draft,
                  "repository_private": repo.private, "main_unchanged": True}
        (OUTPUT / "publication.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return result
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--check-access", action="store_true")
    args = parser.parse_args()
    if args.check_access:
        github_access()
        return
    manifest = prepare()
    if args.publish:
        publish(manifest)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        from github import GithubException

        if isinstance(error, GithubException):
            print(f"GitHub operation failed (HTTP {error.status}). Check account permissions and repository access; no repository was created and visibility was not changed.", file=sys.stderr)
            sys.exit(1)
        raise
