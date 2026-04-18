## Need to pull the documents from each repo. This can be used to do just that. The test is using the readme files from langchain's github repo.

import subprocess
from pathlib import Path

# Repo information is stored in the .repoinfo file located in the root dir
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REPOINFO_PATH = REPO_ROOT / ".repoinfo"

def load_repo_config(path: Path = REPOINFO_PATH) -> list[tuple[str,str,str]]:
    """
    Parses .repoinfo into a list of (name,url,docs_path) tuples.
    Skips blank lines and lines starting with # (comments)
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — have you created your .repoinfo file?")

    repos = []
    for line_num, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3:
            raise ValueError(f".repoinfo line {line_num} malformed — expected 3 fields, got {len(parts)}: '{line}'")
        name, url, docs_path = parts
        repos.append((name, url, docs_path))

    return repos

def sparse_clone(name: str, url: str, docs_path: str) -> None:
    """
    Runs git commands via subprocess to clone only the README docs from the listed repo in the .repoinfo file
    """
    target = Path(name)
    # Exist if repo already cloned
    if target.exists():
        print(f"{name} already exists, skipping clone")
        return
    subprocess.run(["git", "clone", "--no-checkout", "--depth", "1", url, name], check=True)
    subprocess.run(["git", "sparse-checkout", "init", "--cone"], cwd=name, check=True)
    subprocess.run(["git", "sparse-checkout", "set", docs_path], cwd=name, check=True)
    subprocess.run(["git", "checkout"], cwd=name, check=True)
    print(f"{name} docs pulled")

def fetch_all_repos() -> None:
    repos = load_repo_config()
    for name,url,docs_path in repos:
        sparse_clone(name,url,docs_path)

if __name__ == "__main__":
    fetch_all_repos()

