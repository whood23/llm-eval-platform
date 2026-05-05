## Need to pull the documents from each repo. This can be used to do just that. The test is using the readme files from langchain's github repo.

import subprocess
from pathlib import Path
import shutil

# Repo information is stored in the .repoinfo file located in the root dir
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REPOINFO_PATH = REPO_ROOT / ".repoinfo"

# File type and folder filtering
ALLOWED_EXTENSIONS = {".md", ".py", ".html"}
ALLOWED_FOLDER_PATTERNS = {"doc", "docs", "example", "examples"}  # Case-insensitive matching

def load_repo_config(path: Path = REPOINFO_PATH) -> list[tuple[str,str,list[str]]]:
    """
    Parses .repoinfo into a list of (name,url,docs_paths) tuples.
    docs_paths can contain multiple paths separated by semicolons (e.g., "docs;examples").
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
        name, url, docs_paths_str = parts
        # Support multiple paths separated by semicolons
        docs_paths = [p.strip() for p in docs_paths_str.split(";") if p.strip()]
        repos.append((name, url, docs_paths))

    return repos

def filter_files(target_dir: Path) -> None:
    """
    Removes files that don't match allowed extensions (.md, .py, .html) or
    aren't in allowed folders (doc, docs, example, examples).
    Deletes entire subdirectories that don't contain relevant files.
    """
    target_path = Path(target_dir)
    
    # Recursively find and remove non-matching files
    for file_path in list(target_path.rglob("*")):
        # Skip .git directory and its contents
        if ".git" in file_path.parts:
            continue
            
        if file_path.is_file():
            # Check if file has allowed extension
            if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                file_path.unlink()
                continue
            
            # Check if file is in an allowed folder
            relative_path = file_path.relative_to(target_path)
            parts_lower = [part.lower() for part in relative_path.parts[:-1]]  # Exclude filename
            
            if not any(pattern in parts_lower for pattern in ALLOWED_FOLDER_PATTERNS):
                file_path.unlink()
    
    # Remove empty directories (skip .git directory)
    for dir_path in sorted(target_path.rglob("*"), reverse=True):
        if ".git" in dir_path.parts:
            continue
        if dir_path.is_dir() and not any(dir_path.iterdir()):
            dir_path.rmdir()

def sparse_clone(name: str, url: str, docs_paths: list[str]) -> None:
    """
    Runs git commands via subprocess to clone only specific doc/example folders from the listed repo in the .repoinfo file.
    Supports multiple paths separated by semicolons in .repoinfo.
    """
    # Ensure rawrepo directory exists at project root
    rawrepo_dir = REPO_ROOT / "rawrepo"
    rawrepo_dir.mkdir(exist_ok=True)
    
    target = rawrepo_dir / name
    # Exit if repo already cloned
    if target.exists():
        print(f"{name} already exists, skipping clone")
        return
    subprocess.run(["git", "clone", "--no-checkout", "--depth", "1", url, str(target)], check=True)
    subprocess.run(["git", "sparse-checkout", "init", "--cone"], cwd=str(target), check=True)
    subprocess.run(["git", "sparse-checkout", "set"] + docs_paths, cwd=str(target), check=True)
    subprocess.run(["git", "checkout"], cwd=str(target), check=True)
    
    # Filter files to only keep allowed types and folders
    filter_files(target)
    
    print(f"{name} docs pulled")

def fetch_all_repos() -> None:
    repos = load_repo_config()
    for name,url,docs_paths in repos:
        sparse_clone(name,url,docs_paths)

if __name__ == "__main__":
    fetch_all_repos()

