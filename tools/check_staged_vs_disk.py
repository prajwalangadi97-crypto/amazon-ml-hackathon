#!/usr/bin/env python3
"""Check that all project files are staged in Git."""

import os
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    all_disk = set()
    for dirpath, dirnames, filenames in os.walk(PROJECT_ROOT):
        rel_parts = Path(dirpath).relative_to(PROJECT_ROOT).parts
        if ".git" in rel_parts:
            continue
        for f in filenames:
            fp = Path(dirpath) / f
            rel_p = str(fp.relative_to(PROJECT_ROOT)).replace("\\", "/")
            all_disk.add(rel_p)

    raw_tracked = subprocess.check_output(["git", "ls-files"], text=True, cwd=str(PROJECT_ROOT))
    tracked = {line.strip().replace("\\", "/") for line in raw_tracked.splitlines() if line.strip()}

    raw_staged = subprocess.check_output(["git", "diff", "--name-only", "--cached"], text=True, cwd=str(PROJECT_ROOT))
    staged = {line.strip().replace("\\", "/") for line in raw_staged.splitlines() if line.strip()}

    in_git = tracked | staged

    missing = all_disk - in_git
    print(f"Total project files on disk: {len(all_disk)}")
    print(f"Total files in Git (tracked + staged): {len(in_git)}")
    print(f"Total missing from Git: {len(missing)}")

    # Break missing down into pycache/cache vs actual project files
    pycache_missing = [m for m in missing if "__pycache__" in m or ".pytest_cache" in m or m.endswith(".pyc")]
    project_missing = [m for m in missing if m not in pycache_missing]

    print(f"  Pycache / cache files (ignored by default): {len(pycache_missing)}")
    print(f"  Project files missing: {len(project_missing)}")

    for m in sorted(project_missing):
        print(f"    PROJECT FILE MISSING: {m}")

if __name__ == "__main__":
    main()
