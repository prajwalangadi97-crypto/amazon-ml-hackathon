#!/usr/bin/env python3
"""Complete inventory and secret scanner for Amazon ML Challenge 2026 project.
Analyzes every file under the project root (excluding .git).
"""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SECRET_PATTERNS = [
    ("AWS Access Key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("AWS Secret Key", re.compile(r"(?i)aws_secret_access_key\s*=\s*['\"]?([a-zA-Z0-9/+=]{40})['\"]?")),
    ("Generic Private Key", re.compile(r"-----BEGIN (?:RSA|OPENSSH|DSA|EC|PGP)? PRIVATE KEY-----")),
    ("Generic Secret Token", re.compile(r"(?i)(?:api_key|apikey|secret_key|private_key|auth_token)\s*[:=]\s*['\"][a-zA-Z0-9_\-\.]{20,}['\"]")),
    ("GitHub Personal Token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[a-zA-Z0-9]{36}\b")),
]


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 80)
    print("STARTING FULL RECURSIVE PROJECT INVENTORY & AUDIT")
    print("=" * 80)

    # 1. Tracked files
    raw_tracked = subprocess.check_output(["git", "ls-files"], text=True, cwd=str(PROJECT_ROOT))
    tracked_set = {line.strip().replace("\\", "/") for line in raw_tracked.splitlines() if line.strip()}

    # 2. Gather all files
    all_files = []
    for dirpath, dirnames, filenames in os.walk(PROJECT_ROOT):
        rel_dir = os.path.relpath(dirpath, PROJECT_ROOT).replace("\\", "/")
        if rel_dir == ".git" or rel_dir.startswith(".git/"):
            continue
        for f in filenames:
            fp = Path(dirpath) / f
            rel_p = str(fp.relative_to(PROJECT_ROOT)).replace("\\", "/")
            all_files.append((rel_p, fp))

    # 3. Check ignored status via git check-ignore
    # Use -v to see matching pattern, or check individual files / paths
    ignored_set = set()
    try:
        proc = subprocess.run(
            ["git", "status", "--ignored", "--porcelain=v1"],
            stdout=subprocess.PIPE,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        for line in proc.stdout.splitlines():
            if line.startswith("!! "):
                raw_path = line[3:].strip().replace("\\", "/")
                # If directory ignored, all files inside it are ignored
                if raw_path.endswith("/"):
                    for rel_p, _ in all_files:
                        if rel_p.startswith(raw_path):
                            ignored_set.add(rel_p)
                else:
                    ignored_set.add(raw_path)
    except Exception as e:
        print(f"Error checking git ignored: {e}")

    # Also test any remaining files with check-ignore individually or in chunks
    for rel_p, _ in all_files:
        if rel_p not in ignored_set and rel_p not in tracked_set:
            res = subprocess.run(
                ["git", "check-ignore", "-q", rel_p],
                cwd=str(PROJECT_ROOT),
            )
            if res.returncode == 0:
                ignored_set.add(rel_p)

    # 4. Scan each file
    inventory = []
    secrets_flagged = []
    large_files = []
    total_bytes = 0

    for rel_p, fp in sorted(all_files, key=lambda x: x[0]):
        sz = fp.stat().st_size
        total_bytes += sz
        is_tracked = rel_p in tracked_set
        is_ignored = rel_p in ignored_set
        exceeds_100mb = sz > (100 * 1024 * 1024)

        if exceeds_100mb:
            large_files.append((rel_p, sz))

        # Secret scan
        has_secret = False
        flagged_types = []
        if sz < 20 * 1024 * 1024 and not rel_p.endswith((".pkl", ".gz", ".zip", ".png", ".jpg", ".pyc")):
            try:
                content = fp.read_text(encoding="utf-8", errors="ignore")
                for s_name, pat in SECRET_PATTERNS:
                    if pat.search(content):
                        # Filter out regex patterns in this scanner itself
                        if "SECRET_PATTERNS" in content and rel_p.endswith("generate_project_inventory.py"):
                            continue
                        has_secret = True
                        flagged_types.append(s_name)
            except Exception:
                pass

        if has_secret:
            secrets_flagged.append({"path": rel_p, "types": flagged_types})

        inventory.append(
            {
                "path": rel_p,
                "size_bytes": sz,
                "size_mb": round(sz / (1024 * 1024), 2),
                "is_tracked": is_tracked,
                "is_ignored": is_ignored,
                "exceeds_100mb": exceeds_100mb,
                "has_secret": has_secret,
                "secret_types": flagged_types,
            }
        )

    # 5. Verify EXP-009 hashes
    exp009_matching = PROJECT_ROOT / "output" / "exp009" / "matching_results.tsv"
    exp009_candidate = PROJECT_ROOT / "output" / "exp009" / "candidate_pairs.tsv"

    matching_hash = compute_sha256(exp009_matching) if exp009_matching.is_file() else "MISSING"
    candidate_hash = compute_sha256(exp009_candidate) if exp009_candidate.is_file() else "MISSING"

    expected_matching = "91b30bade4b9e8a12b0b16201edd630ebe6cea20696e39e89c8c4231e3cb636e"
    expected_candidate = "b02ea25c38dd7446b10502e060311542e44be1fc523ab9851bbb1033246f2bd7"

    print(f"Total Files Scanned      : {len(inventory):,}")
    print(f"Total Project Size       : {total_bytes:,} bytes ({total_bytes / (1024*1024*1024):.2f} GB)")
    print(f"Currently Tracked Files  : {sum(1 for x in inventory if x['is_tracked']):,}")
    print(f"Currently Ignored Files  : {sum(1 for x in inventory if x['is_ignored']):,}")
    print(f"Files Exceeding 100MB    : {len(large_files)}")
    print(f"Secrets Flagged          : {len(secrets_flagged)}")

    print("\nEXP-009 PRODUCTION HASH VERIFICATION:")
    print(f"  matching_results.tsv : {matching_hash}")
    print(f"    Expected            : {expected_matching}")
    print(f"    Match               : {matching_hash == expected_matching}")
    print(f"  candidate_pairs.tsv  : {candidate_hash}")
    print(f"    Expected            : {expected_candidate}")
    print(f"    Match               : {candidate_hash == expected_candidate}")

    # Output detailed JSON
    out_json = PROJECT_ROOT / "output" / "project_inventory_report.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "total_files": len(inventory),
        "total_bytes": total_bytes,
        "total_gb": round(total_bytes / (1024 * 1024 * 1024), 2),
        "large_files_count": len(large_files),
        "large_files": [{"path": p, "bytes": s, "mb": round(s / (1024 * 1024), 2)} for p, s in large_files],
        "secrets_flagged": secrets_flagged,
        "exp009_matching_sha256": matching_hash,
        "exp009_candidate_sha256": candidate_hash,
        "exp009_hashes_verified": (matching_hash == expected_matching and candidate_hash == expected_candidate),
        "inventory": inventory,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\nSaved inventory report to {out_json}")


if __name__ == "__main__":
    main()
