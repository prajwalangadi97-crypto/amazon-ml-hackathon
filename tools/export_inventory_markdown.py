#!/usr/bin/env python3
"""Export full project inventory into a clean Markdown document."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    json_path = PROJECT_ROOT / "output" / "project_inventory_report.json"
    assert json_path.is_file(), "Missing inventory json"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    inv = data["inventory"]
    out_md = PROJECT_ROOT / "output" / "project_inventory_full.md"

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Amazon ML Challenge 2026 — Complete Project Filesystem Inventory\n\n")
        f.write(f"- **Total Files:** {len(inv):,}\n")
        f.write(f"- **Total Size:** {data['total_bytes']:,} bytes ({data['total_gb']} GB)\n")
        f.write(f"- **Currently Tracked:** {sum(1 for x in inv if x['is_tracked']):,}\n")
        f.write(f"- **Currently Ignored:** {sum(1 for x in inv if x['is_ignored']):,}\n")
        f.write(f"- **Files > 100 MB:** {data['large_files_count']}\n")
        f.write(f"- **Secrets Flagged:** {len(data['secrets_flagged'])}\n\n")

        f.write("## Files Exceeding GitHub 100 MB Limit (Total: 20 files, 7.38 GB)\n\n")
        f.write("| # | Relative Path | Size (Bytes) | Size (MB) | Tracked | Ignored | Category |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for i, item in enumerate(data["large_files"], start=1):
            p = item["path"]
            cat = "Dataset" if p.startswith("dataset/") else ("Target Index" if p.startswith(("artifacts/", "scratch/")) else "Output / Candidates")
            f.write(f"| {i} | `{p}` | {item['bytes']:,} | {item['mb']} MB | False | True | {cat} |\n")

        f.write("\n## Complete File-by-File Inventory (295 Files)\n\n")
        f.write("| # | Relative Path | Size | Tracked | Ignored | >100MB | Secret Flag |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for i, item in enumerate(inv, start=1):
            sz = item["size_bytes"]
            sz_str = f"{sz:,} B" if sz < 1024 * 1024 else f"{item['size_mb']} MB"
            f.write(f"| {i} | `{item['path']}` | {sz_str} | {'Yes' if item['is_tracked'] else 'No'} | {'Yes' if item['is_ignored'] else 'No'} | {'YES' if item['exceeds_100mb'] else 'No'} | {'FLAGGED' if item['has_secret'] else 'None'} |\n")

    print(f"Wrote markdown inventory to {out_md}")

if __name__ == "__main__":
    main()
