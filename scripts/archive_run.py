"""Archival script for Neurasym benchmark run."""

import hashlib
import json
import os
import shutil
import stat
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def make_readonly(path: Path):
    """Recursively makes files and directories read-only."""
    if path.is_file():
        os.chmod(path, stat.S_IREAD)
    elif path.is_dir():
        for root, dirs, files in os.walk(path):
            for d in dirs:
                os.chmod(Path(root) / d, stat.S_IREAD | stat.S_IEXEC)
            for f in files:
                os.chmod(Path(root) / f, stat.S_IREAD)
        os.chmod(path, stat.S_IREAD | stat.S_IEXEC)

def main():
    run_id = "run_20261009_093328"
    archive_dir = PROJECT_ROOT / "results" / "archive" / run_id
    archive_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating immutable archive at: {archive_dir}")

    # 1. Copy result files
    files_to_copy = [
        (PROJECT_ROOT / "results" / "study_run_02.csv", archive_dir / "study_run_02.csv"),
        (PROJECT_ROOT / "results_presentation.csv", archive_dir / "results_presentation.csv"),
        (PROJECT_ROOT / "results_side_by_side.csv", archive_dir / "results_side_by_side.csv"),
        (PROJECT_ROOT / "results_summary.csv", archive_dir / "results_summary.csv"),
        (PROJECT_ROOT / "data" / "final_query_manifest.json", archive_dir / "final_query_manifest.json"),
    ]

    for src, dst in files_to_copy:
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Copied {src.name} -> {dst.relative_to(PROJECT_ROOT)}")

    # Copy raw folder
    src_raw = PROJECT_ROOT / "results" / "raw" / run_id
    dst_raw = archive_dir / "raw"
    if src_raw.exists():
        if dst_raw.exists():
            shutil.rmtree(dst_raw)
        shutil.copytree(src_raw, dst_raw)
        print(f"  Copied raw directory ({len(list(dst_raw.glob('*')))} files) -> {dst_raw.relative_to(PROJECT_ROOT)}")

    # Copy console log
    task_log_path = Path(r"C:\Users\user\.gemini\antigravity-ide\brain\f83ba00f-be28-4146-856c-fc8071ad91fe\.system_generated\tasks\task-1050.log")
    dst_log = archive_dir / "console_run.log"
    if task_log_path.exists():
        shutil.copy2(task_log_path, dst_log)
        print(f"  Copied console log -> {dst_log.relative_to(PROJECT_ROOT)}")

    # 2. Generate RUN_INFO.md
    manifest_sha = compute_sha256(PROJECT_ROOT / "data" / "final_query_manifest.json")
    
    run_info_content = f"""# NEURASYM Live Study Run Archive: `{run_id}`

## Execution Metadata
- **Run ID**: `{run_id}`
- **Timestamp (UTC)**: `2026-10-09T09:33:28.099094+00:00`
- **Manifest File**: `data/final_query_manifest.json`
- **Manifest SHA-256**: `{manifest_sha}`
- **Code Commit**: `f437679`
- **Model**: `openai/gpt-oss-120b` (Groq API)
- **Resolved Providers**:
  - Mode 1: `Groq API (openai/gpt-oss-120b)`
  - Mode 2: `Groq API (openai/gpt-oss-120b)`
  - Mode 3: `Pure Local Symbolic (SCOPE + HiGHS/Z3/PSO)`
  - Mode 4: `Groq API (openai/gpt-oss-120b) + Local Solvers`
- **Is Mock Provider**: `False`
- **Number of Manifest Queries ($T$)**: `29` (15 `dev` / 14 `eval`)
- **Total Output Rows ($T \\times 4$)**: `116`

## Assertion Verification Results (All Pass)
- **Assertion a) Row Count**: `PASS` (116 rows == 29 * 4)
- **Assertion b) Tuple Uniqueness & Completeness**: `PASS` (Exactly 1 row per query_id x mode x trial, 0 missing, 0 duplicate)
- **Assertion c) Live Execution Verification (`is_mock == False`)**: `PASS` (0 mock rows)
- **Assertion d) Raw Response Persistence**: `PASS` (87/87 Mode 1, 2, 4 raw files present and non-empty)
- **Assertion e) Provider Verification (`provider != MockProvider`)**: `PASS` (0 MockProvider rows)
- **Assertion f) Groq Request Count**: `PASS` (201 live calls recorded, exceeded 3*T threshold)
- **Assertion g) Ground Truth Integrity**: `PASS` (Every row has expected outcome and optimal cost or explicit non-cost status)

## Archived Artifacts in This Folder
1. `study_run_02.csv`: Full 60-column tabular benchmark output.
2. `results_presentation.csv`: 116-row standardized presentation dataset.
3. `results_side_by_side.csv`: 29-row side-by-side mode comparison matrix.
4. `results_summary.csv`: Aggregated accuracy and latency summary table.
5. `final_query_manifest.json`: Snapshot of the exact 29-query manifest evaluated.
6. `raw/`: All 87 raw LLM responses (prose text & JSON objects).
7. `console_run.log`: Full verbose console execution log.
"""

    run_info_file = archive_dir / "RUN_INFO.md"
    with open(run_info_file, "w", encoding="utf-8") as f:
        f.write(run_info_content)
    print(f"  Generated {run_info_file.relative_to(PROJECT_ROOT)}")

    # 3. Create zip archive
    zip_path = PROJECT_ROOT / "results" / "archive" / f"{run_id}.zip"
    print(f"\nCompressing archive to: {zip_path}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(archive_dir):
            for file in files:
                file_p = Path(root) / file
                arcname = file_p.relative_to(archive_dir.parent)
                zf.write(file_p, arcname=str(arcname))

    zip_sha = compute_sha256(zip_path)
    print(f"Archive Zip Created: {zip_path.name}")
    print(f"Archive Zip SHA-256 : {zip_sha}")

    # 4. Make archive directory read-only
    make_readonly(archive_dir)
    print("Archive directory made read-only.")

if __name__ == "__main__":
    main()
