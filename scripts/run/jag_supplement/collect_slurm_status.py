#!/usr/bin/env python3
"""Collect R1 revision Slurm states from submission manifests."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUPPLEMENT_DIR = PROJECT_ROOT / "scripts" / "run" / "jag_supplement"
DEFAULT_OUTPUT = SUPPLEMENT_DIR / "revision_status.tsv"
MANIFEST_PATTERNS = (
    "submissions_revision_matrix_*.tsv",
    "resubmissions_six_methods_*.tsv",
    "submissions_single_modality_*.tsv",
    "submissions_profile_*.txt",
    "../submissions_missing_modality_*.txt",
    "../submissions_gffloodnet_holdout_*.txt",
)
# Local Slurm job IDs are 5--7 digits. Restricting the width avoids treating
# years (2026) or manifest timestamps (20260830_141253) as job IDs.
JOB_ID_RE = re.compile(r"(?<![\w.-])(\d{5,7})(?![\w.-])")


def discover_job_ids() -> list[str]:
    job_ids: set[str] = set()
    for pattern in MANIFEST_PATTERNS:
        for path in SUPPLEMENT_DIR.glob(pattern):
            text = path.read_text(encoding="utf-8", errors="ignore")
            job_ids.update(JOB_ID_RE.findall(text))
    return sorted(job_ids, key=int)


def query_sacct(job_ids: list[str]) -> list[dict[str, str]]:
    if not job_ids:
        return []
    command = [
        "sacct",
        "-X",
        "--noheader",
        "--parsable2",
        "-j",
        ",".join(job_ids),
        "--format=JobIDRaw,JobName%64,State,ExitCode,Elapsed,Start,End",
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    fields = ["job_id", "job_name", "state", "exit_code", "elapsed", "start", "end"]
    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        values = line.rstrip("|").split("|")
        if len(values) != len(fields):
            continue
        row = dict(zip(fields, values))
        if row["job_id"] in job_ids:
            rows.append(row)
    return rows


def write_status(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["job_id", "job_name", "state", "exit_code", "elapsed", "start", "end"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    job_ids = discover_job_ids()
    rows = query_sacct(job_ids)
    write_status(rows, args.output)

    counts = Counter(row["state"].split()[0] for row in rows)
    summary = ", ".join(f"{state}={count}" for state, count in sorted(counts.items()))
    print(f"Tracked {len(rows)}/{len(job_ids)} jobs: {summary or 'no Slurm records yet'}")
    print(f"Status file: {args.output}")
    failures = [
        row
        for row in rows
        if row["state"].startswith(("FAILED", "TIMEOUT", "OUT_OF_MEMORY"))
        or (
            row["state"].startswith("CANCELLED")
            and row["exit_code"] != "0:0"
        )
    ]
    for row in failures:
        print(
            f"FAIL {row['job_id']} {row['job_name']} "
            f"{row['state']} exit={row['exit_code']}"
        )


if __name__ == "__main__":
    main()
