#!/usr/bin/env python3
"""
Stage 04, one-off: file the monomer winners before the stage is rewritten.

    tables/stage_04_results.csv         <- read: which sequences passed
    outputs/<sequence_id>_monomer/      <- read: the AF3 job directory
    outputs_clean/passed/<experiment>/<group>.tar.gz   <- written
    tables/stage_04_monomer_passed.csv  <- written: just the passing rows

The old compute_rmsd filed a sequence only if its monomer AND its dimer both
passed. No dimer ever did, so nothing was ever saved. This applies the monomer
test alone and ignores the dimer entirely -- only <sequence_id>_monomer/ is
archived.

The pass test is recomputed from monomer_rmsd and monomer_matched_fraction
rather than read from the monomer_pass column, so the thresholds are visible
here and adjustable from the command line. It cross-checks itself against that
column and reports any disagreement, which would mean the table was written
with different thresholds than these.

Nothing is deleted or moved: outputs/ is left exactly as it is.

Usage:
    ./save_monomer_passed.py --stage ~/1cbh_clear/04_alphafold --dry-run
    ./save_monomer_passed.py --stage ~/1cbh_clear/04_alphafold
    ./save_monomer_passed.py --stage ... --max-rmsd 3.0 --min-fraction 0.9
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MAX_RMSD = 5.0        # monomer_rmsd must be BELOW this
MIN_FRACTION = 0.8    # monomer_matched_fraction must be ABOVE this


def _log(*parts: object) -> None:
    print(*parts, flush=True)


def monomer_passes(row: Dict[str, str], max_rmsd: float, min_fraction: float) -> Optional[bool]:
    """True/False, or None if the row has no usable monomer numbers."""
    try:
        rmsd = float(row["monomer_rmsd"])
        fraction = float(row["monomer_matched_fraction"])
    except (KeyError, TypeError, ValueError):
        return None
    return rmsd < max_rmsd and fraction > min_fraction


def archive_group(members: List[Tuple[Path, str]], archive_path: Path) -> int:
    """Write one group's monomer directories to <group>.tar.gz.

    Built to a temp file and moved into place, so re-running never leaves a
    half-written archive where a good one used to be.
    """
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = archive_path.with_name(f".{archive_path.name}.tmp{os.getpid()}")
    try:
        with tarfile.open(temp_path, "w:gz") as out:
            for source_dir, arcname in members:
                out.add(source_dir, arcname=arcname)
        os.replace(temp_path, archive_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return len(members)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=Path, required=True,
                        help="the 04_alphafold directory")
    parser.add_argument("--max-rmsd", type=float, default=MAX_RMSD)
    parser.add_argument("--min-fraction", type=float, default=MIN_FRACTION)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be filed; write nothing")
    args = parser.parse_args(argv)

    stage = args.stage.expanduser().resolve()
    table_path = stage / "tables" / "stage_04_results.csv"
    outputs_dir = stage / "outputs"
    passed_root = stage / "outputs_clean" / "passed"

    if not table_path.is_file():
        print(f"ERROR: no results table at {table_path}", file=sys.stderr)
        return 2
    if not outputs_dir.is_dir():
        print(f"ERROR: no outputs directory at {outputs_dir}", file=sys.stderr)
        return 2

    with table_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    by_group: Dict[Tuple[str, str], List[Tuple[Path, str]]] = defaultdict(list)
    kept_rows: List[Dict[str, str]] = []
    missing: List[str] = []
    unusable = 0
    disagreements = 0

    for row in rows:
        verdict = monomer_passes(row, args.max_rmsd, args.min_fraction)
        if verdict is None:
            unusable += 1
            continue
        # Cross-check against what the table itself recorded. A mismatch means
        # the table was written under different thresholds than these.
        recorded = str(row.get("monomer_pass", "")).strip().lower()
        if recorded in ("true", "false") and (recorded == "true") != verdict:
            disagreements += 1
        if not verdict:
            continue

        sequence_id = row["sequence_id"]
        job_name = f"{sequence_id}_monomer"
        job_dir = outputs_dir / job_name
        if not job_dir.is_dir():
            missing.append(job_name)
            continue
        by_group[(row["experiment"], row["group"])].append((job_dir, job_name))
        kept_rows.append(row)

    total = sum(len(members) for members in by_group.values())
    _log(f"[table] {len(rows)} row(s) from {table_path.name}")
    _log(f"[filter] monomer_rmsd < {args.max_rmsd} and "
         f"monomer_matched_fraction > {args.min_fraction}: "
         f"{len(kept_rows) + len(missing)} pass, {total} have a job directory")
    if unusable:
        _log(f"[filter] {unusable} row(s) had no usable monomer numbers, skipped")
    if disagreements:
        _log(f"[filter] WARNING: {disagreements} row(s) disagree with the table's own "
             f"monomer_pass column -- the table was written with other thresholds")
    if missing:
        _log(f"[filter] {len(missing)} passing sequence(s) have no directory under "
             f"outputs/, e.g. {missing[:3]}")
    if not total:
        _log("[done] nothing to file")
        return 0

    for (experiment, group), members in sorted(by_group.items()):
        archive_path = passed_root / experiment / f"{group}.tar.gz"
        if args.dry_run:
            _log(f"[dry-run] {len(members):>3} monomer(s) -> {archive_path}")
            continue
        replacing = " (replacing existing)" if archive_path.exists() else ""
        count = archive_group(members, archive_path)
        size_mb = archive_path.stat().st_size / (1024 * 1024)
        _log(f"[write] {count:>3} monomer(s) -> {archive_path} "
             f"({size_mb:.1f} MiB){replacing}")

    kept_table = stage / "tables" / "stage_04_monomer_passed.csv"
    if args.dry_run:
        _log(f"[dry-run] would write {len(kept_rows)} row(s) -> {kept_table}")
    else:
        with kept_table.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept_rows)
        _log(f"[write] {len(kept_rows)} row(s) -> {kept_table}")

    _log(f"[done] {total} monomer(s) across {len(by_group)} group(s); "
         f"outputs/ untouched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
