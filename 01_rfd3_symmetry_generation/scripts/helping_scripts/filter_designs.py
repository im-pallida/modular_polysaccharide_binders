#!/usr/bin/env python3
"""
Stage 01 geometry filter: sorts generated structures into passed / rejected.

Reads what the diffusion step archived and re-files it one level deeper:

    outputs_clean/<experiment>/<group>.tar.gz           <- diffusion writes
    sorted_clean/<experiment>/passed/<group>.tar.gz     <- this writes
    sorted_clean/<experiment>/rejected/<group>.tar.gz

A structure keeps the group it was generated under, so nothing is renamed
or regrouped on the way through.

Rule (unchanged from 15082026_03_filtering.py)
    n_chainbreaks == 0                  else REJECTED (chainbreaks)
    non_loop_fraction > 0.7             else REJECTED (loopy_structure)
    radius_of_gyration                  recorded only, never rejects"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import tarfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# job_paths.py is shared by every stage and lives at the repo root, so a
# fix to the naming rules reaches all of them at once.
# helping_scripts/ -> scripts/ -> <stage>/ -> <repo root>/common
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import job_paths as jp  # noqa: E402
from archives import (  # noqa: E402
    ArchiveError,
    SortResult,
    extract_member,
    group_members_by_protein,
    load_table,
    read_archive_members,
    regroup_and_archive,
    save_table,
)

STAGE = Path(__file__).resolve().parents[2]

NON_LOOP_CUTOFF = 0.7
CHECKPOINT_SIZE = 100

# tarfile grew an extraction `filter` in 3.12 and makes it mandatory in 3.14.
# Passing it unconditionally would break older interpreters on the cluster.
_EXTRACT_KWARGS = {"filter": "data"} if sys.version_info >= (3, 12) else {}

RESULT_FIELDS = [
    "protein_id",
    "experiment_name",
    "time_stamp",
    "n_chainbreaks",
    "non_loop_fraction",
    "non_loop_cutoff",
    "radius_of_gyration",
    "status",
    "rejection_reason",
    "error_reason",
]


class Stage01InputError(Exception):
    """Malformed/missing input or metric -- a pipeline error, not a biological
    rejection."""


def _log(*parts: object) -> None:
    print(*parts, flush=True)

@dataclass
class FilterReport:
    scanned: int = 0
    skipped: int = 0
    counts: Dict[str, int] = field(default_factory=dict)
    sorted_archives: List[SortResult] = field(default_factory=list)
    tables: List[Path] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """REJECTED is a normal outcome; ERROR means a structure fell out of
        the pipeline without reaching stage 02, which is not."""
        return self.counts.get("ERROR", 0) == 0

    def summary(self) -> str:
        if not self.scanned:
            return f"filter: nothing new ({self.skipped} already filtered)"
        parts = ", ".join(f"{key}={value}" for key, value in sorted(self.counts.items()))
        return (f"filter: {self.scanned} evaluated ({parts}), "
                f"{self.skipped} skipped, {len(self.sorted_archives)} archive(s) updated")


# Stage 1. I/O.


def archives_to_scan(stage: Path, experiment: Optional[str] = None) -> Dict[str, List[Path]]:
    """{experiment: [tar.gz, ...]} under outputs_clean/."""
    names = [experiment] if experiment else jp.clean_experiments(stage)
    found: Dict[str, List[Path]] = {}
    for name in names:
        directory = jp.clean_dir(stage, name)
        if not directory.is_dir():
            raise Stage01InputError(f"no such experiment under outputs_clean/: {directory}")
        tars = sorted(directory.glob("*.tar.gz"))
        if tars:
            found[name] = tars
    return found

def route_files(json_path: Path, structure_path: Path, target_dir: Path) -> None:
    """Move an evaluated pair into sorted_raw/<experiment>/<outcome>/."""
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path.rename(target_dir / json_path.name)
    structure_path.rename(target_dir / structure_path.name)

# Step 2. Metrics and criteria.


def _table_sort_key(row: Dict[str, str]) -> tuple:
    """Worst structures first: ERROR rows, then ascending non_loop_fraction."""
    try:
        non_loop = float(row.get("non_loop_fraction", ""))
    except ValueError:
        non_loop = float("-inf")
    return (row.get("status") != "ERROR", non_loop, row.get("protein_id", ""))


def numeric_metric(metrics: Dict[str, Any], name: str) -> float:
    if name not in metrics:
        raise Stage01InputError(f"missing_metric:{name}")
    value = metrics[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Stage01InputError(f"non_numeric_metric:{name}")
    value = float(value)
    if not math.isfinite(value):
        raise Stage01InputError(f"non_finite_metric:{name}")
    return value


def read_metrics(json_path: Path) -> Dict[str, float]:
    try:
        with json_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise Stage01InputError(f"malformed_json:line_{exc.lineno}:column_{exc.colno}") from exc
    except OSError as exc:
        raise Stage01InputError(f"unreadable_json:{exc}") from exc

    metrics = payload.get("metrics") if isinstance(payload, dict) else None
    if not isinstance(metrics, dict):
        raise Stage01InputError("missing_metrics_object")

    return {
        "n_chainbreaks": numeric_metric(metrics, "n_chainbreaks"),
        "non_loop_fraction": numeric_metric(metrics, "non_loop_fraction"),
        "radius_of_gyration": numeric_metric(metrics, "radius_of_gyration"),
    }


def decide_status(n_chainbreaks: float, non_loop_fraction: float) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    if n_chainbreaks != 0:
        reasons.append("chainbreaks")
    if not (non_loop_fraction > NON_LOOP_CUTOFF):
        reasons.append("loopy_structure")
    return ("REJECTED" if reasons else "PASSED"), reasons


def build_row(protein_id: str, experiment_name: str, metrics: Dict[str, float],
              status: str, reasons: Sequence[str], error_reason: str = "") -> Dict[str, str]:
    def number(name: str, spec: str) -> str:
        return format(metrics[name], spec) if name in metrics else ""

    return {
        "protein_id": protein_id,
        "experiment_name": experiment_name,
        "time_stamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_chainbreaks": number("n_chainbreaks", ".6g"),
        "non_loop_fraction": number("non_loop_fraction", ".10g"),
        "non_loop_cutoff": f"{NON_LOOP_CUTOFF:.10g}",
        "radius_of_gyration": number("radius_of_gyration", ".10g"),
        "status": status,
        "rejection_reason": ";".join(reasons),
        "error_reason": error_reason,
    }


# Step 3. Orchestrator.


def _scan_archive(stage: Path, experiment: str, tar_path: Path,
                  by_id: Dict[str, Dict[str, str]], table_path: Path,
                  report: FilterReport) -> None:
    """Evaluate every not-yet-filtered structure in one tarball."""
    scratch_dir = jp.sorted_scratch_dir(stage, experiment)

    with tarfile.open(tar_path, "r:gz") as archive:
        groups = group_members_by_protein(archive.getnames())

        for protein_id, members in sorted(groups.items()):
            if protein_id in by_id:
                report.skipped += 1
                continue

            def record(status: str, metrics: Dict[str, float],
                       reasons: Sequence[str] = (), error: str = "") -> None:
                by_id[protein_id] = build_row(
                    protein_id, experiment, metrics, status, reasons, error
                )
                report.scanned += 1
                report.counts[status] = report.counts.get(status, 0) + 1

            try:
                jp.group_key_from_job_name(protein_id)
            except ValueError as exc:
                record("ERROR", {}, error="unparseable_job_name")
                _log(f"  {protein_id}: ERROR (unparseable_job_name -- {exc})")
                continue

            json_member = members.get("json")
            structure_member = members.get("structure")
            if json_member is None:
                record("ERROR", {}, error="missing_metadata_json")
                _log(f"  {protein_id}: ERROR (missing_metadata_json)")
                continue
            if structure_member is None:
                record("ERROR", {}, error="missing_structure_file")
                _log(f"  {protein_id}: ERROR (missing_structure_file)")
                continue

            json_path = extract_member(archive, json_member, scratch_dir)
            structure_path = extract_member(archive, structure_member, scratch_dir)

            try:
                metrics = read_metrics(json_path)
            except Stage01InputError as exc:
                record("ERROR", {}, error=str(exc))
                _log(f"  {protein_id}: ERROR ({exc})")
                json_path.unlink(missing_ok=True)
                structure_path.unlink(missing_ok=True)
                continue

            status, reasons = decide_status(
                metrics["n_chainbreaks"], metrics["non_loop_fraction"]
            )
            record(status, metrics, reasons)
            _log(f"  {protein_id}: n_chainbreaks={metrics['n_chainbreaks']:.4g} "
                 f"non_loop_fraction={metrics['non_loop_fraction']:.4g} "
                 f"radius_of_gyration={metrics['radius_of_gyration']:.4g} -> {status}"
                 + (f" ({';'.join(reasons)})" if reasons else ""))

            outcome = "passed" if status == "PASSED" else "rejected"
            route_files(json_path, structure_path,
                        jp.sorted_raw_dir(stage, experiment, outcome))

            if report.scanned % CHECKPOINT_SIZE == 0:
                save_table(by_id, table_path, RESULT_FIELDS, _table_sort_key)
                _log(f"  [checkpoint] {report.scanned} evaluated, table saved")


def run_filter(stage: Path, experiment: Optional[str] = None) -> FilterReport:
    """Filter everything under outputs_clean/ that is not already in a table."""
    report = FilterReport()
    by_experiment = archives_to_scan(stage, experiment)
    if not by_experiment:
        _log(f"[filter] nothing to scan under {jp.clean_root(stage)}")
        return report

    _log(f"[filter] criteria: n_chainbreaks == 0, non_loop_fraction > {NON_LOOP_CUTOFF} "
         f"(radius_of_gyration recorded only)")

    for experiment_name, tar_paths in by_experiment.items():
        table_path = jp.results_table_path(stage, experiment_name)
        by_id = load_table(
            table_path,
            legacy_path=jp.legacy_results_table_path(stage),
            experiment=experiment_name,
        )
        before = len(by_id)
        _log(f"[filter] {experiment_name}: {len(tar_paths)} archive(s), "
             f"{before} structure(s) already filtered")

        for tar_path in tar_paths:
            _scan_archive(stage, experiment_name, tar_path, by_id, table_path, report)

        if len(by_id) == before:
            continue

        save_table(by_id, table_path, RESULT_FIELDS, _table_sort_key)
        report.tables.append(table_path)
        for result in regroup_and_archive(stage, experiment_name):
            report.sorted_archives.append(result)
            _log(f"[filter] {result.outcome}/{result.group_key}: +{result.added} new, "
                 f"{result.total} member(s) total -> {result.archive}")

    return report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=Path, default=STAGE,
                        help=f"stage root directory (default: {STAGE})")
    parser.add_argument("--experiment", default=None,
                        help="only scan this one experiment (default: all of them)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = run_filter(args.stage.resolve(), args.experiment)
    except (OSError, ValueError, ArchiveError, Stage01InputError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    _log(f"[done] {report.summary()}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

