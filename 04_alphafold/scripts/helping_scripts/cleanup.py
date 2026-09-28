#!/usr/bin/env python3
"""
Stage 04, last step: delete what the archives already hold.

    outputs/<sequence_id>_ligand/                   <- removed
    af3_runs/<sequence_id>_ligand/                  <- removed
    job_runs/af3/<experiment>/<group>/<job>.json    <- removed

Nothing is deleted on the strength of the results table alone. A sequence is
only cleaned when BOTH are true:

    it has a row in tables/stage_04_results_<experiment>.csv, and
    its files are inside sorted_clean/<experiment>/<outcome>/<group>.tar.gz

The archive is opened and the member names checked, every time. A row in a table
is a claim; a member in the tarball is the structure itself.

This is safe to run because af3_prepare and run_af3 treat a scored sequence as
folded whether or not outputs/ still exists -- so a cleaned sequence is skipped
rather than refolded. Before that change, deleting outputs/ meant refolding
everything: an hour on a workstation, days on the cluster.

Scratch belonging to jobs that FAILED is left alone: there is no table row for
them, so there is nothing to inspect them against, and a failed run is exactly
when you want the inputs still on disk.

Usage:
    ./cleanup.py --stage ~/1cbh_clear/04_alphafold --dry-run
    ./cleanup.py --stage ~/1cbh_clear/04_alphafold
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import job_paths as jp  # noqa: E402
from archives import load_table  # noqa: E402

STAGE = Path(__file__).resolve().parents[2]


def _log(*parts: object) -> None:
    print(*parts, flush=True)


@dataclass
class CleanReport:
    removed: int = 0
    kept_unscored: int = 0
    kept_unarchived: int = 0
    bytes_freed: int = 0
    paths: List[Path] = field(default_factory=list)

    def summary(self) -> str:
        freed = self.bytes_freed / (1024 ** 3)
        parts = [f"clean: {self.removed} sequence(s) cleared ({freed:.2f} GiB)"]
        if self.kept_unscored:
            parts.append(f"{self.kept_unscored} not scored yet")
        if self.kept_unarchived:
            parts.append(f"{self.kept_unarchived} not in an archive")
        return ", ".join(parts)


def archived_jobs(stage: Path, experiment: str,
                  suffix: str = jp.LIGAND_SUFFIX) -> Set[str]:
    """Job names whose files are actually inside a sorted_clean archive.

    Read from the archives themselves rather than inferred from the table, so a
    table written before an archive was updated cannot authorise a deletion.
    """
    found: Set[str] = set()
    directories = [jp.sorted_clean_dir(stage, experiment, outcome)
                   for outcome in jp.OUTCOMES]
    # A stage with no pass/fail split archives straight into outputs_clean/.
    directories.append(jp.clean_dir(stage, experiment))
    for directory in directories:
        if not directory.is_dir():
            continue
        for archive_path in sorted(directory.glob("*.tar.gz")):
            with tarfile.open(archive_path, "r:gz") as archive:
                for name in archive.getnames():
                    parts = (name[2:] if name.startswith("./") else name).split("/")
                    # <protein_id>/<sequence_id>_monomer/...
                    if len(parts) > 2 and parts[1].endswith(f"_{suffix}"):
                        found.add(parts[1])
    return found


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def clean_experiment(stage: Path, experiment: str, report: CleanReport,
                     dry_run: bool) -> None:
    table = jp.results_table_path(stage, experiment)
    scored: Dict[str, dict] = load_table(table, key="sequence_id")
    suffix = jp.LIGAND_SUFFIX
    if not scored:
        _log(f"[clean] {experiment}: nothing scored yet, nothing to clear")
        return

    archived = archived_jobs(stage, experiment, suffix)
    _log(f"[clean] {experiment}: {len(scored)} scored, "
         f"{len(archived)} present in sorted_clean archives")

    for sequence_id, row in sorted(scored.items()):
        job_name = jp.af3_job_name(sequence_id, suffix)
        if job_name not in archived:
            report.kept_unarchived += 1
            continue

        group_key = row.get("group", "")
        targets = [
            jp.af3_output_dir(stage, job_name),
            jp.af3_run_dir(stage, job_name),
        ]
        if group_key:
            targets.append(
                jp.af3_json_path(stage, experiment, group_key, sequence_id, suffix)
            )
        present = [path for path in targets if path.exists()]
        if not present:
            continue

        size = sum(directory_size(path) for path in present)
        if dry_run:
            _log(f"[dry-run] {job_name}: would remove {len(present)} path(s), "
                 f"{size / (1024 ** 2):.1f} MiB")
        else:
            for path in present:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        report.removed += 1
        report.bytes_freed += size
        report.paths.extend(present)

    # Sequences that were folded but never scored keep everything they have.
    outputs_root = stage / "outputs"
    if outputs_root.is_dir():
        for job_dir in outputs_root.iterdir():
            if not job_dir.is_dir():
                continue
            name = job_dir.name
            ending = f"_{suffix}"
            sequence_id = (name[: -len(ending)]
                           if name.endswith(ending) else name)
            if sequence_id not in scored:
                report.kept_unscored += 1


def remove_empty_dirs(stage: Path, dry_run: bool) -> int:
    """Tidy the directories the deletions emptied. Never removes a stage root."""
    removed = 0
    for root_name in ("outputs", "af3_runs", "job_runs"):
        root = stage / root_name
        if not root.is_dir():
            continue
        for directory in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if directory.is_dir() and not any(directory.iterdir()):
                if not dry_run:
                    directory.rmdir()
                removed += 1
    return removed


def run_cleanup(stage: Path, experiment: Optional[str] = None,
                dry_run: bool = False) -> CleanReport:
    report = CleanReport()
    names = [experiment] if experiment else jp.inputs_experiments(stage)
    if not names:
        _log(f"[clean] nothing under {jp.inputs_root(stage)}")
        return report
    for experiment_name in names:
        clean_experiment(stage, experiment_name, report, dry_run)
    emptied = remove_empty_dirs(stage, dry_run)
    if emptied:
        _log(f"[clean] {emptied} empty director{'y' if emptied == 1 else 'ies'} "
             f"{'would be removed' if dry_run else 'removed'}")
    return report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would go; delete nothing")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    report = run_cleanup(args.stage.resolve(), args.experiment, args.dry_run)
    _log(f"[done] {report.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
