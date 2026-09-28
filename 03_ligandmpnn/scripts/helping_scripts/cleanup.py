#!/usr/bin/env python3
"""
Stage 03, last step: delete what the archives already hold.

    job_runs/ligandmpnn/<experiment>/<group>/out/<protein_id>/   <- removed
    job_runs/ligandmpnn/<experiment>/<group>/_preflight/         <- removed
    job_runs/ligandmpnn/<experiment>/<group>/_scratch/           <- removed
    job_runs/ligandmpnn/<experiment>/<group>/redesign.jsonl      <- removed
    inputs_prepared/<experiment>/<group>/<protein_id>.pdb        <- removed

LigandMPNN writes sixteen repacked backbones per protein and nothing reads them.
Stage 04 folds with AlphaFold3 against the PREPARED COMPLEX, not against these,
so they are the largest thing in the stage and the least useful.

Nothing is deleted on the strength of the results table alone. A protein is only
cleaned when BOTH are true:

    it has at least one row in tables/stage_03_results_<experiment>.csv, and
    its designs are inside sorted_clean/<experiment>/<outcome>/<group>.tar.gz

The archive is opened and the member names checked, every time. A row in a table
is a claim; a member in the tarball is the sequence itself.

The prepared complex and the redesign spec both go too, because select_sequences
copies them into the passed archive first -- so they survive in the one place
stage 04 actually reads them from. The check here is for the pdb specifically:
a group archived before the complex travelled keeps its inputs_prepared/, since
deleting it would leave stage 04 with no ligand to fold against.

Scratch belonging to proteins that FAILED is left alone: there is no table row
for them, so there is nothing to inspect them against, and a failed run is
exactly when you want the inputs still on disk.

Usage:
    ./cleanup.py --stage ~/1cbh_clear/03_ligandmpnn --dry-run
    ./cleanup.py --stage ~/1cbh_clear/03_ligandmpnn
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import job_paths as jp  # noqa: E402
from archives import load_table  # noqa: E402

STAGE = Path(__file__).resolve().parents[2]

SCRATCH_DIRS = ("_preflight", "_scratch")


def _log(*parts: object) -> None:
    print(*parts, flush=True)


@dataclass
class CleanReport:
    removed: int = 0
    kept_unscored: int = 0
    kept_unarchived: int = 0
    kept_no_complex: int = 0
    bytes_freed: int = 0
    paths: List[Path] = field(default_factory=list)

    def summary(self) -> str:
        freed = self.bytes_freed / (1024 ** 3)
        parts = [f"clean: {self.removed} protein(s) cleared ({freed:.2f} GiB)"]
        if self.kept_unscored:
            parts.append(f"{self.kept_unscored} not selected yet")
        if self.kept_unarchived:
            parts.append(f"{self.kept_unarchived} not in an archive")
        if self.kept_no_complex:
            parts.append(f"{self.kept_no_complex} whose complex never travelled")
        return ", ".join(parts)


def archived_designs(stage: Path, experiment: str) -> Tuple[Set[str], Set[str]]:
    """(proteins with designs archived, proteins whose complex pdb is archived).

    Read from the archives themselves rather than inferred from the table, so a
    table written before an archive was updated cannot authorise a deletion.
    """
    designed: Set[str] = set()
    with_complex: Set[str] = set()
    for outcome in jp.OUTCOMES:
        directory = jp.sorted_clean_dir(stage, experiment, outcome)
        if not directory.is_dir():
            continue
        for archive_path in sorted(directory.glob("*.tar.gz")):
            with tarfile.open(archive_path, "r:gz") as archive:
                for name in archive.getnames():
                    cleaned = name[2:] if name.startswith("./") else name
                    parts = cleaned.split("/")
                    if len(parts) < 2:
                        continue
                    protein_id = parts[0]
                    if parts[-1].endswith(".fa"):
                        designed.add(protein_id)
                    elif parts[-1] == f"{protein_id}.pdb":
                        with_complex.add(protein_id)
    return designed, with_complex


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def remove(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def clean_experiment(stage: Path, experiment: str, report: CleanReport,
                     dry_run: bool) -> None:
    table = jp.results_table_path(stage, experiment)
    rows = load_table(table, key="sequence_id")
    if not rows:
        _log(f"[clean] {experiment}: nothing selected yet, nothing to clear")
        return

    # One protein has many sequences; the table is keyed on the sequence.
    groups: Dict[str, str] = {}
    for row in rows.values():
        protein_id = row.get("protein_id", "")
        if protein_id:
            groups[protein_id] = row.get("group", "")

    designed, with_complex = archived_designs(stage, experiment)
    _log(f"[clean] {experiment}: {len(groups)} protein(s) selected, "
         f"{len(designed)} present in sorted_clean archives")

    for protein_id, group_key in sorted(groups.items()):
        if protein_id not in designed:
            report.kept_unarchived += 1
            continue

        targets: List[Path] = [
            jp.ligand_work_dir(stage, experiment, group_key) / "out" / protein_id
        ]
        # The complex only goes if it is provably in the passed archive; stage 04
        # has no ligand to fold against without it.
        if protein_id in with_complex:
            targets.append(
                jp.prepared_dir(stage, experiment, group_key) / f"{protein_id}.pdb"
            )
        else:
            report.kept_no_complex += 1

        present = [path for path in targets if path.exists()]
        if not present:
            continue

        size = sum(directory_size(path) for path in present)
        if dry_run:
            _log(f"[dry-run] {protein_id}: would remove {len(present)} path(s), "
                 f"{size / (1024 ** 2):.1f} MiB")
        else:
            for path in present:
                remove(path, dry_run)
        report.removed += 1
        report.bytes_freed += size
        report.paths.extend(present)

    # Group scratch: the pre-flight run and the extraction scratch belong to no
    # protein and are regenerated on the next run.
    for group_key in sorted(set(groups.values())):
        work_dir = jp.ligand_work_dir(stage, experiment, group_key)
        for name in SCRATCH_DIRS:
            path = work_dir / name
            if not path.exists():
                continue
            size = directory_size(path)
            if dry_run:
                _log(f"[dry-run] {group_key}/{name}: would remove "
                     f"{size / (1024 ** 2):.1f} MiB")
            else:
                remove(path, dry_run)
            report.bytes_freed += size
            report.paths.append(path)

        # The spec is in the passed archive, one json per protein, so the jsonl
        # here is a duplicate -- but only for the proteins that got that far.
        spec = jp.redesign_spec_path(stage, experiment, group_key)
        if spec.is_file() and all(pid in with_complex for pid, grp in groups.items()
                                  if grp == group_key):
            size = spec.stat().st_size
            if dry_run:
                _log(f"[dry-run] {group_key}/{spec.name}: would remove")
            else:
                remove(spec, dry_run)
            report.bytes_freed += size
            report.paths.append(spec)

    # Proteins designed but never selected keep everything they have.
    for group_key in sorted(set(groups.values())):
        out_root = jp.ligand_work_dir(stage, experiment, group_key) / "out"
        if not out_root.is_dir():
            continue
        for job_dir in out_root.iterdir():
            if job_dir.is_dir() and job_dir.name not in groups:
                report.kept_unscored += 1


def remove_empty_dirs(stage: Path, dry_run: bool) -> int:
    """Tidy the directories the deletions emptied. Never removes a stage root."""
    removed = 0
    for root_name in ("job_runs", "inputs_prepared"):
        root = stage / root_name
        if not root.is_dir():
            continue
        for directory in sorted(root.rglob("*"),
                                key=lambda p: len(p.parts), reverse=True):
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
