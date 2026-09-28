#!/usr/bin/env python3
"""
Stage 06, last step: delete what the archives already hold.

    outputs_raw/<experiment>/<key>/<name>.cif    <- removed once archived
    outputs_raw/<experiment>/<key>/<name>.json   <- removed once archived
    outputs_raw/.tmp/                            <- removed
    inputs_prepared/<experiment>/<group>/<sid>.pdb   <- KEPT, see below

Nothing is deleted on the strength of a filename. A design goes only when BOTH
its halves are inside sorted_clean/<experiment>/passed/<group>.tar.gz, and the
archive is opened and the member names checked, every time. Half a design in the
archive means neither half is removed: the cif alone is not usable in stage 07,
so a cleanup that left it archived and deleted the json would quietly destroy the
only record of which residues are the linker.

The jsons in json/ are never touched. They hold the lengths you typed, they are
small, and re-running this stage reads them again to decide what is outstanding.
Deleting them would make an already-generated pair look unfilled.

The laid-out pdbs stay. Every json names one by absolute path, and jsons are run
over and over as you fill lengths in over weeks -- so deleting the file a json
points at, on the grounds that the next unpack would put it back, is churn that
turns a direct run of run_rfd3.py into a missing-input error. They are the RFD3
inputs, and nothing deletes stage 01's seeds either. Twenty-eight of them are a
few megabytes; outputs_raw is where the disk actually goes.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import job_paths as jp  # noqa: E402

STAGE = Path(__file__).resolve().parents[2]

DESIGN_SUFFIXES = (".cif", ".json")


def _log(*parts: object) -> None:
    print(*parts, flush=True)


@dataclass
class CleanReport:
    removed: int = 0
    kept_unarchived: int = 0
    kept_partial: int = 0
    bytes_freed: int = 0
    paths: List[Path] = field(default_factory=list)

    def summary(self) -> str:
        freed = self.bytes_freed / (1024 ** 2)
        parts = [f"clean: {self.removed} file(s) cleared ({freed:.1f} MiB)"]
        if self.kept_unarchived:
            parts.append(f"{self.kept_unarchived} not in an archive")
        if self.kept_partial:
            parts.append(f"{self.kept_partial} archived without their pair")
        return ", ".join(parts)


def archived_members(stage: Path, experiment: str) -> Set[str]:
    """Every member name inside this experiment's passed archives."""
    found: Set[str] = set()
    directory = jp.sorted_clean_dir(stage, experiment, "passed")
    if not directory.is_dir():
        return found
    for archive_path in sorted(directory.glob("*.tar.gz")):
        with tarfile.open(archive_path, "r:gz") as archive:
            for name in archive.getnames():
                found.add(Path(name[2:] if name.startswith("./") else name).name)
    return found


def _remove(path: Path, stage: Path, report: CleanReport, dry_run: bool) -> None:
    size = path.stat().st_size
    if dry_run:
        _log(f"[dry-run] {path.relative_to(stage)}: would remove {size / 1024:.0f} KiB")
    else:
        path.unlink()
    report.removed += 1
    report.bytes_freed += size
    report.paths.append(path)


def clean_experiment(stage: Path, experiment: str, report: CleanReport,
                     dry_run: bool) -> None:
    archived = archived_members(stage, experiment)
    _log(f"[clean] {experiment}: {len(archived)} member(s) present in the passed "
         f"archives")

    raw_root = jp.raw_root(stage) / experiment
    if raw_root.is_dir():
        # Grouped by design so both halves are judged together. A design with
        # only one half archived keeps both: the pair is what stage 07 needs.
        designs: dict = {}
        for path in sorted(raw_root.rglob("*")):
            if path.is_file() and path.suffix in DESIGN_SUFFIXES:
                designs.setdefault(path.stem, []).append(path)
        for design_name, paths in sorted(designs.items()):
            wanted = {f"{design_name}{suffix}" for suffix in DESIGN_SUFFIXES}
            present = wanted & archived
            if not present:
                report.kept_unarchived += len(paths)
                continue
            if present != wanted:
                report.kept_partial += len(paths)
                _log(f"[clean] {design_name}: only {sorted(present)} archived, "
                     f"keeping both halves")
                continue
            for path in paths:
                _remove(path, stage, report, dry_run)



def remove_scratch(stage: Path, dry_run: bool, report: CleanReport) -> None:
    for name in (jp.raw_root(stage) / ".tmp",):
        if not name.is_dir():
            continue
        size = sum(item.stat().st_size for item in name.rglob("*") if item.is_file())
        if dry_run:
            _log(f"[dry-run] {name.relative_to(stage)}: would remove")
        else:
            shutil.rmtree(name)
        report.bytes_freed += size
        report.paths.append(name)


def remove_empty_dirs(stage: Path, dry_run: bool) -> int:
    removed = 0
    for root_name in ("outputs_raw", "inputs_prepared"):
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
    raw_root = jp.raw_root(stage)
    names = [experiment] if experiment else sorted(
        path.name for path in raw_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ) if raw_root.is_dir() else []
    if not names:
        _log(f"[clean] nothing generated under {raw_root}")
    for experiment_name in names:
        clean_experiment(stage, experiment_name, report, dry_run)
    remove_scratch(stage, dry_run, report)
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
