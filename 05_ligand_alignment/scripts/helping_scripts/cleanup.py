#!/usr/bin/env python3
"""
Stage 05, last step: delete what the archive already holds.

    inputs_prepared/<experiment>/<group>/_scratch/    <- removed
    inputs_prepared/<experiment>/<group>/<sid>.pdb    <- removed once archived

Nothing is deleted on the strength of a filename. The archive is opened and the
member names checked, every time; a pdb that is not in there is left alone and
counted, however tidy removing it would look.

The loose pdbs go by default. They are working state -- the pair lives in
sorted_clean/<experiment>/passed/<group>.tar.gz and travels to stage 06 in it,
so a copy lying beside the archive is a second copy of a file nobody edits.
--keep-pairs leaves them if you would rather open them without extracting.
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


def _log(*parts: object) -> None:
    print(*parts, flush=True)


@dataclass
class CleanReport:
    removed: int = 0
    kept_pairs: int = 0
    kept_unarchived: int = 0
    bytes_freed: int = 0
    paths: List[Path] = field(default_factory=list)

    def summary(self) -> str:
        freed = self.bytes_freed / (1024 ** 2)
        parts = [f"clean: {self.removed} file(s) cleared ({freed:.1f} MiB)"]
        if self.kept_pairs:
            parts.append(f"{self.kept_pairs} pair(s) kept on disk")
        if self.kept_unarchived:
            parts.append(f"{self.kept_unarchived} not in an archive")
        return ", ".join(parts)


def archived_members(stage: Path, experiment: str, outcome: str) -> Set[str]:
    """Every member name inside one outcome's archives, as written."""
    found: Set[str] = set()
    directory = jp.sorted_clean_dir(stage, experiment, outcome)
    if not directory.is_dir():
        return found
    for archive_path in sorted(directory.glob("*.tar.gz")):
        with tarfile.open(archive_path, "r:gz") as archive:
            for name in archive.getnames():
                found.add(name[2:] if name.startswith("./") else name)
    return found


def archived(sequence_id: str, members: Set[str]) -> bool:
    """True if this sequence's pdb is inside the passed archive.

    One file per sequence, stored flat, so this is a membership test and not the
    reconstruction argument it used to be: when the pair, the stripped copy and
    the fibril were three files, deleting one meant checking the other two could
    rebuild it. There is nothing left to rebuild from, and nothing to rebuild.
    """
    return f"{sequence_id}.pdb" in members


def clean_experiment(stage: Path, experiment: str, report: CleanReport,
                     dry_run: bool, clean_pairs: bool) -> None:
    passed = archived_members(stage, experiment, "passed")
    _log(f"[clean] {experiment}: {len(passed)} member(s) present in the passed "
         f"archives")

    root = stage / "inputs_prepared"
    if root.is_dir():
        for pair_file in sorted(root.rglob("*.pdb")):
            sequence_id = pair_file.stem
            if not archived(sequence_id, passed):
                report.kept_unarchived += 1
                continue
            if not clean_pairs:
                report.kept_pairs += 1
                continue
            size = pair_file.stat().st_size
            if dry_run:
                _log(f"[dry-run] {pair_file.relative_to(stage)}: would remove "
                     f"{size / 1024:.0f} KiB")
            else:
                pair_file.unlink()
            report.removed += 1
            report.bytes_freed += size
            report.paths.append(pair_file)

    # Scratch belongs to no pair and is rebuilt on the next run.
    for scratch in sorted((stage / "inputs_prepared").rglob("_scratch")):
        if not scratch.is_dir():
            continue
        size = sum(item.stat().st_size for item in scratch.rglob("*") if item.is_file())
        if dry_run:
            _log(f"[dry-run] {scratch.relative_to(stage)}: would remove")
        else:
            shutil.rmtree(scratch)
        report.bytes_freed += size
        report.paths.append(scratch)


def remove_empty_dirs(stage: Path, dry_run: bool) -> int:
    removed = 0
    for root_name in ("inputs_prepared",):
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
                dry_run: bool = False, clean_pairs: bool = True) -> CleanReport:
    report = CleanReport()
    names = [experiment] if experiment else jp.inputs_experiments(stage)
    if not names:
        _log(f"[clean] nothing under {jp.inputs_root(stage)}")
        return report
    for experiment_name in names:
        clean_experiment(stage, experiment_name, report, dry_run, clean_pairs)
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
    parser.add_argument("--keep-pairs", action="store_true",
                        help="leave the loose pdbs on disk instead of removing "
                             "the ones already in the archive")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    report = run_cleanup(args.stage.resolve(), args.experiment,
                         args.dry_run, not args.keep_pairs)
    _log(f"[done] {report.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
