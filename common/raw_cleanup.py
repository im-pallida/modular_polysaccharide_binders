#!/usr/bin/env python3
"""
Delete raw files that an archive already holds. Shared by the stages that
generate them.

THE RULE, and it is the only one

    a file is deleted when, and only when, a member of that exact name is
    found inside the archive it belongs to.

Not when a results table says it was processed -- a row is a claim. Not when a
tarball exists beside it -- that is a coincidence. The archive is opened and
its names are read, every run. Stages 03 to 08 each implemented this rule
separately; this is that rule, once, for the stages that were missing it.

WHAT MUST SURVIVE

Deleting is easy and irreversible, so each stage's cleanup names its raw trees
explicitly and touches nothing else. In particular none of them go near:

    json/               configuration, read by stages 02 and 08
    job_runs/           stage 01's fixed_residues jsonl, read by 02, 03 and 04
                        stage 07's linker.jsonl, read by 08
    outputs_clean/      the archives themselves
    sorted_clean/       the archives themselves
    tables/             the results

RESUME COMES FIRST

A cleanup is only safe once the stage's "have I already done this?" check reads
the archive rather than the raw file. Stage 06 learned that the hard way --
cleanup cleared outputs_raw, the runner saw no raw cif, and RFD3 regenerated
every design. Stage 01's check was pointed at the archive before this file was
allowed anywhere near it.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set

from archives import archived_stems  # noqa: E402


@dataclass
class CleanReport:
    removed: int = 0
    kept_unarchived: int = 0
    bytes_freed: int = 0
    dirs_removed: int = 0
    paths: List[Path] = field(default_factory=list)
    unarchived: List[Path] = field(default_factory=list)

    def summary(self) -> str:
        freed = self.bytes_freed / (1024 ** 3)
        parts = [f"clean: {self.removed} file(s) cleared ({freed:.2f} GiB)"]
        if self.kept_unarchived:
            parts.append(f"{self.kept_unarchived} NOT in an archive, kept")
        if self.dirs_removed:
            parts.append(f"{self.dirs_removed} empty director(y/ies) removed")
        return ", ".join(parts)


def clean_tree(raw_root: Path,
               archive_for: Callable[[Path], Optional[Path]],
               report: CleanReport,
               dry_run: bool = False,
               keep: Sequence[str] = (),
               log: Optional[Callable[..., None]] = None) -> None:
    """Walk raw_root and delete what the archives already hold.

    archive_for(path) names the archive a given raw file belongs in, or None if
    it cannot be worked out -- in which case the file is kept and reported,
    because an unplaceable file is exactly the one not to guess about.

    keep names basenames that are never deleted whatever the archives say.

    Archives are opened once each, not once per file: a 16k run would otherwise
    read the same tarball sixteen thousand times.
    """
    if not raw_root.is_dir():
        return
    held: Dict[Path, Set[str]] = {}
    protected = set(keep)

    for path in sorted(raw_root.rglob("*")):
        if not path.is_file() or path.name in protected:
            continue
        archive = archive_for(path)
        if archive is None:
            report.kept_unarchived += 1
            report.unarchived.append(path)
            continue
        if archive not in held:
            held[archive] = archived_stems(archive)
        if path.name not in held[archive]:
            report.kept_unarchived += 1
            report.unarchived.append(path)
            continue
        size = path.stat().st_size
        if dry_run:
            if log:
                log(f"[dry-run] would remove {path}")
        else:
            path.unlink()
        report.removed += 1
        report.bytes_freed += size
        report.paths.append(path)


def remove_empty_dirs(roots: Iterable[Path], report: CleanReport,
                      dry_run: bool = False) -> None:
    """Tidy the directories the deletions emptied. Never removes a root."""
    for root in roots:
        if not root.is_dir():
            continue
        for directory in sorted(root.rglob("*"),
                                key=lambda p: len(p.parts), reverse=True):
            if directory.is_dir() and not any(directory.iterdir()):
                if not dry_run:
                    directory.rmdir()
                report.dirs_removed += 1


def drop_tree(path: Path, report: CleanReport, dry_run: bool = False) -> None:
    """Remove a whole scratch directory that holds nothing worth verifying.

    Only for trees the stage itself creates and re-creates -- a temp folder, a
    pre-flight run -- never for anything a later stage reads.
    """
    if not path.is_dir():
        return
    size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    if not dry_run:
        shutil.rmtree(path)
    report.bytes_freed += size
    report.paths.append(path)
