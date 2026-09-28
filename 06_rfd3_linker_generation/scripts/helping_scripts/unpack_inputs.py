#!/usr/bin/env python3
"""
Stage 06, step 1: lay out the structures stage 05 handed over.

    inputs/<experiment>/<group>.tar.gz              <- stage 05, flat, pdbs only
    inputs_prepared/<experiment>/<group>/<sid>.pdb  <- RFD3 diffuses on this

One file per sequence: two aligned protein copies with the cellulose already
dropped in stage 05. There is no fibril to lay out and no monomer.

The jsons are not in the archive and never pass through here:
stage 05 writes them straight into this stage's json/ directory, so that the
file you fill a length into is the only copy of it anywhere. Extracting a second
one over the top is exactly what this stage must not do.

Unpacking to this relative path is what makes each json's "input" field resolve
without rewriting -- stage 05 and stage 06 both spell it
inputs_prepared/<experiment>/<group>/<sid>.pdb.

An existing file is not overwritten. Nothing edits these pdbs by hand, so that
is a cheap guard rather than a necessary one; it also means re-running costs a
directory listing when there is nothing new.

Usage:
    ./unpack_inputs.py --stage ~/1cbh_clear/06_rfd3_linker_generation
    ./unpack_inputs.py --experiment NAME --force     # replace what is there
"""
from __future__ import annotations

import argparse
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import job_paths as jp  # noqa: E402

STAGE = Path(__file__).resolve().parents[2]

STRUCTURE_SUFFIXES = (".pdb",)


class UnpackError(Exception):
    """A group could not be laid out."""


def _log(*parts: object) -> None:
    print(*parts, flush=True)


@dataclass
class UnpackReport:
    structures: int = 0
    kept: int = 0            # already on disk, left alone
    problems: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        if not self.structures:
            return f"unpack: nothing new ({self.kept} structure(s) already in place)"
        kept = f", {self.kept} left as they were" if self.kept else ""
        return f"unpack: {self.structures} structure(s) laid out{kept}"


def destination_for(stage: Path, experiment: str, group_key: str,
                    filename: str) -> Optional[Path]:
    """Where one archive member lands, or None if it is not ours to lay out.

    A json returns None deliberately. Older archives, written before the jsons
    moved into stage 06, still carry them; extracting one would drop a pristine
    LINKER copy on top of a length already set. They are ignored by name rather
    than assumed absent.
    """
    if filename.endswith(STRUCTURE_SUFFIXES):
        return jp.pair_dir(stage, experiment, group_key) / filename
    return None


def unpack_group(stage: Path, experiment: str, group_key: str,
                 report: UnpackReport, force: bool = False) -> None:
    archive_path = jp.input_archive_path(stage, experiment, group_key)
    if not archive_path.is_file():
        raise UnpackError(f"no input archive: {archive_path}")

    with tarfile.open(archive_path, "r:gz") as archive:
        for name in sorted(archive.getnames()):
            cleaned = name[2:] if name.startswith("./") else name
            # Flat now, but an archive written before the flattening nests each
            # file under <sid>/. Taking the basename reads both.
            destination = destination_for(stage, experiment, group_key,
                                          Path(cleaned).name)
            if destination is None:
                continue
            if destination.is_file() and not force:
                report.kept += 1
                continue
            handle = archive.extractfile(name)
            if handle is None:
                report.problems.append(f"{experiment}/{cleaned}: unreadable")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(handle.read())
            report.structures += 1


def run_unpack(stage: Path, experiment: Optional[str] = None,
               force: bool = False) -> UnpackReport:
    report = UnpackReport()
    names = [experiment] if experiment else jp.inputs_experiments(stage)
    if not names:
        _log(f"[unpack] nothing handed over under {jp.inputs_root(stage)}")
        return report
    for experiment_name in names:
        directory = jp.inputs_root(stage) / experiment_name
        if not directory.is_dir():
            continue
        for archive_path in sorted(directory.glob("*.tar.gz")):
            group_key = archive_path.name[: -len(".tar.gz")]
            try:
                unpack_group(stage, experiment_name, group_key, report, force)
            except (UnpackError, OSError, tarfile.TarError) as exc:
                report.problems.append(f"{experiment_name}/{group_key}: {exc}")
                _log(f"[unpack] {experiment_name}/{group_key}: FAILED ({exc})")
    return report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--force", action="store_true",
                        help="replace structures that are already there")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    report = run_unpack(args.stage.resolve(), args.experiment, args.force)
    _log(f"[done] {report.summary()}")
    for problem in report.problems[:10]:
        _log(f"[done]   {problem}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
