#!/usr/bin/env python3
"""
Stage 06, step 3: fold the generated backbones into per-group archives.

    outputs_raw/<experiment>/<sid>_AB/<sid>_AB_000001.cif    <- RFD3 wrote
    outputs_raw/<experiment>/<sid>_AB/<sid>_AB_000001.json   <- diffused_index_map
    sorted_clean/<experiment>/passed/<group>.tar.gz          <- one per SOURCE group

The regrouping is the whole job. RFD3 files its output under the json it ran,
so a run leaves one directory per json -- and with two orientations per pair
that is fifty-six directories for twenty-eight pairs. Every other stage in this
pipeline carries one tarball per group, and stage 07 would otherwise receive
fifty-six inputs where stage 05 received one.

So each design is filed back under the group its pair came from, which is read
from stage 06's own input archives: <group>.tar.gz holds <sid>.pdb, so the
sequence id says which group it belongs to, and <sid>_AB_000001 says which
sequence id. Nothing has to be passed along or remembered.

A design whose sequence id is in no input archive is reported and left where it
is, not filed under a guess. That happens when outputs_raw holds a run from an
experiment whose inputs have since been cleaned, and quietly inventing a group
for it would put it in an archive stage 07 then reads as something else.

Both files travel together. The cif is the backbone; the json carries
diffused_index_map, which is the only record of which residues are the linker
and which are the two copies -- stage 07 redesigns the first and holds the
second, so a cif without its json is not usable there.
"""
from __future__ import annotations

import argparse
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import job_paths as jp  # noqa: E402
from archives import merge_files_into_archive  # noqa: E402

STAGE = Path(__file__).resolve().parents[2]

DESIGN_SUFFIXES = (".cif", ".json")


class ArchiveError(Exception):
    """A group could not be archived."""


def _log(*parts: object) -> None:
    print(*parts, flush=True)


@dataclass
class ArchiveReport:
    archived: int = 0
    groups: int = 0
    unplaced: List[str] = field(default_factory=list)
    incomplete: List[str] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        if not self.archived:
            return "archive: nothing new"
        parts = [f"archive: {self.archived} file(s) into {self.groups} group(s)"]
        if self.incomplete:
            parts.append(f"{len(self.incomplete)} design(s) missing a half")
        if self.unplaced:
            parts.append(f"{len(self.unplaced)} could not be placed")
        return ", ".join(parts)


def group_of_sequence(stage: Path, experiment: str) -> Dict[str, str]:
    """{sequence_id: source group}, read from this stage's input archives.

    The input archive for a group holds one <sid>.pdb per pair, so its member
    names are the mapping. Read once per experiment rather than per design.
    """
    mapping: Dict[str, str] = {}
    directory = jp.inputs_root(stage) / experiment
    if not directory.is_dir():
        return mapping
    for archive_path in sorted(directory.glob("*.tar.gz")):
        group_key = archive_path.name[: -len(".tar.gz")]
        try:
            with tarfile.open(archive_path, "r:gz") as archive:
                for name in archive.getnames():
                    if name.endswith(".pdb"):
                        mapping[Path(name).stem] = group_key
        except (OSError, tarfile.TarError) as exc:
            _log(f"[archive] cannot read {archive_path.name}: {exc}")
    return mapping


def sequence_of_design(design_name: str, known: Dict[str, str]) -> Optional[str]:
    """The sequence id a design belongs to, by longest matching prefix.

    <sid>_AB_000001 -> <sid>. Matched against the ids actually present rather
    than parsed, because a sequence id contains underscores itself and there is
    no rule that survives both '1cbh_1xcellulose_dp20_00078714' and the '_AB'
    and '_000001' stuck on the end of it.
    """
    for sequence_id in sorted(known, key=len, reverse=True):
        if design_name.startswith(f"{sequence_id}_"):
            return sequence_id
    return None


def archive_experiment(stage: Path, experiment: str, report: ArchiveReport) -> None:
    raw_root = jp.raw_root(stage) / experiment
    if not raw_root.is_dir():
        return

    known = group_of_sequence(stage, experiment)
    if not known:
        _log(f"[archive] {experiment}: no input archives, so nothing can be filed")
        return

    # {group: {design name: [paths]}} -- collected whole before anything is
    # written, so a design missing its json is reported rather than half-filed.
    by_group: Dict[str, Dict[str, List[Path]]] = {}
    for path in sorted(raw_root.rglob("*")):
        if not path.is_file() or path.suffix not in DESIGN_SUFFIXES:
            continue
        design_name = path.stem
        sequence_id = sequence_of_design(design_name, known)
        if sequence_id is None:
            report.unplaced.append(f"{experiment}/{path.name}")
            continue
        group_key = known[sequence_id]
        by_group.setdefault(group_key, {}).setdefault(design_name, []).append(path)

    for group_key, designs in sorted(by_group.items()):
        files: List[Path] = []
        for design_name, paths in sorted(designs.items()):
            suffixes = {path.suffix for path in paths}
            if suffixes != set(DESIGN_SUFFIXES):
                missing = sorted(set(DESIGN_SUFFIXES) - suffixes)
                report.incomplete.append(
                    f"{experiment}/{design_name}: no {', '.join(missing)}")
                _log(f"[archive] {design_name}: skipped, missing {', '.join(missing)}")
                continue
            files.extend(paths)
        if not files:
            continue
        # sorted_clean/<experiment>/passed/, not outputs_clean/, because that
        # is where the shared transfer looks. Nothing was sorted here -- this
        # stage filters nothing -- but "passed" is the machinery's name for
        # "moves on to the next stage", and stage 05 writes there for the same
        # reason. Inventing a second convention for one stage would cost more
        # than the slightly loud word does.
        archive_path = jp.sorted_archive_path(stage, experiment, "passed", group_key)
        added, total = merge_files_into_archive(archive_path, files)
        if added:
            report.archived += added
            report.groups += 1
            _log(f"[archive] {group_key}: +{added} file(s), {total} total "
                 f"-> {archive_path}")


def run_archive(stage: Path, experiment: Optional[str] = None) -> ArchiveReport:
    report = ArchiveReport()
    raw_root = jp.raw_root(stage)
    if not raw_root.is_dir():
        _log(f"[archive] nothing generated under {raw_root}")
        return report
    names = [experiment] if experiment else sorted(
        path.name for path in raw_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    for experiment_name in names:
        try:
            archive_experiment(stage, experiment_name, report)
        except (ArchiveError, OSError, tarfile.TarError) as exc:
            report.problems.append(f"{experiment_name}: {exc}")
            _log(f"[archive] {experiment_name}: FAILED ({exc})")
    return report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--experiment", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    report = run_archive(args.stage.resolve(), args.experiment)
    _log(f"[done] {report.summary()}")
    for entry in (report.incomplete + report.unplaced)[:10]:
        _log(f"[done]   {entry}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
