#!/usr/bin/env python3
"""
Stage 05, step 3: keep the best three redesigns per complex.

    outputs/<experiment>/<group>.tar.gz                  <- all 16 per complex
    sorted_clean/<experiment>/passed/<group>.tar.gz      <- the best 3
    sorted_clean/<experiment>/rejected/<group>.tar.gz    <- the rest
    tables/stage_05_results_<experiment>.csv             <- one row per sequence

Selection is on overall_confidence, and HIGHER IS BETTER -- the opposite of
stage 03. LigandMPNN prints np.exp(-loss), a probability, where ProteinMPNN
printed the loss itself. Sorting these the way stage 03 sorts its scores would
keep precisely the three worst designs of every sixteen, and nothing downstream
would look wrong.

ligand_confidence is recorded beside it: the same quantity restricted to the
residues near the polysaccharide, which is the part this stage was trying to
improve. It does not select, because a design that binds well and folds badly
is not wanted -- but it is the column to read when judging whether the redesign
achieved anything.
"""
from __future__ import annotations

import argparse
import re
import sys
import tarfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import job_paths as jp  # noqa: E402
from archives import load_table, merge_members_into_archive, save_table  # noqa: E402

STAGE = Path(__file__).resolve().parents[2]

KEEP_BEST = 3
CHAIN_SEPARATOR = ":"

RESULT_FIELDS = [
    "design_id", "sequence_id", "protein_id", "experiment_name", "group",
    "time_stamp", "overall_confidence", "ligand_confidence", "seq_rec",
    "length", "ala_content", "gly_content", "status",
]


def _field(header: str, name: str) -> Optional[float]:
    match = re.search(rf"\b{re.escape(name)}\s*=\s*([0-9eE.+-]+)", header)
    return float(match.group(1)) if match else None


class SelectionError(Exception):
    """A group's designs could not be scored or filed."""


def _log(*parts: object) -> None:
    print(*parts, flush=True)


@dataclass
class SelectReport:
    complexes: int = 0
    sequences: int = 0
    kept: int = 0
    already: int = 0
    tables: List[Path] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        if not self.sequences:
            return f"select: nothing new ({self.already} already selected)"
        return (f"select: {self.complexes} complex(es), {self.sequences} design(s), "
                f"{self.kept} kept, {self.already} already selected")


def composition(sequence: str) -> Tuple[int, float, float]:
    """(length, Ala fraction, Gly fraction) of ONE chain.

    The chains are tied and identical, so the first is measured; the
    concatenation would report double the length.
    """
    chain = sequence.split(CHAIN_SEPARATOR)[0]
    if not chain:
        raise SelectionError("empty sequence")
    return len(chain), chain.count("A") / len(chain), chain.count("G") / len(chain)


def read_member(archive: tarfile.TarFile, name: str) -> Tuple[str, str]:
    handle = archive.extractfile(name)
    if handle is None:
        raise SelectionError(f"could not read {name!r}")
    header, parts = "", []
    for line in handle.read().decode("utf-8").splitlines():
        if line.startswith(">"):
            header = line[1:].strip()
        elif line.strip():
            parts.append(line.strip())
    if not header or not parts:
        raise SelectionError(f"{name!r} is not a readable fasta record")
    return header, "".join(parts)


def group_by_complex(member_names: Sequence[str]) -> Dict[str, List[str]]:
    """{sequence_id: [member, ...]} from <sequence_id>/<design_id>.fa."""
    grouped: Dict[str, List[str]] = {}
    for name in member_names:
        if name.endswith(".fa") and "/" in name:
            grouped.setdefault(name.split("/")[0], []).append(name)
    return grouped


def protein_of(sequence_id: str, specs: Dict[str, str]) -> str:
    return specs.get(sequence_id, "")


def select_group(stage: Path, experiment: str, group_key: str,
                 by_id: Dict[str, Dict[str, str]], report: SelectReport,
                 proteins: Dict[str, str], keep: int) -> None:
    source = jp.ligand_outputs_path(stage, experiment, group_key)
    if not source.is_file():
        raise SelectionError(f"no LigandMPNN output for this group: {source}")

    passed: List[Tuple[tarfile.TarInfo, bytes]] = []
    rejected: List[Tuple[tarfile.TarInfo, bytes]] = []

    with tarfile.open(source, "r:gz") as archive:
        for sequence_id, names in sorted(group_by_complex(archive.getnames()).items()):
            scored: List[Tuple[float, str, str]] = []
            for name in sorted(names):
                design_id = Path(name).stem
                if design_id in by_id:
                    report.already += 1
                    continue
                header, sequence = read_member(archive, name)
                confidence = _field(header, "overall_confidence")
                if confidence is None:
                    raise SelectionError(
                        f"no overall_confidence in {name!r}: {header[:90]!r}"
                    )
                length, ala, gly = composition(sequence)
                scored.append((confidence, design_id, name))
                by_id[design_id] = {
                    "design_id": design_id,
                    "sequence_id": sequence_id,
                    "protein_id": protein_of(sequence_id, proteins),
                    "experiment_name": experiment,
                    "group": group_key,
                    "time_stamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "overall_confidence": f"{confidence:.6g}",
                    "ligand_confidence": _fmt(_field(header, "ligand_confidence")),
                    "seq_rec": _fmt(_field(header, "seq_rec")),
                    "length": str(length),
                    "ala_content": f"{ala:.4f}",
                    "gly_content": f"{gly:.4f}",
                    "status": "",
                }
            if not scored:
                continue

            report.complexes += 1
            report.sequences += len(scored)
            # HIGHEST confidence first. design_id breaks ties so the choice does
            # not depend on the order the archive happened to list members in.
            scored.sort(key=lambda item: (-item[0], item[1]))
            for rank, (_, design_id, name) in enumerate(scored):
                keeping = rank < keep
                by_id[design_id]["status"] = "KEPT" if keeping else "NOT_KEPT"
                handle = archive.extractfile(archive.getmember(name))
                if handle is None:
                    raise SelectionError(f"could not re-read {name!r}")
                payload = handle.read()
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                (passed if keeping else rejected).append((info, payload))
                if keeping:
                    report.kept += 1

    # The complex the designs were made on travels with them: stage 06 folds
    # against its ligand, and whatever compares the fold to the design later
    # needs the pre-fold coordinates. Reaching back into stage 05 would break
    # the moment stage 05 is cleaned the way stage 04 now is.
    for sequence_id in sorted({info.name.split("/")[0] for info, _ in passed}):
        complex_file = jp.complex_path(stage, experiment, group_key, sequence_id)
        if not complex_file.is_file():
            _log(f"[select] {sequence_id}: no complex pdb at {complex_file}, "
                 f"designs travel without their reference")
            continue
        payload = complex_file.read_bytes()
        info = tarfile.TarInfo(f"{sequence_id}/{sequence_id}.pdb")
        info.size = len(payload)
        passed.append((info, payload))

    for outcome, members in (("passed", passed), ("rejected", rejected)):
        if not members:
            continue
        destination = jp.sorted_archive_path(stage, experiment, outcome, group_key)
        existing: set = set()
        if destination.is_file():
            with tarfile.open(destination, "r:gz") as archive:
                existing = set(archive.getnames())
        fresh = [(info, payload) for info, payload in members
                 if info.name not in existing]
        if not fresh:
            continue
        added, total = merge_members_into_archive(destination, fresh)
        _log(f"[select] {outcome}/{group_key}: +{added} design(s), {total} total "
             f"-> {destination}")


def _fmt(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.6g}"


def protein_map(stage: Path, experiment: str, group_key: str) -> Dict[str, str]:
    """{sequence_id: protein_id}, from the spec ligand_prepare wrote."""
    import json
    path = jp.redesign_spec_path(stage, experiment, group_key)
    mapping: Dict[str, str] = {}
    if not path.is_file():
        return mapping
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                mapping[record["sequence_id"]] = record.get("protein_id", "")
    return mapping


def run_selection(stage: Path, experiment: Optional[str] = None,
                  keep: int = KEEP_BEST) -> SelectReport:
    report = SelectReport()
    outputs_root = stage / "outputs"
    if not outputs_root.is_dir():
        _log(f"[select] nothing designed yet under {outputs_root}")
        return report

    names = [experiment] if experiment else sorted(
        path.name for path in outputs_root.iterdir() if path.is_dir()
    )
    for experiment_name in names:
        archives = sorted((outputs_root / experiment_name).glob("*.tar.gz"))
        if not archives:
            continue
        table_path = jp.results_table_path(stage, experiment_name)
        by_id = load_table(table_path, key="design_id")
        before = len(by_id)

        for archive_path in archives:
            group_key = archive_path.name[: -len(".tar.gz")]
            try:
                select_group(stage, experiment_name, group_key, by_id, report,
                             protein_map(stage, experiment_name, group_key), keep)
            except SelectionError as exc:
                report.problems.append(f"{experiment_name}/{group_key}: {exc}")
                _log(f"[select] {experiment_name}/{group_key}: FAILED ({exc})")

        if len(by_id) != before:
            save_table(by_id, table_path, RESULT_FIELDS,
                       lambda row: (row.get("sequence_id", ""),
                                    -float(row.get("overall_confidence") or 0)))
            report.tables.append(table_path)
    return report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--keep", type=int, default=KEEP_BEST)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = run_selection(args.stage.resolve(), args.experiment, args.keep)
    except (OSError, SelectionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    _log(f"[done] {report.summary()}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

