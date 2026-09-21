#!/usr/bin/env python3
"""
Stage 03, step 3: score every designed sequence and keep the best three.

    outputs/<experiment>/<group>.tar.gz                    <- run_mpnn wrote, all 16
    sorted_clean/<experiment>/passed/<group>.tar.gz        <- the best 3 per protein
    sorted_clean/<experiment>/rejected/<group>.tar.gz      <- the other 13
    tables/stage_03_results_<experiment>.csv               <- one row per sequence

Lower ProteinMPNN score is better -- it is the model's negative log likelihood
for the sequence given the backbone -- so the three lowest per protein go
forward. Ties break on sequence_id, so the choice is reproducible rather than
dependent on archive ordering.

Ala and Gly content are recorded but never select: they are there because the
composition bias pushes against both, and the table is how you see whether it
worked.

passed/ and rejected/ hold the same shape as the source, <protein_id>/<id>.fa,
because that is what stage 04 reads. They are written directly rather than
through regroup_and_archive, which flattens member names.
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

RESULT_FIELDS = [
    "sequence_id",
    "protein_id",
    "experiment_name",
    "group",
    "time_stamp",
    "score",
    "length",
    "ala_content",
    "gly_content",
    "other_content",
    "status",
]

SCORE_PATTERN = re.compile(r"score\s*=\s*([0-9eE.+-]+)")


class SelectionError(Exception):
    """A group's sequences could not be scored or filed."""


def _log(*parts: object) -> None:
    print(*parts, flush=True)


@dataclass
class SelectReport:
    proteins: int = 0
    sequences: int = 0
    skipped_already: int = 0
    kept: int = 0
    tables: List[Path] = field(default_factory=list)
    archives: List[Path] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        if not self.sequences:
            return f"select: nothing new ({self.skipped_already} already selected)"
        return (f"select: {self.proteins} protein(s), {self.sequences} sequence(s), "
                f"{self.kept} kept, {self.skipped_already} already selected")


def parse_score(header: str) -> float:
    match = SCORE_PATTERN.search(header)
    if not match:
        raise SelectionError(f"no score in fasta header: {header!r}")
    return float(match.group(1))


def composition(sequence: str) -> Tuple[int, float, float, float]:
    """(length, Ala fraction, Gly fraction, everything else) of ONE chain.

    The chains are tied and therefore identical, so the first is measured
    rather than the concatenation -- otherwise the length would read double.
    """
    chain = sequence.split("/")[0]
    if not chain:
        raise SelectionError("empty sequence")
    total = len(chain)
    ala = chain.count("A") / total
    gly = chain.count("G") / total
    return total, ala, gly, 1.0 - ala - gly


def read_fasta_member(archive: tarfile.TarFile, name: str) -> Tuple[str, str]:
    """(header, sequence) of a one-record fasta member."""
    extracted = archive.extractfile(name)
    if extracted is None:
        raise SelectionError(f"could not read {name!r}")
    text = extracted.read().decode("utf-8")
    header, sequence = "", []
    for line in text.splitlines():
        if line.startswith(">"):
            header = line[1:].strip()
        elif line.strip():
            sequence.append(line.strip())
    if not header or not sequence:
        raise SelectionError(f"{name!r} is not a readable fasta record")
    return header, "".join(sequence)


def group_sequences(member_names: Sequence[str]) -> Dict[str, List[str]]:
    """{protein_id: [member name, ...]} from <protein_id>/<sequence_id>.fa."""
    grouped: Dict[str, List[str]] = {}
    for name in member_names:
        if not name.endswith(".fa") or "/" not in name:
            continue
        protein_id = name.split("/")[0]
        grouped.setdefault(protein_id, []).append(name)
    return grouped


def select_group(stage: Path, experiment: str, group_key: str,
                 by_id: Dict[str, Dict[str, str]], report: SelectReport,
                 keep: int = KEEP_BEST) -> None:
    """Score one group's sequences and split them into passed and rejected."""
    source = jp.mpnn_outputs_path(stage, experiment, group_key)
    if not source.is_file():
        raise SelectionError(f"no MPNN output for this group: {source}")

    passed_members: List[Tuple[tarfile.TarInfo, bytes]] = []
    rejected_members: List[Tuple[tarfile.TarInfo, bytes]] = []

    with tarfile.open(source, "r:gz") as archive:
        for protein_id, names in sorted(group_sequences(archive.getnames()).items()):
            scored: List[Tuple[float, str, str]] = []
            for name in sorted(names):
                sequence_id = Path(name).stem
                if sequence_id in by_id:
                    report.skipped_already += 1
                    continue
                header, sequence = read_fasta_member(archive, name)
                score = parse_score(header)
                length, ala, gly, other = composition(sequence)
                scored.append((score, sequence_id, name))
                by_id[sequence_id] = {
                    "sequence_id": sequence_id,
                    "protein_id": protein_id,
                    "experiment_name": experiment,
                    "group": group_key,
                    "time_stamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "score": f"{score:.6g}",
                    "length": str(length),
                    "ala_content": f"{ala:.4f}",
                    "gly_content": f"{gly:.4f}",
                    "other_content": f"{other:.4f}",
                    "status": "",
                }
            if not scored:
                continue

            report.proteins += 1
            report.sequences += len(scored)
            # Lower score is better; sequence_id breaks ties so the choice does
            # not depend on the order the archive happened to list members in.
            scored.sort(key=lambda item: (item[0], item[1]))
            for rank, (_, sequence_id, name) in enumerate(scored):
                keeping = rank < keep
                by_id[sequence_id]["status"] = "KEPT" if keeping else "NOT_KEPT"
                member = archive.getmember(name)
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise SelectionError(f"could not re-read {name!r}")
                payload = extracted.read()
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                (passed_members if keeping else rejected_members).append((info, payload))
                if keeping:
                    report.kept += 1

    for outcome, members in (("passed", passed_members), ("rejected", rejected_members)):
        if not members:
            continue
        destination = jp.sorted_archive_path(stage, experiment, outcome, group_key)
        existing = set()
        if destination.is_file():
            with tarfile.open(destination, "r:gz") as archive:
                existing = set(archive.getnames())
        fresh = [(info, payload) for info, payload in members if info.name not in existing]
        if not fresh:
            continue
        added, total = merge_members_into_archive(destination, fresh)
        report.archives.append(destination)
        _log(f"[select] {outcome}/{group_key}: +{added} sequence(s), "
             f"{total} total -> {destination}")


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
        by_id = load_table(table_path, key="sequence_id")
        before = len(by_id)
        _log(f"[select] {experiment_name}: {len(archives)} group(s), "
             f"{before} sequence(s) already selected")

        for archive_path in archives:
            group_key = archive_path.name[: -len(".tar.gz")]
            try:
                select_group(stage, experiment_name, group_key, by_id, report, keep)
            except SelectionError as exc:
                report.problems.append(f"{experiment_name}/{group_key}: {exc}")
                _log(f"[select] {experiment_name}/{group_key}: FAILED ({exc})")

        if len(by_id) != before:
            save_table(by_id, table_path, RESULT_FIELDS,
                       lambda row: (row.get("protein_id", ""), float(row.get("score") or 0)))
            report.tables.append(table_path)
    return report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--keep", type=int, default=KEEP_BEST,
                        help=f"sequences to keep per protein (default: {KEEP_BEST})")
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

