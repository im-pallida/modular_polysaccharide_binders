#!/usr/bin/env python3
"""
Stage 03, step 3: score every designed sequence and keep the best three.

    outputs/<experiment>/<group>.tar.gz                    <- run_ligandmpnn wrote, all 16
    sorted_clean/<experiment>/passed/<group>.tar.gz        <- the best 3 per protein
    sorted_clean/<experiment>/rejected/<group>.tar.gz      <- the other 13
    tables/stage_03_results_<experiment>.csv               <- one row per sequence

HIGHER overall_confidence is better, and this is the sort that matters most in
the whole pipeline to get right. ProteinMPNN printed 'score', a negative log
likelihood, so the three LOWEST went forward. LigandMPNN prints
overall_confidence = exp(-loss), a probability, so the three HIGHEST do. Sorting
these the way this file used to would keep precisely the three worst designs of
every sixteen, and nothing downstream would look wrong -- the fastas would be
well formed, the folds would run, the RMSDs would come back. Only the yield
would quietly collapse. test_direction() at the bottom asserts it.

ligand_confidence is recorded beside it: the same quantity restricted to the
residues near the polysaccharide. It does not select -- a design that binds well
and folds badly is not wanted -- but it is the column to read when judging
whether designing in the ligand's presence achieved anything.

Ala and Gly content are recorded but never select: they are there because the
composition bias pushes against both, and the table is how you see whether it
worked.

passed/ and rejected/ hold the same shape as the source, <protein_id>/<id>.fa,
because that is what stage 04 reads. They are written directly rather than
through regroup_and_archive, which flattens member names.
"""
from __future__ import annotations

import argparse
import json
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

# LigandMPNN's default, unlike ProteinMPNN's '/'. Splitting on the wrong one
# silently returns the whole two-chain concatenation, so every length and
# composition figure would come out doubled but perfectly plausible.
CHAIN_SEPARATOR = ":"

RESULT_FIELDS = [
    "sequence_id",
    "protein_id",
    "experiment_name",
    "group",
    "time_stamp",
    "overall_confidence",
    "ligand_confidence",
    "seq_rec",
    "length",
    "ala_content",
    "gly_content",
    "other_content",
    "status",
]


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


def _field(header: str, name: str) -> Optional[float]:
    """A named float out of a LigandMPNN fasta header, or None if absent."""
    match = re.search(rf"\b{re.escape(name)}\s*=\s*([0-9eE.+-]+)", header)
    return float(match.group(1)) if match else None


def parse_confidence(header: str) -> float:
    value = _field(header, "overall_confidence")
    if value is None:
        raise SelectionError(f"no overall_confidence in fasta header: {header!r}")
    return value


def _fmt(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.6g}"


def composition(sequence: str) -> Tuple[int, float, float, float]:
    """(length, Ala fraction, Gly fraction, everything else) of ONE chain.

    The chains are tied and therefore identical, so the first is measured
    rather than the concatenation -- otherwise the length would read double.
    """
    chain = sequence.split(CHAIN_SEPARATOR)[0]
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


def redesign_specs(stage: Path, experiment: str, group_key: str) -> Dict[str, dict]:
    """{protein_id: spec record} from the group's redesign.jsonl, if it is there.

    Missing is not an error: a group designed before the spec travelled, or one
    whose job_runs/ has already been cleaned, simply sends its designs without it.
    """
    path = jp.redesign_spec_path(stage, experiment, group_key)
    records: Dict[str, dict] = {}
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = record.get("id")
            if key:
                records[key] = record
    return records


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
                confidence = parse_confidence(header)
                length, ala, gly, other = composition(sequence)
                scored.append((confidence, sequence_id, name))
                by_id[sequence_id] = {
                    "sequence_id": sequence_id,
                    "protein_id": protein_id,
                    "experiment_name": experiment,
                    "group": group_key,
                    "time_stamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "overall_confidence": f"{confidence:.6g}",
                    "ligand_confidence": _fmt(_field(header, "ligand_confidence")),
                    "seq_rec": _fmt(_field(header, "seq_rec")),
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
            # HIGHEST overall_confidence first -- see the module docstring; this
            # is the inverse of the ProteinMPNN sort this replaced. sequence_id
            # breaks ties so the choice does not depend on the order the archive
            # happened to list members in.
            scored.sort(key=lambda item: (-item[0], item[1]))
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

    # The complex the sequences were designed on travels with them. Stage 04
    # folds against its ligand and measures backbone RMSD against its protein,
    # and reaching back into inputs_prepared/ would break the moment stage 03
    # is cleaned. One copy per protein, not per sequence.
    prepared = jp.prepared_dir(stage, experiment, group_key)
    specs = redesign_specs(stage, experiment, group_key)
    for protein_id in sorted({info.name.split("/")[0] for info, _ in passed_members}):
        complex_file = prepared / f"{protein_id}.pdb"
        if not complex_file.is_file():
            _log(f"[select] {protein_id}: no prepared complex at {complex_file}, "
                 f"sequences travel without it -- stage 04 will have no ligand to fold "
                 f"against")
            continue
        payload = complex_file.read_bytes()
        info = tarfile.TarInfo(f"{protein_id}/{protein_id}.pdb")
        info.size = len(payload)
        passed_members.append((info, payload))

        # The spec travels too: which residues were free, which were held, which
        # ligand chain was kept. It is the only record of what this design was
        # allowed to be, and it lives in job_runs/, which cleanup deletes.
        record = specs.get(protein_id)
        if record is not None:
            payload = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
            info = tarfile.TarInfo(f"{protein_id}/{protein_id}_redesign.json")
            info.size = len(payload)
            passed_members.append((info, payload))

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
                       lambda row: (row.get("protein_id", ""),
                                    -float(row.get("overall_confidence") or 0)))
            report.tables.append(table_path)
    return report


def test_direction() -> None:
    """Assert that the HIGHEST overall_confidence is what survives.

    Runs the real select_group over a synthetic group rather than re-checking
    the comparison by hand, so what is asserted is the sort that actually ships.
    If someone ever restores the ProteinMPNN ordering, this fails loudly instead
    of the pipeline quietly keeping the worst three of every sixteen.
    """
    import tempfile

    confidences = {"d01": 0.11, "d02": 0.93, "d03": 0.47, "d04": 0.88}
    with tempfile.TemporaryDirectory() as raw:
        stage = Path(raw) / "03_ligandmpnn"
        source = jp.mpnn_outputs_path(stage, "EXP", "grp")
        source.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(source, "w:gz") as archive:
            for index, (design_id, value) in enumerate(sorted(confidences.items()), 1):
                header = (f"id={index}, T=0.2, seed=37, overall_confidence={value}, "
                          f"ligand_confidence={value / 2:.4f}, seq_rec=0.25")
                payload = f">{header}\nMKV{CHAIN_SEPARATOR}MKV\n".encode("utf-8")
                info = tarfile.TarInfo(f"p001/{design_id}.fa")
                info.size = len(payload)
                archive.addfile(info, __import__("io").BytesIO(payload))

        by_id: Dict[str, Dict[str, str]] = {}
        select_group(stage, "EXP", "grp", by_id, SelectReport(), keep=2)

    kept = {name for name, row in by_id.items() if row["status"] == "KEPT"}
    expected = {"d02", "d04"}          # 0.93 and 0.88, the two highest
    assert kept == expected, (
        f"selection kept {sorted(kept)}, expected {sorted(expected)} -- the sort "
        f"is inverted, so the WORST designs are going forward"
    )
    assert by_id["d01"]["status"] == "NOT_KEPT", "the lowest confidence was kept"
    # A two-chain sequence must measure one chain, not the concatenation.
    assert by_id["d02"]["length"] == "3", (
        f"length read {by_id['d02']['length']}, expected 3 -- the chain separator "
        f"is wrong, so composition is being measured over both chains"
    )
    assert by_id["d02"]["ligand_confidence"] == "0.465"
    print("[self-test] highest overall_confidence is kept; one chain measured; ok")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--keep", type=int, default=KEEP_BEST,
                        help=f"sequences to keep per protein (default: {KEEP_BEST})")
    parser.add_argument("--self-test", action="store_true",
                        help="check the selection direction and exit")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        test_direction()
        return 0
    try:
        report = run_selection(args.stage.resolve(), args.experiment, args.keep)
    except (OSError, SelectionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    _log(f"[done] {report.summary()}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
