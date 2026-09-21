#!/usr/bin/env python3
"""
Stage 04, step 1: build one AlphaFold3 input json per designed sequence.

    inputs/<experiment>/<group>.tar.gz              <- stage 03's transfer
    job_runs/af3/<experiment>/<group>/<sequence_id>_monomer.json

Monomer only. The designs are tied homooligomers -- stage 03 ran ProteinMPNN
with --homooligomer 1, so chain A and chain B carry the same sequence by
construction -- which is why folding the dimer told us about the predicted
chain ASSOCIATION rather than the fold. That association test never passed a
single sequence, so it is gone: one chain, one job.

No selection happens here. Stage 03 already kept the best three sequences per
protein and transferred exactly those, so whatever is in the archive is what
gets folded.

Idempotent twice over: a sequence whose json exists is skipped, and so is one
whose AF3 output already exists -- the second check is what stops a cleared
job_runs/ from re-queueing days of GPU time.
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import job_paths as jp  # noqa: E402
from archives import load_table  # noqa: E402

STAGE = Path(__file__).resolve().parents[2]


class PrepareError(Exception):
    """A group's jsons could not be built."""


def _log(*parts: object) -> None:
    print(*parts, flush=True)


@dataclass
class PrepareReport:
    written: int = 0
    already_built: int = 0
    already_folded: int = 0
    seen: int = 0
    chain_mismatches: List[str] = field(default_factory=list)
    malformed: List[str] = field(default_factory=list)
    jsons: List[Path] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.malformed

    def summary(self) -> str:
        return (f"prepare: {self.seen} sequence(s) seen, {self.written} json(s) written, "
                f"{self.already_built} already built, {self.already_folded} already folded")


def first_chain(content: bytes, member_name: str) -> Tuple[str, bool]:
    """(chain A's sequence, whether the two tied chains differed).

    ProteinMPNN writes the tied chains joined by '/'. They are expected to be
    identical; a difference means the tying did not take for that sequence, so
    it is reported rather than silently folded as if it had.
    """
    lines = [line for line in content.decode("utf-8").splitlines() if line.strip()]
    sequence_lines = [line for line in lines if not line.startswith(">")]
    if not sequence_lines:
        raise ValueError(f"{member_name}: no sequence line")
    chains = "".join(sequence_lines).split("/")
    if not chains[0]:
        raise ValueError(f"{member_name}: empty first chain")
    return chains[0], len(set(chains)) > 1


def build_af3_json(job_name: str, sequence: str) -> dict:
    """One protein chain 'A', no ligand, no MSA supplied (AF3 searches its own)."""
    return {
        "dialect": "alphafold3",
        "version": 2,
        "name": job_name,
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": sequence,
                    "modifications": [],
                    "unpairedMsa": "",
                    "pairedMsa": "",
                    "templates": [],
                }
            }
        ],
        "modelSeeds": [1],
        "userCCD": None,
    }


def scored_sequences(stage: Path, experiment: str) -> set:
    """Sequences already scored, which is the durable record that they folded.

    outputs/ used to be that record, but cleanup deletes it once the structures
    are archived. Keying on the table instead is what makes the deletion safe:
    a cleaned sequence is skipped, not refolded.
    """
    return set(load_table(jp.results_table_path(stage, experiment),
                          key="sequence_id"))


def prepare_group(stage: Path, experiment: str, group_key: str,
                  report: PrepareReport, scored: Optional[set] = None) -> None:
    archive_path = jp.input_archive_path(stage, experiment, group_key)
    if not archive_path.is_file():
        raise PrepareError(f"no input archive: {archive_path}")

    json_dir = jp.af3_json_dir(stage, experiment, group_key)
    json_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as archive:
        for member_name in sorted(archive.getnames()):
            if "/" not in member_name or not member_name.endswith(".fa"):
                continue
            sequence_id = Path(member_name).stem
            report.seen += 1

            job_name = jp.af3_job_name(sequence_id)
            if scored and sequence_id in scored:
                report.already_folded += 1
                continue
            json_path = jp.af3_json_path(stage, experiment, group_key, sequence_id)
            if json_path.is_file():
                report.already_built += 1
                continue
            if jp.af3_output_dir(stage, job_name).is_dir():
                report.already_folded += 1
                continue

            handle = archive.extractfile(member_name)
            if handle is None:
                report.malformed.append(member_name)
                continue
            try:
                sequence, mismatched = first_chain(handle.read(), member_name)
            except ValueError as exc:
                _log(f"[prepare] {exc} -- skipped")
                report.malformed.append(member_name)
                continue
            if mismatched:
                report.chain_mismatches.append(sequence_id)

            json_path.write_text(
                json.dumps(build_af3_json(job_name, sequence), indent=2) + "\n",
                encoding="utf-8",
            )
            report.written += 1
            report.jsons.append(json_path)


def run_prepare(stage: Path, experiment: Optional[str] = None) -> PrepareReport:
    report = PrepareReport()
    names = [experiment] if experiment else jp.inputs_experiments(stage)
    if not names:
        _log(f"[prepare] nothing handed over under {jp.inputs_root(stage)}")
        return report

    for experiment_name in names:
        directory = jp.inputs_root(stage) / experiment_name
        if not directory.is_dir():
            continue
        scored = scored_sequences(stage, experiment_name)
        for archive_path in sorted(directory.glob("*.tar.gz")):
            group_key = archive_path.name[: -len(".tar.gz")]
            prepare_group(stage, experiment_name, group_key, report, scored)

    if report.chain_mismatches:
        _log(f"[prepare] WARNING: {len(report.chain_mismatches)} sequence(s) had "
             f"non-identical tied chains, e.g. {report.chain_mismatches[:3]} -- "
             f"chain A was folded, but the tying is worth checking in stage 03")
    if report.malformed:
        _log(f"[prepare] WARNING: {len(report.malformed)} unreadable fasta(s), "
             f"e.g. {report.malformed[:3]}")
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
    try:
        report = run_prepare(args.stage.resolve(), args.experiment)
    except (PrepareError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    _log(f"[done] {report.summary()}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

