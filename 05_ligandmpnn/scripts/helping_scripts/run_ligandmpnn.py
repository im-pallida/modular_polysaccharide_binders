#!/usr/bin/env python3
"""
Stage 05, step 2: redesign the ligand-facing shell with LigandMPNN.

    inputs_prepared/<experiment>/<group>/<sequence_id>.pdb   <- ligand_prepare wrote
    job_runs/ligandmpnn/<experiment>/<group>/redesign.jsonl  <- which residues are free
    outputs/<experiment>/<group>.tar.gz                      <- <sequence_id>/<id>NN.fa

One invocation per complex, not one per group. LigandMPNN reads
--symmetry_residues once and applies it to every structure in a multi-pdb run,
and the 8 A shell differs from one complex to the next -- so a batched run would
tie the wrong positions together. --homo_oligomer would derive the ties per
structure instead, but it assumes the ligand chain never reaches the chain list,
which is not worth betting a batch on. Per structure also means one bad complex
costs one complex.

The flags mirror stage 03's ProteinMPNN run: the same composition bias, the same
16 sequences, the two chains tied so the design stays symmetric. What differs is
that the ligand is in the room -- atom context on, and the scaffolded side chains
offered as context too, since those are the residues that make the binding site.

Before the batch, the tying is verified on the first complex: two sequences are
generated and the chains compared. Untied designs would not be caught until
stage 06 tried to link two different chains together.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import job_paths as jp  # noqa: E402
from archives import merge_members_into_archive  # noqa: E402

STAGE = Path(__file__).resolve().parents[2]

NUM_SEQ = 16
TEMPERATURE = "0.2"
SEED = "37"
BIAS_AA = "A:-1.0,G:-0.8"
DEFAULT_CHECKPOINT = "model_params/ligandmpnn_v_32_010_25.pt"
CHAIN_SEPARATOR = ":"   # LigandMPNN's default, unlike ProteinMPNN's '/'


class LigandMpnnError(Exception):
    """LigandMPNN could not be run, or produced nothing usable."""


def _log(*parts: object) -> None:
    print(*parts, flush=True)


@dataclass
class RunReport:
    designed: int = 0
    sequences: int = 0
    already: int = 0
    archives: List[Path] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        if not self.designed:
            return f"ligandmpnn: nothing new ({self.already} already designed)"
        return (f"ligandmpnn: {self.designed} complex(es), {self.sequences} sequence(s), "
                f"{self.already} already designed")


def ligandmpnn_root() -> Path:
    raw = os.environ.get("LIGANDMPNN_ROOT", "").strip()
    if not raw:
        raise LigandMpnnError(
            "LIGANDMPNN_ROOT is not set -- point it at the LigandMPNN checkout "
            "(the folder holding run.py)"
        )
    root = Path(raw)
    if not (root / "run.py").is_file():
        raise LigandMpnnError(f"LIGANDMPNN_ROOT={root} has no run.py")
    return root


def checkpoint_path(root: Path) -> Path:
    """The ligand_mpnn weights, from LIGANDMPNN_CHECKPOINT or the usual place."""
    raw = os.environ.get("LIGANDMPNN_CHECKPOINT", "").strip()
    path = Path(raw) if raw else root / DEFAULT_CHECKPOINT
    if not path.is_file():
        raise LigandMpnnError(
            f"no LigandMPNN checkpoint at {path} -- set LIGANDMPNN_CHECKPOINT, or "
            f"download the weights into {root / 'model_params'}"
        )
    return path


def read_specs(path: Path) -> Dict[str, dict]:
    """{sequence_id: record} from redesign.jsonl. Later lines win."""
    specs: Dict[str, dict] = {}
    if not path.is_file():
        return specs
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                specs[record["sequence_id"]] = record
            except (json.JSONDecodeError, KeyError) as exc:
                raise LigandMpnnError(f"{path}:{number}: malformed record: {exc}")
    return specs


def invoke(root: Path, checkpoint: Path, pdb_path: Path, spec: dict,
           out_folder: Path, num_seq: int) -> None:
    out_folder.mkdir(parents=True, exist_ok=True)
    groups = spec["symmetry_residues"]
    weights = "|".join(",".join(["0.5"] * len(group.split(","))) for group in groups)
    command = [
        sys.executable, str(root / "run.py"),
        "--model_type", "ligand_mpnn",
        "--checkpoint_ligand_mpnn", str(checkpoint),
        "--pdb_path", str(pdb_path),
        "--out_folder", str(out_folder),
        "--redesigned_residues", " ".join(spec["redesigned_residues"]),
        "--symmetry_residues", "|".join(groups),
        "--symmetry_weights", weights,
        "--bias_AA", BIAS_AA,
        "--temperature", TEMPERATURE,
        "--seed", SEED,
        "--batch_size", str(num_seq),
        "--number_of_batches", "1",
        # The ligand is the whole reason this stage exists, and the scaffolded
        # side chains are the binding site -- both belong in the context.
        "--ligand_mpnn_use_atom_context", "1",
        "--ligand_mpnn_use_side_chain_context", "1",
    ]
    _log("$ " + " ".join(command))
    completed = subprocess.run(command, cwd=str(root))
    if completed.returncode != 0:
        raise LigandMpnnError(f"run.py exited with code {completed.returncode}")


def read_fasta(text: str) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    header: Optional[str] = None
    parts: List[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(parts)))
            header, parts = line[1:].strip(), []
        elif line.strip():
            parts.append(line.strip())
    if header is not None:
        records.append((header, "".join(parts)))
    return records


def designed_records(records: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Just the designs.

    LigandMPNN writes the input sequence first, headed with T=/seed=/num_res=,
    then the designs, each headed with 'id='. Filtering on 'id=' is what keeps
    the native sequence from being counted and scored as a seventeenth design.
    """
    return [(header, sequence) for header, sequence in records
            if "id=" in header.lower()]


def output_fasta(out_folder: Path, pdb_path: Path) -> Path:
    expected = out_folder / "seqs" / f"{pdb_path.stem}.fa"
    if expected.is_file():
        return expected
    candidates = sorted(out_folder.rglob("*.fa"))
    if not candidates:
        raise LigandMpnnError(
            f"no fasta under {out_folder} -- it contains "
            f"{sorted(p.name for p in out_folder.iterdir())[:8]}"
        )
    return candidates[0]


def verify_tying(fasta_path: Path) -> int:
    """Check the two designed chains really did come out identical."""
    designed = designed_records(read_fasta(fasta_path.read_text(encoding="utf-8")))
    if not designed:
        raise LigandMpnnError(
            f"{fasta_path.name}: no designed sequence found, so the tying could not "
            f"be checked -- expected records headed with 'id='"
        )
    for _, sequence in designed:
        chains = sequence.split(CHAIN_SEPARATOR)
        if len(chains) < 2:
            raise LigandMpnnError(
                f"{fasta_path.name}: expected two chains separated by "
                f"{CHAIN_SEPARATOR!r}, got one -- check --fasta_seq_separation"
            )
        if len(set(chains)) != 1:
            raise LigandMpnnError(
                f"{fasta_path.name}: the {len(chains)} designed chains are NOT "
                f"identical, so --symmetry_residues did not take effect. Lengths "
                f"{[len(chain) for chain in chains]}"
            )
    return len(designed)


def design_group(stage: Path, experiment: str, group_key: str, report: RunReport,
                 num_seq: int = NUM_SEQ, preflight: bool = True) -> None:
    root = ligandmpnn_root()
    checkpoint = checkpoint_path(root)
    specs = read_specs(jp.redesign_spec_path(stage, experiment, group_key))
    if not specs:
        _log(f"[ligandmpnn] {experiment}/{group_key}: nothing prepared")
        return

    archive_path = jp.ligand_outputs_path(stage, experiment, group_key)
    already: set = set()
    if archive_path.is_file():
        with tarfile.open(archive_path, "r:gz") as archive:
            already = {name.split("/")[0] for name in archive.getnames() if "/" in name}

    work_dir = jp.ligand_work_dir(stage, experiment, group_key)
    members: List[Tuple[tarfile.TarInfo, bytes]] = []
    checked = not preflight

    for sequence_id, spec in sorted(specs.items()):
        if sequence_id in already:
            report.already += 1
            continue
        pdb_path = jp.complex_path(stage, experiment, group_key, sequence_id)
        if not pdb_path.is_file():
            report.problems.append(f"{experiment}/{sequence_id}: no complex at {pdb_path}")
            continue
        try:
            if not checked:
                _log(f"[ligandmpnn] pre-flight on {sequence_id}: verifying the tying "
                     f"before the group")
                check_folder = work_dir / "_preflight"
                invoke(root, checkpoint, pdb_path, spec, check_folder, 2)
                found = verify_tying(output_fasta(check_folder, pdb_path))
                _log(f"  pre-flight: {found} sequence(s) checked, chains identical "
                     f"-- tying is in effect")
                checked = True

            out_folder = work_dir / "out" / sequence_id
            invoke(root, checkpoint, pdb_path, spec, out_folder, num_seq)
            fasta_path = output_fasta(out_folder, pdb_path)
            designed = designed_records(read_fasta(fasta_path.read_text(encoding="utf-8")))
            if not designed:
                raise LigandMpnnError(f"{fasta_path.name}: no designed sequences")

            for index, (header, sequence) in enumerate(designed, start=1):
                name = f"{sequence_id}/{sequence_id}{index:02d}.fa"
                payload = f">{header}\n{sequence}\n".encode("utf-8")
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                members.append((info, payload))
            report.designed += 1
            report.sequences += len(designed)
            _log(f"[ligandmpnn] {sequence_id}: {len(designed)} design(s), "
                 f"{len(spec['symmetry_residues'])} position(s) redesigned")
        except LigandMpnnError as exc:
            report.problems.append(f"{experiment}/{sequence_id}: {exc}")
            _log(f"[ligandmpnn] {sequence_id}: FAILED ({exc})")

    if members:
        existing: set = set()
        if archive_path.is_file():
            with tarfile.open(archive_path, "r:gz") as archive:
                existing = set(archive.getnames())
        fresh = [(info, payload) for info, payload in members
                 if info.name not in existing]
        if fresh:
            added, total = merge_members_into_archive(archive_path, fresh)
            report.archives.append(archive_path)
            _log(f"[ligandmpnn] {group_key}: +{added} sequence(s), {total} total "
                 f"-> {archive_path}")


def run_design(stage: Path, experiment: Optional[str] = None,
               num_seq: int = NUM_SEQ) -> RunReport:
    report = RunReport()
    names = [experiment] if experiment else jp.inputs_experiments(stage)
    for experiment_name in names:
        directory = jp.inputs_root(stage) / experiment_name
        if not directory.is_dir():
            continue
        for archive_path in sorted(directory.glob("*.tar.gz")):
            group_key = archive_path.name[: -len(".tar.gz")]
            design_group(stage, experiment_name, group_key, report, num_seq)
    return report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--num-seq", type=int, default=NUM_SEQ)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = run_design(args.stage.resolve(), args.experiment, args.num_seq)
    except LigandMpnnError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    _log(f"[done] {report.summary()}")
    for problem in report.problems[:10]:
        _log(f"[done]   {problem}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

