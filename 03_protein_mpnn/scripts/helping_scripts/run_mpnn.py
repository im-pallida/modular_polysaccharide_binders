#!/usr/bin/env python3
"""
Stage 03, step 2: run ProteinMPNN over one group, symmetrically.

    inputs_prepared/<experiment>/<group>/*.pdb       <- mpnn_prepare wrote
    outputs/<experiment>/<group>.tar.gz              <- this writes, <pid>/<pid>NN.fa

Four ProteinMPNN invocations, in order:

    parse_multiple_chains.py      the backbones -> parsed.jsonl
    assign_fixed_chains.py        both chains designable, each conditioned on
                                  the other
    make_tied_positions_dict.py   --homooligomer 1: chain A and chain B tied, so
                                  one sequence is designed and applied to both.
                                  This is what makes the design symmetric
    protein_mpnn_run.py           with the tied positions, the fixed positions
                                  mpnn_prepare computed, and the composition bias

Before the full batch, the tying is verified on a single structure: two
sequences are generated and the two chains compared. A silent failure there
would produce a whole group of asymmetric designs that look fine until stage 04.

The chain names are read from the prepared backbones rather than assumed, and
MPNN itself is located through MPNN_ROOT.
"""
from __future__ import annotations

import argparse
import io
import os
import re
import subprocess
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import gemmi

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import job_paths as jp  # noqa: E402
from archives import merge_members_into_archive  # noqa: E402
from structures import classify_chains  # noqa: E402

STAGE = Path(__file__).resolve().parents[2]

NUM_SEQ = 16
SAMPLING_TEMP = "0.2"
BACKBONE_NOISE = "0.1"
SEED = "37"
MODEL_NAME = "v_48_020"


class MpnnError(Exception):
    """ProteinMPNN could not be run, or produced nothing usable."""


def _log(*parts: object) -> None:
    print(*parts, flush=True)


@dataclass
class RunResult:
    experiment: str
    group_key: str
    archive: Path
    proteins: int = 0
    sequences: int = 0
    chains: List[str] = field(default_factory=list)


def mpnn_root() -> Path:
    """The ProteinMPNN install, from MPNN_ROOT, checked before anything runs."""
    raw = os.environ.get("MPNN_ROOT", "").strip()
    if not raw:
        raise MpnnError(
            "MPNN_ROOT is not set -- point it at the ProteinMPNN checkout "
            "(the folder holding protein_mpnn_run.py)"
        )
    root = Path(raw)
    missing = [name for name in (
        "protein_mpnn_run.py",
        "helper_scripts/parse_multiple_chains.py",
        "helper_scripts/assign_fixed_chains.py",
        "helper_scripts/make_tied_positions_dict.py",
    ) if not (root / name).is_file()]
    if missing:
        raise MpnnError(f"MPNN_ROOT={root} is missing {missing}")
    return root


def prepared_chains(prepared_dir: Path) -> List[str]:
    """The protein chain names, read from a prepared backbone.

    Read rather than assumed: the tying and the chain list have to name the
    chains that are actually there, and nothing guarantees they are 'A' and 'B'.
    """
    backbones = sorted(prepared_dir.glob("*.pdb"))
    if not backbones:
        raise MpnnError(f"no prepared backbones in {prepared_dir}")
    structure = gemmi.read_structure(str(backbones[0]))
    structure.setup_entities()
    protein, ligand = classify_chains(structure)
    if ligand:
        raise MpnnError(
            f"{backbones[0].name} still carries non-protein chain(s) {ligand}; "
            f"mpnn_prepare should have stripped them"
        )
    if len(protein) != 2:
        raise MpnnError(f"{backbones[0].name} has {len(protein)} protein chain(s) {protein}")
    return protein


def _run(command: Sequence[str], label: str) -> None:
    _log("$ " + " ".join(str(part) for part in command))
    proc = subprocess.run([str(part) for part in command])
    if proc.returncode != 0:
        raise MpnnError(f"{label} exited with code {proc.returncode}")


def build_jsonls(root: Path, prepared_dir: Path, work_dir: Path,
                 chains: Sequence[str]) -> Tuple[Path, Path, Path]:
    """parsed / assigned / tied, in that order -- each feeds the next."""
    parsed = work_dir / "parsed.jsonl"
    assigned = work_dir / "assigned.jsonl"
    tied = work_dir / "tied.jsonl"

    _run([sys.executable, root / "helper_scripts" / "parse_multiple_chains.py",
          f"--input_path={prepared_dir}/", f"--output_path={parsed}"], "parse_multiple_chains")
    _run([sys.executable, root / "helper_scripts" / "assign_fixed_chains.py",
          f"--input_path={parsed}", f"--output_path={assigned}",
          "--chain_list", " ".join(chains)], "assign_fixed_chains")
    _run([sys.executable, root / "helper_scripts" / "make_tied_positions_dict.py",
          f"--input_path={parsed}", f"--output_path={tied}",
          "--homooligomer", "1"], "make_tied_positions_dict")
    return parsed, assigned, tied


def invoke_mpnn(root: Path, parsed: Path, assigned: Path, tied: Path,
                fixed_positions: Path, bias: Path, out_folder: Path,
                num_seq: int) -> None:
    out_folder.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, root / "protein_mpnn_run.py",
        "--jsonl_path", parsed,
        "--chain_id_jsonl", assigned,
        "--fixed_positions_jsonl", fixed_positions,
        "--tied_positions_jsonl", tied,
        "--bias_AA_jsonl", bias,
        "--out_folder", out_folder,
        "--num_seq_per_target", str(num_seq),
        "--batch_size", str(num_seq),
        "--sampling_temp", SAMPLING_TEMP,
        "--backbone_noise", BACKBONE_NOISE,
        "--seed", SEED,
        "--path_to_model_weights", root / "vanilla_model_weights",
        "--model_name", MODEL_NAME,
    ]
    _run(command, "protein_mpnn_run")


def read_fasta(text: str) -> List[Tuple[str, str]]:
    """[(header, sequence)] from one ProteinMPNN fasta."""
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

    ProteinMPNN writes the input sequence as the first record and the designs
    after it, each headed with "sample=N". The native record also carries the
    chain separator, so filtering on that alone would count it as a design and
    every protein would come out with one sequence too many.
    """
    return [(header, seq) for header, seq in records if "sample=" in header.lower()]


def verify_tying(out_folder: Path, chains: Sequence[str]) -> None:
    """Check that the two tied chains really did come out identical.

    ProteinMPNN writes the designed chains separated by '/'. If tying silently
    failed, the halves differ -- and a whole group of asymmetric designs would
    otherwise look fine until stage 04 tried to fold them.
    """
    fastas = sorted((out_folder / "seqs").glob("*.fa"))
    if not fastas:
        raise MpnnError(f"no fastas in {out_folder / 'seqs'} after the pre-flight run")
    designed = designed_records(read_fasta(fastas[0].read_text(encoding="utf-8")))
    if not designed:
        raise MpnnError(
            f"{fastas[0].name}: no designed sequence found, so the tying could not "
            f"be checked -- expected records headed with 'sample='"
        )
    for header, sequence in designed:
        halves = sequence.split("/")
        if len(set(halves)) != 1:
            raise MpnnError(
                f"{fastas[0].name}: the {len(halves)} designed chains are NOT identical, "
                f"so --homooligomer tying did not take effect. Lengths "
                f"{[len(half) for half in halves]}"
            )
    _log(f"  pre-flight: {len(designed)} sequence(s) checked, "
         f"{len(chains)} chains identical -- tying is in effect")


def split_and_archive(out_folder: Path, archive_path: Path) -> Tuple[int, int]:
    """One fasta per designed sequence, nested <protein_id>/<sequence_id>.fa.

    That nesting is what stage 04 reads, so it is preserved exactly. The
    archive is extended rather than rewritten, like every other in this project.
    """
    members: List[Tuple[tarfile.TarInfo, bytes]] = []
    proteins = 0
    for fasta in sorted((out_folder / "seqs").glob("*.fa")):
        protein_id = fasta.stem
        designed = designed_records(read_fasta(fasta.read_text(encoding="utf-8")))
        if not designed:
            _log(f"  [warn] {fasta.name}: no designed sequences, skipped")
            continue
        proteins += 1
        for index, (header, sequence) in enumerate(designed, start=1):
            name = f"{protein_id}/{protein_id}{index:02d}.fa"
            payload = f">{header}\n{sequence}\n".encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            members.append((info, payload))

    if not members:
        raise MpnnError(f"no designed sequences anywhere under {out_folder / 'seqs'}")

    existing = set()
    if archive_path.is_file():
        with tarfile.open(archive_path, "r:gz") as archive:
            existing = set(archive.getnames())
    fresh = [(info, payload) for info, payload in members if info.name not in existing]
    if fresh:
        merge_members_into_archive(archive_path, fresh)
    return proteins, len(fresh)


def run_group(stage: Path, experiment: str, group_key: str,
              num_seq: int = NUM_SEQ) -> RunResult:
    root = mpnn_root()
    prepared = jp.prepared_dir(stage, experiment, group_key)
    work_dir = jp.mpnn_work_dir(stage, experiment, group_key)
    fixed_positions = jp.fixed_positions_path(stage, experiment, group_key)
    if not fixed_positions.is_file():
        raise MpnnError(f"no fixed positions for this group: {fixed_positions}\n"
                        f"  run mpnn_prepare.py first")
    bias = jp.bias_aa_path(stage)
    if not bias.is_file():
        raise MpnnError(f"no composition bias file: {bias}")

    chains = prepared_chains(prepared)
    _log(f"[mpnn] {experiment}/{group_key}: chains {chains}, "
         f"{len(list(prepared.glob('*.pdb')))} backbone(s)")

    parsed, assigned, tied = build_jsonls(root, prepared, work_dir, chains)

    _log("[mpnn] pre-flight: verifying the tying on this group before the full batch")
    check_folder = work_dir / "_preflight"
    invoke_mpnn(root, parsed, assigned, tied, fixed_positions, bias, check_folder, 2)
    verify_tying(check_folder, chains)

    out_folder = work_dir / "out"
    invoke_mpnn(root, parsed, assigned, tied, fixed_positions, bias, out_folder, num_seq)

    archive_path = jp.mpnn_outputs_path(stage, experiment, group_key)
    proteins, sequences = split_and_archive(out_folder, archive_path)
    return RunResult(experiment, group_key, archive_path, proteins, sequences, chains)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("experiment")
    parser.add_argument("group_key")
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--num-seq", type=int, default=NUM_SEQ)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = run_group(args.stage.resolve(), args.experiment, args.group_key, args.num_seq)
    except MpnnError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    _log(f"[mpnn] {result.proteins} protein(s), {result.sequences} new sequence(s) "
         f"-> {result.archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

