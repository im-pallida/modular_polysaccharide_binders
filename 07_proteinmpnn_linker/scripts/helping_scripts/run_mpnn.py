#!/usr/bin/env python3
"""
Stage 07, step 2: design the linker with vanilla ProteinMPNN.

    inputs_prepared/<experiment>/<group>/*.pdb       <- linker_prepare wrote
    outputs/<experiment>/<group>.tar.gz              <- this writes, <id>/<id>NN.fa

Three ProteinMPNN invocations, in order:

    parse_multiple_chains.py      the constructs -> parsed.jsonl
    assign_fixed_chains.py        the fused chain, designable
    protein_mpnn_run.py           with the fixed and tied positions
                                  linker_prepare computed, and the
                                  composition bias

LigandMPNN designed the copies at stage 03 because they had a fibril to bind.
The linker binds nothing -- it only has to get from one lobe to the other -- so
vanilla ProteinMPNN is the right model here, and passing a ligand it cannot use
would only mean carrying one.

Before the full batch, TWO THINGS are verified on one structure, from two
sequences generated into _preflight/:

    the fixing   -- the held positions came back unchanged. This is the only
                    thing standing between "design the linker" and "redesign
                    the whole construct": if --fixed_positions_jsonl silently
                    failed, the two validated copies would be overwritten by a
                    model that never saw the fibril.

    the symmetry -- the two copies came back identical. linker_prepare mirrors
                    the linker's shell into both and ties each offset to its
                    twin, but same-chain tying is an assumption about how
                    ProteinMPNN groups tied positions, and an assumption that
                    silently fails is worth nothing. Untied, the mirrored
                    positions are free to diverge, which is exactly the
                    asymmetry this stage was producing before.

Both failures look identical downstream -- a bad RMSD at stage 08, long after
the cause -- so both are caught here, for the price of two sequences.

The chain names are read from the prepared constructs rather than assumed, and
MPNN itself is located through MPNN_ROOT.
"""
from __future__ import annotations

import argparse
import io
import json
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
    ) if not (root / name).is_file()]
    if missing:
        raise MpnnError(f"MPNN_ROOT={root} is missing {missing}")
    return root


def prepared_chains(prepared_dir: Path) -> List[str]:
    """The protein chain names, read from a prepared construct.

    Read rather than assumed: --chain_list has to name the chains that are
    actually there, and nothing guarantees what RFD3 called them.
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
            f"linker_prepare should have stripped them"
        )
    # One chain, because RFD3 fused the two copies into one. Stage 03 demanded
    # exactly two here, for a dimer it was about to tie; demanding that now
    # would reject every construct this stage exists to design.
    if len(protein) != 1:
        raise MpnnError(
            f"{backbones[0].name} has {len(protein)} protein chain(s) {protein}; "
            f"a fused construct should be one -- if RFD3 left the copies as "
            f"separate chains, the linker did not join them"
        )
    return protein


def _run(command: Sequence[str], label: str) -> None:
    _log("$ " + " ".join(str(part) for part in command))
    proc = subprocess.run([str(part) for part in command])
    if proc.returncode != 0:
        raise MpnnError(f"{label} exited with code {proc.returncode}")


def build_jsonls(root: Path, prepared_dir: Path, work_dir: Path,
                 chains: Sequence[str]) -> Tuple[Path, Path]:
    """parsed then assigned.

    The tied positions are NOT built here. Stage 03 could use MPNN's own
    --homooligomer 1, which ties chain A to chain B position for position;
    RFD3 has since fused those copies into one chain, so there is no second
    chain to tie to and that helper has nothing to work with. linker_prepare
    writes the ties instead, within the single chain, because only it knows
    where one copy ends and the other begins.
    """
    parsed = work_dir / "parsed.jsonl"
    assigned = work_dir / "assigned.jsonl"

    _run([sys.executable, root / "helper_scripts" / "parse_multiple_chains.py",
          f"--input_path={prepared_dir}/", f"--output_path={parsed}"], "parse_multiple_chains")
    _run([sys.executable, root / "helper_scripts" / "assign_fixed_chains.py",
          f"--input_path={parsed}", f"--output_path={assigned}",
          "--chain_list", " ".join(chains)], "assign_fixed_chains")
    return parsed, assigned


def invoke_mpnn(root: Path, parsed: Path, assigned: Path,
                fixed_positions: Path, tied_positions: Path, bias: Path,
                out_folder: Path, num_seq: int) -> None:
    out_folder.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, root / "protein_mpnn_run.py",
        "--jsonl_path", parsed,
        "--chain_id_jsonl", assigned,
        "--fixed_positions_jsonl", fixed_positions,
        "--tied_positions_jsonl", tied_positions,
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


AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def input_sequence(pdb_path: Path) -> str:
    """The construct's own sequence, read straight off its backbone."""
    structure = gemmi.read_structure(str(pdb_path))
    structure.setup_entities()
    letters: List[str] = []
    for chain in structure[0]:
        for residue in chain:
            info = gemmi.find_tabulated_residue(residue.name)
            if info is not None and info.is_amino_acid():
                letters.append(AA3TO1.get(residue.name, "X"))
    return "".join(letters)


def held_indices(fixed_positions: Path, design_name: str) -> List[int]:
    """Zero-based offsets into the designed sequence that must not change.

    RFD3 fuses the two copies into ONE chain, so a residue's number is its
    position and the arithmetic is just number - 1. More than one chain would
    need each chain's length to offset the next, and rather than guess at that
    silently this refuses: a multi-chain construct is not what stage 06 built.
    """
    payload = json.loads(fixed_positions.read_text(encoding="utf-8"))
    entry = payload.get(design_name) or {}
    if len(entry) > 1:
        raise MpnnError(
            f"{design_name} has {len(entry)} chains {sorted(entry)}; the fused "
            f"construct should be one, so the held positions cannot be located "
            f"in the designed sequence"
        )
    numbers = next(iter(entry.values()), [])
    return sorted(int(number) - 1 for number in numbers if int(number) >= 1)


def verify_fixing(out_folder: Path, prepared_dir: Path,
                  fixed_positions: Path) -> None:
    """Check the held residues really did come back unchanged.

    --fixed_positions_jsonl is the only thing standing between "design the
    linker" and "redesign the whole construct". If it silently fails to apply,
    the two copies that were designed with the ligand, folded and validated get
    overwritten by a model that never saw the fibril -- and the first sign of it
    is a bad RMSD at stage 08, long after the cause.

    So the sequence is compared against the input at exactly those positions.
    """
    fastas = sorted((out_folder / "seqs").glob("*.fa"))
    if not fastas:
        raise MpnnError(f"no fastas in {out_folder / 'seqs'} after the pre-flight run")
    design_name = fastas[0].stem
    pdb_path = prepared_dir / f"{design_name}.pdb"
    if not pdb_path.is_file():
        raise MpnnError(f"no prepared construct at {pdb_path} to compare against")

    native = input_sequence(pdb_path)
    held = held_indices(fixed_positions, design_name)
    if not held:
        raise MpnnError(
            f"{fixed_positions.name} holds no fixed positions for {design_name}, "
            f"so every residue would be redesigned"
        )

    designed = designed_records(read_fasta(fastas[0].read_text(encoding="utf-8")))
    if not designed:
        raise MpnnError(
            f"{fastas[0].name}: no designed sequence found, so the fixing could not "
            f"be checked -- expected records headed with 'sample='"
        )

    for _, sequence in designed:
        joined = sequence.replace("/", "")
        if len(joined) != len(native):
            raise MpnnError(
                f"{fastas[0].name}: designed sequence is {len(joined)} residues but "
                f"the construct is {len(native)}; they cannot be compared"
            )
        changed = [index for index in held
                   if index < len(joined) and joined[index] != native[index]]
        if changed:
            shown = ", ".join(
                f"{index + 1}:{native[index]}->{joined[index]}" for index in changed[:5]
            )
            raise MpnnError(
                f"{fastas[0].name}: {len(changed)} held residue(s) were redesigned, "
                f"so --fixed_positions_jsonl did not take effect ({shown})"
            )
    _log(f"  pre-flight: {len(designed)} sequence(s) checked, {len(held)} held "
         f"position(s) unchanged -- the fixing is in effect")


def copy_layout(spec_path: Path, design_name: str) -> Tuple[int, int, int]:
    """(residues per copy, where copy A starts, where copy B starts), 1-based.

    Read from linker_prepare's own record rather than recounted here. It found
    the copies by splitting the chain on the diffused run; deriving them a
    second time, by a second method, is how the two quietly disagree -- and the
    check below would then be comparing the wrong two slices and passing.
    """
    if not spec_path.is_file():
        raise MpnnError(
            f"no {spec_path.name} beside the run, so where the two copies sit "
            f"is unknown and the symmetry cannot be checked -- re-run "
            f"linker_prepare for this group"
        )
    record: Optional[dict] = None
    for line in spec_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Appended to on every run, so the last entry for this design wins.
        if str(payload.get("id", "")) == design_name and payload.get("copies"):
            record = payload
    if record is None:
        raise MpnnError(
            f"{spec_path.name} has no copy layout for {design_name}; it was "
            f"written by an older linker_prepare that did not mirror the shell, "
            f"so re-run that step before designing"
        )
    copies = record["copies"]
    return int(copies["length"]), int(copies["a_start"]), int(copies["b_start"])


def verify_symmetry(out_folder: Path, prepared_dir: Path,
                    spec_path: Path) -> None:
    """Check the two copies really did come out identical.

    The shell was mirrored into both copies and each offset tied to its twin,
    so every position that was free to change was free in both and should have
    decoded to the same residue. If same-chain tying did not take effect, the
    mirrored positions were merely designed twice, independently, and the two
    units diverge -- the construct is then no longer two copies of one binder,
    which is the whole premise of the design.

    ProteinMPNN is not importable here, so the tying cannot be tested directly.
    It is asserted on its output instead, once, before the batch.
    """
    fastas = sorted((out_folder / "seqs").glob("*.fa"))
    if not fastas:
        raise MpnnError(f"no fastas in {out_folder / 'seqs'} after the pre-flight run")
    design_name = fastas[0].stem
    length, a_start, b_start = copy_layout(spec_path, design_name)

    designed = designed_records(read_fasta(fastas[0].read_text(encoding="utf-8")))
    if not designed:
        raise MpnnError(
            f"{fastas[0].name}: no designed sequence found, so the symmetry could "
            f"not be checked -- expected records headed with 'sample='"
        )
    # The input first: two copies that already differ cannot be made to agree,
    # and saying so is a different diagnosis from "the tying did not apply".
    candidates = [("the construct itself", input_sequence(
        prepared_dir / f"{design_name}.pdb"))]
    candidates += [(f"sequence {index}", text)
                   for index, (_, text) in enumerate(designed, start=1)]

    for label, sequence in candidates:
        joined = sequence.replace("/", "")
        first = joined[a_start - 1: a_start - 1 + length]
        second = joined[b_start - 1: b_start - 1 + length]
        if len(first) != length or len(second) != length:
            raise MpnnError(
                f"{fastas[0].name}: {label} is {len(joined)} residues, too short "
                f"for two copies of {length} starting at {a_start} and {b_start}"
            )
        if first == second:
            continue
        differ = [offset for offset in range(length) if first[offset] != second[offset]]
        shown = ", ".join(
            f"offset {offset + 1}: {first[offset]} vs {second[offset]}"
            for offset in differ[:5]
        )
        if label == "the construct itself":
            raise MpnnError(
                f"{design_name}: the construct's own two copies differ at "
                f"{len(differ)} position(s) ({shown}), so no design can make "
                f"them identical -- linker_prepare should have refused this"
            )
        raise MpnnError(
            f"{fastas[0].name}: {label} has {len(differ)} position(s) where the "
            f"two copies differ ({shown}), so --tied_positions_jsonl did not "
            f"take effect and the units would not be identical"
        )
    _log(f"  pre-flight: both copies of {length} residue(s) identical in every "
         f"sequence -- the tying is in effect")


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
                        f"  run linker_prepare.py first")
    tied_positions = jp.tied_positions_path(stage, experiment, group_key)
    if not tied_positions.is_file():
        raise MpnnError(f"no tied positions for this group: {tied_positions}\n"
                        f"  run linker_prepare.py first -- without the ties the "
                        f"two copies come out with different sequences")
    spec_path = jp.linker_spec_path(stage, experiment, group_key)
    bias = jp.bias_aa_path(stage)
    if not bias.is_file():
        raise MpnnError(f"no composition bias file: {bias}")

    chains = prepared_chains(prepared)
    _log(f"[mpnn] {experiment}/{group_key}: chains {chains}, "
         f"{len(list(prepared.glob('*.pdb')))} backbone(s)")

    parsed, assigned = build_jsonls(root, prepared, work_dir, chains)

    _log("[mpnn] pre-flight: verifying the fixing and the tying on this group "
         "before the full batch")
    check_folder = work_dir / "_preflight"
    invoke_mpnn(root, parsed, assigned, fixed_positions, tied_positions, bias,
                check_folder, 2)
    verify_fixing(check_folder, prepared, fixed_positions)
    verify_symmetry(check_folder, prepared, spec_path)

    out_folder = work_dir / "out"
    invoke_mpnn(root, parsed, assigned, fixed_positions, tied_positions, bias,
                out_folder, num_seq)

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
