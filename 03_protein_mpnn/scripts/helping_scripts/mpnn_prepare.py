#!/usr/bin/env python3
"""
Stage 03, step 1: prepare one group for ProteinMPNN.

    inputs/<experiment>/<group>.tar.gz                  <- stage 02's transfer
    inputs_prepared/<experiment>/<group>/<pid>.pdb      <- backbones, ligand stripped
    job_runs/mpnn/<experiment>/<group>/fixed_positions.jsonl

Two things have to happen before MPNN can run.

The polysaccharide has to go. Vanilla ProteinMPNN has no notion of a ligand, and
parse_multiple_chains.py would treat the fibril as a third chain to design. It
is dropped by residue type rather than by chain letter, so cellulose and chitin
are both handled without either being named here.

And the scaffolded residues have to be named in MPNN's own terms. The design
config fixed certain side chains in the SEED's numbering; this structure has its
own. The pairing was recorded by stage 01 as the structure was generated, and is
derived from the config and the structure's metadata when no record exists --
for anything generated before the record did.

Prepared per group rather than into one flat folder: the old layout lost which
experiment a structure came from and had to recover it afterwards from a
committed map file, which is now unnecessary.
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gemmi

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import job_paths as jp  # noqa: E402
from archives import extract_member, group_members_by_protein  # noqa: E402
from designs import (  # noqa: E402
    DesignError,
    design_config_for_group,
    fixed_residue_map,
    generated_positions,
    read_records,
)
from structures import classify_chains, drop_chains  # noqa: E402

STAGE = Path(__file__).resolve().parents[2]


class PrepareError(Exception):
    """A group could not be prepared."""


def _log(*parts: object) -> None:
    print(*parts, flush=True)


@dataclass
class PrepareResult:
    experiment: str
    group_key: str
    prepared_dir: Path
    fixed_positions: Path
    prepared: List[str] = field(default_factory=list)
    already: List[str] = field(default_factory=list)
    failed: Dict[str, str] = field(default_factory=dict)

    @property
    def nothing_to_do(self) -> bool:
        return not self.prepared and not self.failed and bool(self.already)

    @property
    def ok(self) -> bool:
        return not self.failed and (bool(self.prepared) or bool(self.already))


def strip_ligand(structure: gemmi.Structure) -> Tuple[List[str], List[str]]:
    """Drop every non-protein chain. Returns (protein chains, dropped chains)."""
    protein, ligand = classify_chains(structure)
    if ligand:
        drop_chains(structure, ligand)
    if len(protein) != 2:
        raise PrepareError(
            f"expected exactly two protein chains for a tied design, found {protein}"
        )
    return protein, ligand


def prepare_group(stage: Path, experiment: str, group_key: str) -> PrepareResult:
    """Unpack one group's backbones and work out what MPNN must not redesign."""
    archive_path = jp.input_archive_path(stage, experiment, group_key)
    if not archive_path.is_file():
        raise PrepareError(f"no input archive: {archive_path}")

    try:
        _, config = design_config_for_group(stage, experiment, group_key)
    except DesignError as exc:
        raise PrepareError(str(exc))

    records = read_records(jp.fixed_residues_path(jp.stage01_root(stage), experiment))

    prepared_dir = jp.prepared_dir(stage, experiment, group_key)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    work_dir = jp.mpnn_work_dir(stage, experiment, group_key)
    work_dir.mkdir(parents=True, exist_ok=True)
    scratch = work_dir / "_scratch"

    # Proteins already designed are skipped, so adding structures to a group
    # costs MPNN only the new ones rather than a re-run over the whole group.
    already: set = set()
    designed = jp.mpnn_outputs_path(stage, experiment, group_key)
    if designed.is_file():
        with tarfile.open(designed, "r:gz") as archive:
            already = {name.split("/")[0] for name in archive.getnames() if "/" in name}

    result = PrepareResult(
        experiment, group_key, prepared_dir, jp.fixed_positions_path(stage, experiment, group_key)
    )
    fixed: Dict[str, Dict[str, List[int]]] = {}

    with tarfile.open(archive_path, "r:gz") as archive:
        groups = group_members_by_protein(archive.getnames())
        for protein_id, members in sorted(groups.items()):
            if protein_id in already:
                result.already.append(protein_id)
                continue
            if "structure" not in members or "json" not in members:
                result.failed[protein_id] = "incomplete_file_pair"
                continue
            structure_path = extract_member(archive, members["structure"], scratch)
            json_path = extract_member(archive, members["json"], scratch)
            try:
                structure = gemmi.read_structure(str(structure_path))
                structure.setup_entities()
                protein_chains, dropped = strip_ligand(structure)

                mapped = records.get(protein_id)
                if not mapped:
                    payload = json.loads(json_path.read_text(encoding="utf-8"))
                    diffused = payload.get("diffused_index_map") or {}
                    mapped, _ = fixed_residue_map(config, diffused)

                fixed[protein_id] = generated_positions(mapped, protein_chains)
                structure.write_pdb(str(prepared_dir / f"{protein_id}.pdb"))
                result.prepared.append(protein_id)
            except (PrepareError, DesignError, OSError, RuntimeError, ValueError,
                    json.JSONDecodeError) as exc:
                result.failed[protein_id] = str(exc).splitlines()[0]
            finally:
                structure_path.unlink(missing_ok=True)
                json_path.unlink(missing_ok=True)

    if not fixed and result.already:
        _log(f"[prepare] {experiment}/{group_key}: all {len(result.already)} protein(s) "
             f"already designed, nothing to prepare")
        return result
    if not fixed:
        raise PrepareError(
            f"{experiment}/{group_key}: nothing could be prepared "
            f"({len(result.failed)} structure(s) failed)"
        )

    # One object covering the whole group, which is the shape
    # protein_mpnn_run.py's --fixed_positions_jsonl expects.
    result.fixed_positions.write_text(json.dumps(fixed) + "\n", encoding="utf-8")
    return result


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("experiment")
    parser.add_argument("group_key")
    parser.add_argument("--stage", type=Path, default=STAGE)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = prepare_group(args.stage.resolve(), args.experiment, args.group_key)
    except PrepareError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    _log(f"[prepare] {result.experiment}/{result.group_key}: "
         f"{len(result.prepared)} backbone(s) -> {result.prepared_dir}")
    _log(f"[prepare] fixed positions -> {result.fixed_positions}")
    for protein_id, reason in sorted(result.failed.items()):
        _log(f"[prepare]   {protein_id}: NOT prepared ({reason})")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

