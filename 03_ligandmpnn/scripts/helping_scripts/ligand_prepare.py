#!/usr/bin/env python3
"""
Stage 03, step 1: prepare one group for LigandMPNN.

    inputs/<experiment>/<group>.tar.gz                        <- stage 02's transfer
    inputs_prepared/<experiment>/<group>/<protein_id>.pdb     <- protein + ONE ligand chain
    job_runs/ligandmpnn/<experiment>/<group>/redesign.jsonl   <- what may change, and what is tied

This stage used to run vanilla ProteinMPNN, which has no notion of a ligand: the
fibril was stripped before designing, so the sequence was chosen without ever
seeing the thing it was meant to bind. LigandMPNN takes the ligand as context,
so the fibril stays in the pdb handed to it.

That change removes a whole stage. The old pipeline designed blind, folded a
monomer, then superposed the prediction back onto a reference to recover the
ligand context and redesign the interface -- and that superposition was where
the fit error and the clashes came from. Designing on the stage 02 structure
directly keeps the ligand in its original diffused geometry, so there is no fit
and nothing to clash-filter.

A seed can sandwich more than one fibril chain. Only the one the protein
actually sits on is kept, chosen by contact count in common/structures.py, so
this stage and the fold in stage 04 are talking about the same binding site.
Designing against several chains and folding against one would quietly be two
different experiments.

Scaffolded residues are held: "ALL" in select_fixed_atoms means the side chain
was fixed and stays fixed, "BKBN" means only the backbone was, so those side
chains are redesigned like any other. fixed_residues() in common/designs.py
already draws that line, so what arrives here is the ALL-tagged set alone.

Everything else is redesigned, tied position-for-position between the two copies
so they come out identical. The tie is checked rather than assumed -- here, that
both chains carry the same residue numbering, and in the runner, that the two
output chains really are identical.
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
from structures import (  # noqa: E402
    CONTACT_CUTOFF,
    StructureError,
    keep_one_ligand_chain,
)

STAGE = Path(__file__).resolve().parents[2]

SPEC_NAME = "redesign.jsonl"


class PrepareError(Exception):
    """A group could not be prepared."""


def _log(*parts: object) -> None:
    print(*parts, flush=True)


def why_failed(failed: Dict[str, str], limit: int = 5) -> str:
    """The reasons, for an exception message that would otherwise be a count.

    "1 structure(s) failed" sends somebody reading source to find out what
    happened; the reason was already in hand when that message was built.
    Capped, because a group of 500 that all failed for the same reason should
    not print 500 times to say so.
    """
    items = sorted(failed.items())
    shown = "; ".join(f"{name}: {reason}" for name, reason in items[:limit])
    if len(items) > limit:
        shown += f" (+{len(items) - limit} more)"
    return shown or "no reason recorded"


@dataclass
class PrepareResult:
    experiment: str
    group_key: str
    prepared_dir: Path
    spec_path: Path
    prepared: List[str] = field(default_factory=list)
    already: List[str] = field(default_factory=list)
    no_contact: List[str] = field(default_factory=list)
    failed: Dict[str, str] = field(default_factory=dict)

    @property
    def nothing_to_do(self) -> bool:
        return not self.prepared and not self.failed and bool(self.already)

    @property
    def ok(self) -> bool:
        return not self.failed and (bool(self.prepared) or bool(self.already))


def tied_numbering(structure: gemmi.Structure,
                   protein_chains: Sequence[str]) -> List[int]:
    """The residue numbers shared by both copies, or a refusal.

    Tying position n of chain A to position n of chain B is only meaningful if
    both chains number their residues the same way. RFD3's symmetric output
    does, but a structure that did not would produce ties between unrelated
    residues, and LigandMPNN would accept them without complaint -- so this is
    checked rather than trusted.

    The numbers present are enumerated rather than taken as a range: a motif
    scaffold is routinely non-contiguous.
    """
    if len(protein_chains) != 2:
        raise PrepareError(
            f"expected exactly two protein chains for a tied design, "
            f"found {list(protein_chains)}"
        )
    first, second = (
        [residue.seqid.num for residue in structure[0][name]]
        for name in protein_chains
    )
    if sorted(first) != sorted(second):
        only_first = sorted(set(first) - set(second))[:5]
        only_second = sorted(set(second) - set(first))[:5]
        raise PrepareError(
            f"chains {protein_chains[0]} and {protein_chains[1]} do not share a "
            f"residue numbering, so they cannot be tied position for position "
            f"({protein_chains[0]} only: {only_first}, "
            f"{protein_chains[1]} only: {only_second})"
        )
    return sorted(set(first))


def redesign_spec(numbers: Sequence[int], fixed_positions: Dict[str, List[int]],
                  protein_chains: Sequence[str]) -> Tuple[List[str], List[str]]:
    """(residues LigandMPNN may change, symmetry groups tying the two copies).

    Every position except the scaffolded ones, in both chains. The excluded set
    is the union over chains rather than the intersection: a position held in
    one copy must be held in the other, or the tie would ask LigandMPNN to
    redesign a residue in one chain while freezing its partner.
    """
    excluded = {number for numbers_ in fixed_positions.values() for number in numbers_}
    free = [number for number in numbers if number not in excluded]
    residues = [f"{chain}{number}" for number in free for chain in protein_chains]
    groups = [",".join(f"{chain}{number}" for chain in protein_chains)
              for number in free]
    return residues, groups


def prepare_group(stage: Path, experiment: str, group_key: str) -> PrepareResult:
    """Unpack one group's complexes and work out what LigandMPNN may change."""
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
    work_dir = jp.ligand_work_dir(stage, experiment, group_key)
    work_dir.mkdir(parents=True, exist_ok=True)
    scratch = work_dir / "_scratch"
    spec_path = work_dir / SPEC_NAME

    # Proteins already designed are skipped, so adding structures to a group
    # costs LigandMPNN only the new ones rather than a re-run over the whole group.
    already: set = set()
    designed = jp.mpnn_outputs_path(stage, experiment, group_key)
    if designed.is_file():
        with tarfile.open(designed, "r:gz") as archive:
            already = {name.split("/")[0] for name in archive.getnames() if "/" in name}

    result = PrepareResult(experiment, group_key, prepared_dir, spec_path)
    fresh: List[dict] = []

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

                # The fibril stays; only the chain the protein sits on.
                protein_chains, ligand_chain, contacts, dropped = \
                    keep_one_ligand_chain(structure)
                numbers = tied_numbering(structure, protein_chains)

                mapped = records.get(protein_id)
                if not mapped:
                    payload = json.loads(json_path.read_text(encoding="utf-8"))
                    diffused = payload.get("diffused_index_map") or {}
                    mapped, _ = fixed_residue_map(config, diffused)

                fixed_positions = generated_positions(mapped, protein_chains)
                residues, symmetry = redesign_spec(
                    numbers, fixed_positions, protein_chains
                )
                if not residues:
                    raise PrepareError(
                        "every residue is scaffolded, so there is nothing to design"
                    )

                out_path = prepared_dir / f"{protein_id}.pdb"
                structure.write_pdb(str(out_path))
                fresh.append({
                    "id": protein_id,
                    "protein_id": protein_id,
                    "experiment_name": experiment,
                    "group": group_key,
                    "pdb": out_path.name,
                    "protein_chains": list(protein_chains),
                    "ligand_chain": ligand_chain,
                    "ligand_contacts": contacts,
                    "ligand_chains_dropped": dropped,
                    "redesigned_residues": residues,
                    "symmetry_residues": symmetry,
                    "held": sorted(
                        {number for numbers_ in fixed_positions.values()
                         for number in numbers_}
                    ),
                })
                result.prepared.append(protein_id)
                if contacts == 0:
                    # Designing "in the presence of" a fibril the protein never
                    # touches is ligand-aware in name only. Counted so it shows
                    # in the summary instead of scrolling past in the log.
                    result.no_contact.append(protein_id)
                _log(f"[prepare] {protein_id}: {len(symmetry)} position(s) redesignable, "
                     f"{len(fixed_positions.get(protein_chains[0], []))} scaffolded held, "
                     f"ligand {ligand_chain} ({contacts} contact(s))"
                     + (f", dropped {dropped}" if dropped else ""))
            except (PrepareError, DesignError, StructureError, OSError, RuntimeError,
                    ValueError, KeyError, json.JSONDecodeError) as exc:
                result.failed[protein_id] = str(exc).splitlines()[0]
                # Said here as well as collected, because report() -- the only
                # thing that printed these -- never runs when the group has no
                # successes: the raise below returns first. A group that fails
                # entirely is exactly when you need the reason.
                _log(f"[prepare] {protein_id}: FAILED ({exc})")
            finally:
                structure_path.unlink(missing_ok=True)
                json_path.unlink(missing_ok=True)

    if not fresh and result.already:
        _log(f"[prepare] {experiment}/{group_key}: all {len(result.already)} protein(s) "
             f"already designed, nothing to prepare")
        return result
    if not fresh:
        raise PrepareError(
            f"{experiment}/{group_key}: nothing could be prepared "
            f"({len(result.failed)} structure(s) failed) -- {why_failed(result.failed)}"
        )

    # Appended, not rewritten: a later run over new structures in the same group
    # must not erase the records an earlier one wrote.
    with spec_path.open("a", encoding="utf-8") as handle:
        for entry in fresh:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
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
         f"{len(result.prepared)} complex(es) -> {result.prepared_dir}")
    _log(f"[prepare] redesign spec -> {result.spec_path}")
    if result.no_contact:
        _log(f"[prepare] WARNING: {len(result.no_contact)} complex(es) have no protein "
             f"atom within {CONTACT_CUTOFF:g} A of any ligand chain -- the ligand was "
             f"chosen arbitrarily and the design is ligand-aware in name only: "
             f"{', '.join(result.no_contact[:5])}")
    for protein_id, reason in sorted(result.failed.items()):
        _log(f"[prepare]   {protein_id}: NOT prepared ({reason})")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
