#!/usr/bin/env python3
"""
Stage 02 geometry filter: aligns each generated structure onto the seed it was
designed against, brings the seed's polysaccharide with it, and sorts on
clashes and contacts.

    inputs/<experiment>/<group>.tar.gz                     <- stage 01's transfer
    sorted_clean/<experiment>/passed/<group>.tar.gz        <- this writes
    sorted_clean/<experiment>/rejected/<group>.tar.gz

Why alignment is needed at all: RFD3 outputs the designed protein only. The
fibril lives in the stage-01 seed, so the design is superposed back onto the
seed's motif and the seed's non-protein chains are carried across. What gets
written out is the aligned protein plus the polysaccharide -- which is what
stage 03 consumes.

Rule (unchanged from 15082026_01_clashes_and_contacts_filtering.py)
    protein-protein clashes  CA-CA <= 1.8 A   must be 0
    protein-ligand clashes   all-atom <= 2.2 A must be 0
    protein-protein contacts CA-CA in [4, 8]  must be >= 7

Nothing about chains or seeds is hardcoded. The seed is found through the
stage-01 json the group was generated from; protein and ligand chains are
told apart by residue type; and the seed chain each generated chain aligns
onto is read out of the json's own diffused_index_map, which already carries
the seed chain letters in its keys.

Statuses
    PASSED / REJECTED  the structure was evaluated and met or missed the cutoffs
    SKIPPED            the seed has no ligand chain, so there is nothing to
                       check the design against; the structure is left in the
                       input archive untouched
    ERROR              the structure could not be evaluated (no json, no
                       align keys, a seed that cannot be read). Recorded with a
                       reason so one bad structure does not abort a 16k run.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import gemmi
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import job_paths as jp  # noqa: E402
from designs import (  # noqa: E402
    DesignError,
    design_config_for_group,
    fixed_residue_map,
    read_records,
    split_map_key,
)
from structures import classify_chains  # noqa: E402
from archives import (  # noqa: E402
    ArchiveError,
    extract_member,
    group_members_by_protein,
    load_table,
    regroup_and_archive,
    save_table,
)

import tarfile  # noqa: E402  (after the path setup, like the rest)

STAGE = Path(__file__).resolve().parents[2]

# Cutoffs, kept exactly as the production run used them so its 16k rows stay
# directly comparable with anything generated now.
CLASH_MAX = 0
LIGAND_CLASH_MAX = 0
CONTACT_MIN = 7

PROTEIN_CLASH_DISTANCE_A = 1.8
LIGAND_CLASH_DISTANCE_A = 2.2
CONTACT_MIN_DISTANCE_A = 4.0
CONTACT_MAX_DISTANCE_A = 8.0

CHECKPOINT_SIZE = 100

# Protein-ligand distances are all-atom, so the pair count can be large. Rows
# of the distance matrix are computed in blocks to bound peak memory.
DISTANCE_BLOCK = 512

RESULT_FIELDS = [
    "protein_id",
    "experiment_name",
    "time_stamp",
    "seed",
    "ligand_chains",
    "protein_ligand_clash_cutoff",
    "protein_protein_clash_cutoff",
    "protein_protein_contacts_cutoff",
    "rmsd_after_kabsch",
    "protein_ligand_clashes",
    "protein_protein_clashes",
    "clashing_residues",
    "protein_protein_contacts",
    "status",
    "rejection_reason",
    "skip_reason",
]


class GeometryError(Exception):
    """One structure could not be evaluated -- recorded, never fatal."""


class SeedError(Exception):
    """A group's seed could not be resolved. Affects the whole group."""


def _log(*parts: object) -> None:
    print(*parts, flush=True)


@dataclass
class Seed:
    """A group's reference structure, and what its chains are."""
    path: Path
    structure: gemmi.Structure
    protein_chains: List[str]
    ligand_chains: List[str]

    @property
    def has_ligand(self) -> bool:
        return bool(self.ligand_chains)


@dataclass
class FilterReport:
    scanned: int = 0
    skipped_already: int = 0
    counts: Dict[str, int] = field(default_factory=dict)
    tables: List[Path] = field(default_factory=list)
    archives: List[object] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """SKIPPED is an expected outcome and does not fail a run; ERROR means
        a structure fell out of the pipeline without a verdict."""
        return self.counts.get("ERROR", 0) == 0

    def summary(self) -> str:
        if not self.scanned:
            return f"geometry: nothing new ({self.skipped_already} already filtered)"
        parts = ", ".join(f"{key}={value}" for key, value in sorted(self.counts.items()))
        return (f"geometry: {self.scanned} evaluated ({parts}), "
                f"{self.skipped_already} already filtered, "
                f"{len(self.archives)} archive(s) updated")


# ---------------------------------------------------------------------------
# Layer 1: the seed
# ---------------------------------------------------------------------------


def seed_for_group(stage: Path, experiment: str, group_key: str) -> Seed:
    """The seed a group was generated from.

    group_key is the stage-01 json's filename stem, which job_name() encoded
    into every structure's name and the transfer preserved in the archive
    name. So the generating config -- and through it the seed -- is
    recoverable from the archive path alone, with nothing recorded anywhere
    along the way.
    """
    json_path = jp.stage01_json_path(stage, experiment, group_key)
    if not json_path.is_file():
        raise SeedError(
            f"no stage-01 json for group {group_key!r}: {json_path}\n"
            f"  the archive name must match the json that generated it"
        )
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        entries = jp.design_entries(data)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SeedError(f"could not read {json_path}: {exc}")

    raw_inputs = {str(config.get("input", "")) for config in entries.values()}
    raw_inputs.discard("")
    if len(raw_inputs) != 1:
        raise SeedError(
            f"{json_path} names {len(raw_inputs)} different seeds ({sorted(raw_inputs)}); "
            f"a group must come from exactly one"
        )
    raw_input = raw_inputs.pop()

    # Relative paths in a stage-01 json resolve against the stage-01 root, not
    # this one, so that is the root handed to the resolver.
    seed_path = jp.resolve_seed_path(jp.stage01_root(stage), raw_input, json_path)
    if seed_path is None:
        raise SeedError(
            f"seed not found for group {group_key!r}: {raw_input!r}\n"
            f"  (an absolute path written on another machine will not resolve here "
            f"-- make it relative to the stage-01 root)"
        )

    try:
        structure = gemmi.read_structure(str(seed_path))
        structure.setup_entities()
    except (OSError, RuntimeError, ValueError) as exc:
        raise SeedError(f"could not read seed {seed_path}: {exc}")

    protein, ligand = classify_chains(structure)
    if len(protein) != 2:
        raise SeedError(
            f"{seed_path.name} has {len(protein)} protein chain(s) ({protein}); "
            f"the alignment needs exactly two"
        )
    return Seed(seed_path, structure, protein, ligand)


# ---------------------------------------------------------------------------
# Layer 2: alignment
# ---------------------------------------------------------------------------


def fixed_map_for(
    records: Dict[str, Dict[str, str]], protein_id: str, config: dict, payload: dict
) -> Tuple[Dict[str, str], List[str], bool]:
    """{seed key: generated key} for the residues stage 01 fixed the side chains of.

    Prefers the record stage 01 wrote as the structure was generated -- that is
    a fact about what actually happened. Structures generated before the record
    existed are not in it, so their set is derived from the design config and
    the structure's own diffused_index_map instead, and the caller is told which
    route was taken.

    Returns (mapped, unmapped, derived).
    """
    recorded = records.get(protein_id)
    if recorded:
        return dict(recorded), [], False

    diffused_index_map = payload.get("diffused_index_map")
    if not isinstance(diffused_index_map, dict) or not diffused_index_map:
        raise GeometryError("missing_diffused_index_map")
    mapped, unmapped = fixed_residue_map(config, diffused_index_map)
    return mapped, unmapped, True


def index_by_chain_resi(structure: gemmi.Structure) -> Dict[str, Dict[int, Dict[str, np.ndarray]]]:
    """{chain: {residue number: {atom name: xyz}}} for one model."""
    index: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}
    for chain in structure[0]:
        residues: Dict[int, Dict[str, np.ndarray]] = {}
        for residue in chain:
            atoms = {
                atom.name: np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)
                for atom in residue
            }
            residues[residue.seqid.num] = atoms
        index[chain.name] = residues
    return index


def derive_seed_chains(
    mapped: Dict[str, str], seed_protein_chains: Sequence[str]
) -> Tuple[str, str]:
    """(the seed chain the motif sits in, the other one).

    The keys of the fixed-residue map are seed residues, chain letter included,
    so the chain is already in the data and never has to be configured.
    """
    named = {split_map_key(key)[0] for key in mapped}
    if len(named) != 1:
        raise GeometryError(
            f"the fixed residues name {sorted(named)} seed chains; expected exactly one"
        )
    direct = named.pop()
    if direct not in seed_protein_chains:
        raise GeometryError(
            f"the fixed residues name seed chain {direct!r}, but the seed's protein "
            f"chains are {list(seed_protein_chains)}"
        )
    mirrored = next(name for name in seed_protein_chains if name != direct)
    return direct, mirrored


def assign_generated_chains(
    generated_chains: Sequence[str], mapped: Dict[str, str], seed_chains: Tuple[str, str]
) -> Dict[str, str]:
    """{generated chain: seed chain it aligns onto}.

    One generated chain is named by the map's values; the other is its symmetry
    copy and takes the seed's other chain.
    """
    if len(generated_chains) != 2:
        raise GeometryError(
            f"expected exactly two generated protein chains, found {list(generated_chains)}"
        )
    named = {split_map_key(value)[0] for value in mapped.values()}
    if len(named) != 1:
        raise GeometryError(
            f"the fixed residues map to {sorted(named)} generated chains; expected one"
        )
    direct = named.pop()
    if direct not in generated_chains:
        raise GeometryError(
            f"the fixed residues map to chain {direct!r}, but the generated chains "
            f"are {list(generated_chains)}"
        )
    mirrored = next(name for name in generated_chains if name != direct)
    return {direct: seed_chains[0], mirrored: seed_chains[1]}


def build_alignment_pairs(
    seed_index: dict, generated_index: dict, mapped: Dict[str, str],
    seed_chain: str, generated_chain: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Matched (generated, seed) coordinates for one chain pair.

    Fits on EVERY atom the two residues share. "ALL" in select_fixed_atoms means
    RFD3 was told to preserve the whole residue, side chain included, so all of
    it is legitimate to fit on -- and unlike a fixed tyrosine-atom list, this
    works whatever the residue happens to be.
    """
    generated_coords: List[np.ndarray] = []
    seed_coords: List[np.ndarray] = []
    for seed_key, generated_key in sorted(mapped.items()):
        _, seed_resi = split_map_key(seed_key)
        _, generated_resi = split_map_key(generated_key)
        seed_residue = seed_index.get(seed_chain, {}).get(seed_resi)
        generated_residue = generated_index.get(generated_chain, {}).get(generated_resi)
        if seed_residue is None or generated_residue is None:
            continue
        for atom_name in sorted(set(seed_residue) & set(generated_residue)):
            seed_coords.append(seed_residue[atom_name])
            generated_coords.append(generated_residue[atom_name])
    return np.array(generated_coords), np.array(seed_coords)


def kabsch(moving: np.ndarray, fixed: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Rigid transform taking `moving` onto `fixed`, plus the fit rmsd.

    Centred SVD with the determinant forced positive, so a reflection is never
    returned as a fit -- a mirrored protein is not the same protein.
    """
    if moving.shape[0] < 3:
        raise GeometryError(f"need at least 3 fit atoms, got {moving.shape[0]}")
    moving_centre = moving.mean(axis=0)
    fixed_centre = fixed.mean(axis=0)
    correlation = (moving - moving_centre).T @ (fixed - fixed_centre)
    u, _, vt = np.linalg.svd(correlation)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    translation = fixed_centre - rotation @ moving_centre
    deviations = (moving @ rotation.T + translation) - fixed
    rmsd = float(np.sqrt((deviations ** 2).sum(axis=1).mean()))
    return rotation, translation, rmsd


# ---------------------------------------------------------------------------
# Layer 3: geometry
# ---------------------------------------------------------------------------


def _blocked_min_distances(a: np.ndarray, b: np.ndarray):
    """Yield (row offset, distance block) for the a-by-b distance matrix."""
    for start in range(0, a.shape[0], DISTANCE_BLOCK):
        block = a[start:start + DISTANCE_BLOCK]
        yield start, np.linalg.norm(block[:, None, :] - b[None, :, :], axis=2)


def ca_clashes_and_contacts(
    coords_a: np.ndarray, resi_a: Sequence[int], coords_b: np.ndarray, resi_b: Sequence[int]
) -> Tuple[List[Tuple[int, int]], int, int]:
    """(clashing residue pairs, clash count, contact count) between two chains.

    CA atoms only, matching the production definition: a "clash" here means the
    backbones have interpenetrated, not that side chains overlap.
    """
    clashes: List[Tuple[int, int]] = []
    clash_count = 0
    contact_count = 0
    for start, block in _blocked_min_distances(coords_a, coords_b):
        close = np.argwhere(block <= PROTEIN_CLASH_DISTANCE_A)
        for i, j in close:
            clashes.append((int(resi_a[start + i]), int(resi_b[j])))
        clash_count += int(close.shape[0])
        contact_count += int(np.count_nonzero(
            (block >= CONTACT_MIN_DISTANCE_A) & (block <= CONTACT_MAX_DISTANCE_A)
        ))
    return clashes, clash_count, contact_count


def ligand_clashes(protein_coords: np.ndarray, ligand_coords: np.ndarray) -> int:
    """All-atom protein-ligand pairs within the clash distance."""
    if protein_coords.size == 0 or ligand_coords.size == 0:
        return 0
    total = 0
    for _, block in _blocked_min_distances(protein_coords, ligand_coords):
        total += int(np.count_nonzero(block <= LIGAND_CLASH_DISTANCE_A))
    return total


def decide_status(
    clash_count: int, ligand_clash_count: int, contact_count: int
) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    if clash_count > CLASH_MAX:
        reasons.append(f"more_than_{CLASH_MAX}_protein_protein_clashes")
    if ligand_clash_count > LIGAND_CLASH_MAX:
        reasons.append("protein_ligand_clashes")
    if contact_count < CONTACT_MIN:
        reasons.append(f"less_than_{CONTACT_MIN}_contacts")
    return ("REJECTED" if reasons else "PASSED"), reasons


# ---------------------------------------------------------------------------
# Layer 4: rows
# ---------------------------------------------------------------------------


def build_row(protein_id: str, experiment: str, seed: Optional[Seed], status: str,
              rmsd: Optional[float] = None, ligand_clash_count: Optional[int] = None,
              clash_count: Optional[int] = None,
              clashing_residues: Optional[Sequence[Tuple[int, int]]] = None,
              contact_count: Optional[int] = None,
              reasons: Sequence[str] = (), skip_reason: str = "") -> Dict[str, str]:
    def number(value, spec: str) -> str:
        return format(value, spec) if value is not None else ""

    return {
        "protein_id": protein_id,
        "experiment_name": experiment,
        "time_stamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed.path.name if seed is not None else "",
        "ligand_chains": ";".join(seed.ligand_chains) if seed is not None else "",
        "protein_ligand_clash_cutoff": f"{LIGAND_CLASH_MAX:d}",
        "protein_protein_clash_cutoff": f"{CLASH_MAX:d}",
        "protein_protein_contacts_cutoff": f"{CONTACT_MIN:d}",
        "rmsd_after_kabsch": number(rmsd, ".6g"),
        "protein_ligand_clashes": number(ligand_clash_count, "d"),
        "protein_protein_clashes": number(clash_count, "d"),
        "clashing_residues": str(list(clashing_residues)) if clashing_residues else "",
        "protein_protein_contacts": number(contact_count, "d"),
        "status": status,
        "rejection_reason": ";".join(reasons),
        "skip_reason": skip_reason,
    }


def table_sort_key(row: Dict[str, str]) -> tuple:
    """Worst first: ERROR, then SKIPPED, then fewest contacts."""
    rank = {"ERROR": 0, "SKIPPED": 1}.get(row.get("status", ""), 2)
    try:
        contacts = float(row.get("protein_protein_contacts", ""))
    except ValueError:
        contacts = float("-inf")
    return (rank, contacts, row.get("protein_id", ""))


# ---------------------------------------------------------------------------
# Layer 5: one structure
# ---------------------------------------------------------------------------


def _coords_and_resi(structure: gemmi.Structure, chain_name: str,
                     atom_name: Optional[str] = None) -> Tuple[np.ndarray, List[int]]:
    """Coordinates of one chain: a named atom per residue, or every atom."""
    coords: List[List[float]] = []
    resi: List[int] = []
    for chain in structure[0]:
        if chain.name != chain_name:
            continue
        for residue in chain:
            for atom in residue:
                if atom_name is not None and atom.name != atom_name:
                    continue
                coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                resi.append(residue.seqid.num)
    return (np.array(coords, dtype=float) if coords else np.empty((0, 3))), resi


def apply_transforms(structure: gemmi.Structure,
                     transforms: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
    """Move each chain onto its seed chain, in place."""
    for chain in structure[0]:
        if chain.name not in transforms:
            continue
        rotation, translation = transforms[chain.name]
        for residue in chain:
            for atom in residue:
                moved = rotation @ np.array([atom.pos.x, atom.pos.y, atom.pos.z]) + translation
                atom.pos = gemmi.Position(*moved)


def attach_ligand(target: gemmi.Structure, seed: Seed) -> None:
    """Copy the seed's non-protein chains into the aligned structure.

    This is how the fibril reaches stage 03: RFD3 never produced it, so it
    travels with the design from the seed the design was fitted to.
    """
    present = {chain.name for chain in target[0]}
    for chain in seed.structure[0]:
        if chain.name not in seed.ligand_chains:
            continue
        if chain.name in present:
            raise GeometryError(
                f"ligand chain {chain.name!r} collides with a generated chain name"
            )
        copied = gemmi.Chain(chain.name)
        for residue in chain:
            copied.add_residue(residue)
        target[0].add_chain(copied)


def evaluate_structure(structure: gemmi.Structure, mapped: Dict[str, str], seed: Seed
                       ) -> Tuple[float, int, List[Tuple[int, int]], int, int]:
    """Align, attach the ligand, and count. Mutates `structure` into the
    aligned, ligand-bearing form that gets written out.

    Returns (rmsd, clash_count, clashing_residues, contact_count, ligand_clashes).
    """
    generated_protein, _ = classify_chains(structure)
    seed_chains = derive_seed_chains(mapped, seed.protein_chains)
    assignment = assign_generated_chains(generated_protein, mapped, seed_chains)

    seed_index = index_by_chain_resi(seed.structure)
    generated_index = index_by_chain_resi(structure)

    transforms: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    rmsds: List[float] = []
    for generated_chain, seed_chain in assignment.items():
        moving, fixed = build_alignment_pairs(
            seed_index, generated_index, mapped, seed_chain, generated_chain,
        )
        rotation, translation, chain_rmsd = kabsch(moving, fixed)
        transforms[generated_chain] = (rotation, translation)
        rmsds.append(chain_rmsd)

    apply_transforms(structure, transforms)
    attach_ligand(structure, seed)

    first, second = generated_protein
    coords_a, resi_a = _coords_and_resi(structure, first, "CA")
    coords_b, resi_b = _coords_and_resi(structure, second, "CA")
    if coords_a.size == 0 or coords_b.size == 0:
        raise GeometryError(f"no CA atoms in chain {first if coords_a.size == 0 else second}")
    clashing, clash_count, contact_count = ca_clashes_and_contacts(
        coords_a, resi_a, coords_b, resi_b
    )

    protein_atoms = np.vstack([
        _coords_and_resi(structure, name)[0] for name in generated_protein
    ])
    ligand_atoms = [_coords_and_resi(structure, name)[0] for name in seed.ligand_chains]
    ligand_atoms = np.vstack(ligand_atoms) if ligand_atoms else np.empty((0, 3))
    ligand_clash_count = ligand_clashes(protein_atoms, ligand_atoms)

    # Worst of the per-chain fits, which is what the table records.
    return (max(rmsds), clash_count, clashing, contact_count, ligand_clash_count)


# ---------------------------------------------------------------------------
# Layer 6: orchestrator
# ---------------------------------------------------------------------------


def archives_to_scan(stage: Path, experiment: Optional[str] = None) -> Dict[str, List[Path]]:
    """{experiment: [inputs/<experiment>/<group>.tar.gz, ...]}."""
    names = [experiment] if experiment else jp.inputs_experiments(stage)
    found: Dict[str, List[Path]] = {}
    for name in names:
        directory = jp.inputs_root(stage) / name
        if not directory.is_dir():
            raise GeometryError(f"no such experiment under inputs/: {directory}")
        tars = sorted(directory.glob("*.tar.gz"))
        if tars:
            found[name] = tars
    return found


def _scan_archive(stage: Path, experiment: str, tar_path: Path,
                  by_id: Dict[str, Dict[str, str]], table_path: Path,
                  report: FilterReport,
                  records: Dict[str, Dict[str, str]]) -> None:
    group_key = tar_path.name[: -len(".tar.gz")]
    scratch_dir = jp.sorted_scratch_dir(stage, experiment)

    try:
        seed = seed_for_group(stage, experiment, group_key)
        _, config = design_config_for_group(stage, experiment, group_key)
    except (SeedError, DesignError) as exc:
        seed = None
        seed_problem = str(exc)
    else:
        seed_problem = ""
        _log(f"[geometry] {experiment}/{group_key}: seed {seed.path.name} "
             f"(protein {seed.protein_chains}, ligand {seed.ligand_chains or 'none'})")

    with tarfile.open(tar_path, "r:gz") as archive:
        groups = group_members_by_protein(archive.getnames())

        for protein_id, members in sorted(groups.items()):
            if protein_id in by_id:
                report.skipped_already += 1
                continue

            def record(status: str, **kwargs) -> None:
                by_id[protein_id] = build_row(protein_id, experiment, seed, status, **kwargs)
                report.scanned += 1
                report.counts[status] = report.counts.get(status, 0) + 1

            if seed is None:
                record("ERROR", skip_reason=seed_problem.splitlines()[0])
                _log(f"  {protein_id}: ERROR ({seed_problem.splitlines()[0]})")
                continue

            # Checked before anything is extracted or aligned: with no ligand
            # there is nothing to check the design against, and the alignment
            # would fail for its own unrelated reasons.
            if not seed.has_ligand:
                message = (f"alignment and geometry check is not possible for "
                           f"{protein_id}, since the stage-01 input "
                           f"{seed.path.name} has no ligand chains")
                record("SKIPPED", skip_reason="seed_has_no_ligand")
                _log(f"  [SKIPPED] {message}")
                continue

            if "json" not in members or "structure" not in members:
                record("ERROR", skip_reason="incomplete_file_pair")
                _log(f"  {protein_id}: ERROR (has only {sorted(members)})")
                continue

            structure_path = extract_member(archive, members["structure"], scratch_dir)
            json_path = extract_member(archive, members["json"], scratch_dir)
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                mapped, unmapped, derived = fixed_map_for(
                    records, protein_id, config, payload
                )
                if unmapped:
                    _log(f"  [note] {protein_id}: {len(unmapped)} fixed residue(s) "
                         f"{unmapped} are not in diffused_index_map; fitting on the rest")
                structure = gemmi.read_structure(str(structure_path))
                structure.setup_entities()
                rmsd, clashes, clashing, contacts, ligand_hits = evaluate_structure(
                    structure, mapped, seed
                )
            except (GeometryError, DesignError, OSError, RuntimeError, ValueError,
                    json.JSONDecodeError) as exc:
                record("ERROR", skip_reason=str(exc).splitlines()[0])
                _log(f"  {protein_id}: ERROR ({str(exc).splitlines()[0]})")
                structure_path.unlink(missing_ok=True)
                json_path.unlink(missing_ok=True)
                continue

            status, reasons = decide_status(clashes, ligand_hits, contacts)
            record(status, rmsd=rmsd, ligand_clash_count=ligand_hits,
                   clash_count=clashes, clashing_residues=clashing,
                   contact_count=contacts, reasons=reasons)
            _log(f"  {protein_id}: rmsd={rmsd:.3f} clashes={clashes} "
                 f"ligand_clashes={ligand_hits} contacts={contacts} -> {status}"
                 + (f" ({';'.join(reasons)})" if reasons else ""))

            outcome = "passed" if status == "PASSED" else "rejected"
            target_dir = jp.sorted_raw_group_dir(stage, experiment, outcome, group_key)
            target_dir.mkdir(parents=True, exist_ok=True)
            # PDB, not cif: stage 03 reads the chain letter out of column 22.
            structure.write_pdb(str(target_dir / f"{protein_id}.pdb"))
            shutil.move(str(json_path), str(target_dir / f"{protein_id}.json"))
            structure_path.unlink(missing_ok=True)

            if report.scanned % CHECKPOINT_SIZE == 0:
                save_table(by_id, table_path, RESULT_FIELDS, table_sort_key)
                _log(f"  [checkpoint] {report.scanned} evaluated, table saved")


def run_geometry_filter(stage: Path, experiment: Optional[str] = None) -> FilterReport:
    report = FilterReport()
    by_experiment = archives_to_scan(stage, experiment)
    if not by_experiment:
        _log(f"[geometry] nothing to filter under {jp.inputs_root(stage)}")
        return report

    _log(f"[geometry] criteria: protein-protein clashes (CA-CA <= "
         f"{PROTEIN_CLASH_DISTANCE_A} A) <= {CLASH_MAX}, protein-ligand clashes "
         f"(all-atom <= {LIGAND_CLASH_DISTANCE_A} A) <= {LIGAND_CLASH_MAX}, "
         f"contacts (CA-CA {CONTACT_MIN_DISTANCE_A}-{CONTACT_MAX_DISTANCE_A} A) "
         f">= {CONTACT_MIN}")

    for experiment_name, tar_paths in by_experiment.items():
        table_path = jp.results_table_path(stage, experiment_name)
        by_id = load_table(table_path)
        before = len(by_id)
        # What stage 01 recorded as it generated. Absent for anything made
        # before the record existed, which falls back to deriving the set.
        records = read_records(jp.fixed_residues_path(jp.stage01_root(stage), experiment_name))
        _log(f"[geometry] {experiment_name}: {len(tar_paths)} archive(s), "
             f"{before} structure(s) already filtered, "
             f"{len(records)} with a recorded fixed-residue set")

        for tar_path in tar_paths:
            _scan_archive(stage, experiment_name, tar_path, by_id, table_path,
                          report, records)

        if len(by_id) == before:
            continue
        save_table(by_id, table_path, RESULT_FIELDS, table_sort_key)
        report.tables.append(table_path)
        for result in regroup_and_archive(stage, experiment_name):
            report.archives.append(result)
            _log(f"[geometry] {result.outcome}/{result.group_key}: +{result.added} new, "
                 f"{result.total} member(s) total -> {result.archive}")

    return report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=Path, default=STAGE,
                        help=f"stage root directory (default: {STAGE})")
    parser.add_argument("--experiment", default=None,
                        help="only filter this one experiment (default: all of them)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = run_geometry_filter(args.stage.resolve(), args.experiment)
    except (OSError, ArchiveError, GeometryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    _log(f"[done] {report.summary()}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

