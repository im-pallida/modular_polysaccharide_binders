#!/usr/bin/env python3
"""
Stage 05, step 1: rebuild the ligand complex around the validated fold.

    inputs/<experiment>/<group>.tar.gz            <- stage 04's passed AF3 output
    ../02_geometry_filtering/sorted_clean/<experiment>/passed/<group>.tar.gz
                                                  <- the reference, ligand included
    inputs_prepared/<experiment>/<group>/<sequence_id>.pdb
    job_runs/ligandmpnn/<experiment>/<group>/redesign.jsonl

AlphaFold predicted a monomer and knows nothing about the polysaccharide. The
stage-02 design has the polysaccharide but its backbone was never validated.
This puts the two together: the AF3 chain is superposed onto each reference
protein chain in turn using the scaffolded residues, and the reference's ligand
chains are kept where they are. The result is the fold AF3 confirmed, sitting
in the binding geometry the design was built for, twice -- so stage 06 has two
chains to link.

Superposing on the scaffolded residues rather than the whole chain is the point:
those residues are the binding site, they were held fixed from stage 01 onward,
and aligning on them puts the site where it belongs even if the rest of the fold
drifted. All atoms shared between the two copies of a residue are used, not just
CA, because five CA atoms barely constrain a rotation.

The redesign set is every residue with an atom within CUTOFF of any ligand atom,
minus the scaffolded ones. Everything else is held fixed: the fold is finished,
and only the surface facing the polysaccharide is still in question.
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
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import job_paths as jp  # noqa: E402
from archives import extract_member  # noqa: E402
from designs import (  # noqa: E402
    DesignError,
    fixed_residue_map,
    generated_positions,
    read_records,
)
from structures import classify_chains  # noqa: E402

STAGE = Path(__file__).resolve().parents[2]

CUTOFF = 8.0          # angstroms, any protein atom to any ligand atom
MIN_FIT_ATOMS = 6     # below this a superposition is not meaningfully constrained
COPY_CHAINS = ("A", "B")

# Clash limits, on CA atoms only. The sequence is about to be redesigned, so a
# side-chain overlap is exactly what LigandMPNN is there to fix; a CA overlap is
# the backbone itself interpenetrating, which no sequence can repair.
#
# CHAIN_CLASH: two CA atoms in different chains. Even in a tight interface the
# closest cross-chain CA-CA is 4.5-6 A, and within one chain consecutive CAs sit
# at 3.8 A -- so below 4.0 A across chains the backbones are inside each other.
# LIGAND_CLASH: a protein CA against any ligand heavy atom (a sugar has no CA).
# In a real binding site that distance bottoms out around 3.5-4.5 A.
CHAIN_CLASH = 4.0
LIGAND_CLASH = 3.0


class PrepareError(Exception):
    """One complex could not be built."""


def _log(*parts: object) -> None:
    print(*parts, flush=True)


@dataclass
class PrepareReport:
    built: int = 0
    already: int = 0
    clashing: int = 0
    shell_sizes: List[int] = field(default_factory=list)
    clash_rows: List[dict] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        if not self.built:
            clashed = f", {self.clashing} rejected for clashes" if self.clashing else ""
            return f"prepare: nothing new ({self.already} already built{clashed})"
        sizes = self.shell_sizes
        clashed = f", {self.clashing} rejected for clashes" if self.clashing else ""
        return (f"prepare: {self.built} complex(es) built, {self.already} already built"
                f"{clashed}, shell {min(sizes)}-{max(sizes)} residues "
                f"(median {int(np.median(sizes))})")


# ---------------------------------------------------------------------------
# Superposition on a chosen set of residues
# ---------------------------------------------------------------------------

def kabsch(moving: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(rotation, moving centroid, target centroid, rmsd). No reflections."""
    moving_center = moving.mean(axis=0)
    target_center = target.mean(axis=0)
    a = moving - moving_center
    b = target - target_center
    u, _, vt = np.linalg.svd(a.T @ b)
    determinant = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, determinant]) @ u.T
    deviation = a @ rotation.T - b
    rmsd = float(np.sqrt((deviation ** 2).sum(axis=1).mean()))
    return rotation, moving_center, target_center, rmsd


def residue_atoms(residue: gemmi.Residue) -> Dict[str, np.ndarray]:
    return {atom.name: np.array([atom.pos.x, atom.pos.y, atom.pos.z])
            for atom in residue}


def paired_atoms(moving_residues: Sequence[gemmi.Residue],
                 target_residues: Sequence[gemmi.Residue]) -> Tuple[np.ndarray, np.ndarray]:
    """Coordinates of every atom name the two residue lists share, in order.

    Atom names rather than atom order, because a residue written by one program
    need not list its atoms the way another does, and a superposition built on
    mismatched pairs is wrong without ever looking wrong.
    """
    moving_points: List[np.ndarray] = []
    target_points: List[np.ndarray] = []
    for moving_residue, target_residue in zip(moving_residues, target_residues):
        moving_map = residue_atoms(moving_residue)
        target_map = residue_atoms(target_residue)
        for name in sorted(set(moving_map) & set(target_map)):
            moving_points.append(moving_map[name])
            target_points.append(target_map[name])
    if not moving_points:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.array(moving_points), np.array(target_points)


def apply_transform(chain: gemmi.Chain, rotation: np.ndarray,
                    moving_center: np.ndarray, target_center: np.ndarray) -> None:
    for residue in chain:
        for atom in residue:
            point = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
            moved = (point - moving_center) @ rotation.T + target_center
            atom.pos = gemmi.Position(float(moved[0]), float(moved[1]), float(moved[2]))


# ---------------------------------------------------------------------------
# Building the complex
# ---------------------------------------------------------------------------

def polymer_residues(chain: gemmi.Chain) -> List[gemmi.Residue]:
    return [residue for residue in chain
            if (info := gemmi.find_tabulated_residue(residue.name)) is not None
            and info.is_amino_acid()]


def select_fixed(residues: Sequence[gemmi.Residue],
                 positions: Sequence[int]) -> List[gemmi.Residue]:
    wanted = set(positions)
    return [residue for residue in residues if residue.seqid.num in wanted]


def build_complex(model_structure: gemmi.Structure, reference: gemmi.Structure,
                  fixed_positions: Dict[str, List[int]],
                  reference_protein: Sequence[str],
                  reference_ligand: Sequence[str]) -> Tuple[gemmi.Structure, List[float]]:
    """Two superposed copies of the AF3 chain, plus the reference's ligand."""
    model_chains = [chain for chain in model_structure[0] if polymer_residues(chain)]
    if len(model_chains) != 1:
        raise PrepareError(
            f"expected one protein chain in the AF3 model, found {len(model_chains)}"
        )
    model_residues = polymer_residues(model_chains[0])

    out = gemmi.Structure()
    out.add_model(gemmi.Model("1"))
    rmsds: List[float] = []

    for copy_name, reference_name in zip(COPY_CHAINS, reference_protein):
        reference_chain = reference[0][reference_name]
        reference_residues = polymer_residues(reference_chain)
        if len(reference_residues) != len(model_residues):
            raise PrepareError(
                f"chain {reference_name} has {len(reference_residues)} residues but the "
                f"AF3 model has {len(model_residues)} -- they should be the same design, "
                f"so the numbering cannot be transferred safely"
            )

        # Residue i of the prediction is residue i of the reference: AF3 folded
        # exactly the sequence MPNN wrote for this backbone, so the two run
        # one-to-one. The length check above is what makes that safe to assume.
        numbering = [residue.seqid.num for residue in reference_residues]
        positions = fixed_positions.get(reference_name) or []
        index_of = {number: index for index, number in enumerate(numbering)}
        fixed_indices = [index_of[number] for number in positions if number in index_of]
        if not fixed_indices:
            raise PrepareError(
                f"none of the scaffolded residues {positions} are present in reference "
                f"chain {reference_name}"
            )

        working = gemmi.Structure()
        working.add_model(gemmi.Model("1"))
        working[0].add_chain(model_chains[0].clone())
        working_chain = working[0][0]
        working_residues = polymer_residues(working_chain)

        moving_points, target_points = paired_atoms(
            [working_residues[i] for i in fixed_indices],
            [reference_residues[i] for i in fixed_indices],
        )
        if len(moving_points) < MIN_FIT_ATOMS:
            raise PrepareError(
                f"only {len(moving_points)} atom(s) shared across the scaffolded "
                f"residues of chain {reference_name}; need at least {MIN_FIT_ATOMS} "
                f"to place the fold against the ligand"
            )
        rotation, moving_center, target_center, rmsd = kabsch(moving_points, target_points)
        apply_transform(working_chain, rotation, moving_center, target_center)
        rmsds.append(rmsd)

        working_chain.name = copy_name
        for residue, number in zip(working_residues, numbering):
            residue.seqid.num = number
        out[0].add_chain(working_chain)

    for ligand_name in reference_ligand:
        out[0].add_chain(reference[0][ligand_name].clone())

    out.setup_entities()
    return out, rmsds


def shell_residues(structure: gemmi.Structure, protein_chains: Sequence[str],
                   ligand_chains: Sequence[str], cutoff: float) -> List[Tuple[str, int]]:
    """(chain, residue number) for every residue with an atom within cutoff of the ligand."""
    ligand_points: List[np.ndarray] = []
    for name in ligand_chains:
        for residue in structure[0][name]:
            for atom in residue:
                ligand_points.append(np.array([atom.pos.x, atom.pos.y, atom.pos.z]))
    if not ligand_points:
        return []
    ligand_array = np.array(ligand_points)

    near: List[Tuple[str, int]] = []
    for name in protein_chains:
        for residue in structure[0][name]:
            points = np.array([[atom.pos.x, atom.pos.y, atom.pos.z] for atom in residue])
            if points.size == 0:
                continue
            # Full pairwise distance rather than a neighbour search: a residue has
            # a handful of atoms and the ligand a few hundred, so this is small,
            # and it cannot miss a contact the way a mis-set search radius can.
            distances = np.linalg.norm(points[:, None, :] - ligand_array[None, :, :], axis=2)
            if distances.min() <= cutoff:
                near.append((name, residue.seqid.num))
    return near


def alpha_carbons(structure: gemmi.Structure,
                  chain_names: Sequence[str]) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
    """(coordinates, (chain, residue number)) for every CA in the named chains."""
    points: List[np.ndarray] = []
    labels: List[Tuple[str, int]] = []
    for name in chain_names:
        for residue in structure[0][name]:
            for atom in residue:
                if atom.name == "CA":
                    points.append(np.array([atom.pos.x, atom.pos.y, atom.pos.z]))
                    labels.append((name, residue.seqid.num))
                    break
    array = np.array(points) if points else np.zeros((0, 3))
    return array, labels


@dataclass
class ClashReport:
    """What overlaps, and by how much."""
    chain_pairs: int = 0
    ligand_pairs: int = 0
    closest_chain: Optional[float] = None
    closest_ligand: Optional[float] = None
    worst: List[str] = field(default_factory=list)

    @property
    def clashing(self) -> bool:
        return bool(self.chain_pairs or self.ligand_pairs)


def find_clashes(structure: gemmi.Structure, protein_chains: Sequence[str],
                 ligand_chains: Sequence[str],
                 chain_cutoff: float = CHAIN_CLASH,
                 ligand_cutoff: float = LIGAND_CLASH) -> ClashReport:
    """CA-level overlaps: chain against chain, and chain against the ligand.

    Both copies are the same predicted fold placed on the reference's two
    chains. Where that fold differs from the reference it can end up inside its
    own partner, or inside the polysaccharide -- neither of which the reference
    dimer suffered from, so neither is caught anywhere upstream.
    """
    report = ClashReport()
    coords, labels = alpha_carbons(structure, protein_chains)
    if coords.size == 0:
        return report

    # chain against chain
    for index, first in enumerate(protein_chains):
        for second in protein_chains[index + 1:]:
            mask_a = np.array([name == first for name, _ in labels])
            mask_b = np.array([name == second for name, _ in labels])
            if not mask_a.any() or not mask_b.any():
                continue
            distances = np.linalg.norm(
                coords[mask_a][:, None, :] - coords[mask_b][None, :, :], axis=2
            )
            hits = distances < chain_cutoff
            report.chain_pairs += int(hits.sum())
            closest = float(distances.min())
            if report.closest_chain is None or closest < report.closest_chain:
                report.closest_chain = closest
            if hits.any():
                rows, cols = np.nonzero(hits)
                labels_a = [labels[i] for i in np.nonzero(mask_a)[0]]
                labels_b = [labels[i] for i in np.nonzero(mask_b)[0]]
                worst = int(np.argmin(distances[hits]))
                one, two = labels_a[rows[worst]], labels_b[cols[worst]]
                report.worst.append(
                    f"{one[0]}{one[1]}-{two[0]}{two[1]} {distances[hits][worst]:.2f}A"
                )

    # chain against ligand
    ligand_points = [
        np.array([atom.pos.x, atom.pos.y, atom.pos.z])
        for name in ligand_chains for residue in structure[0][name] for atom in residue
    ]
    if ligand_points:
        ligand_array = np.array(ligand_points)
        distances = np.linalg.norm(coords[:, None, :] - ligand_array[None, :, :], axis=2)
        hits = distances < ligand_cutoff
        report.ligand_pairs = int(hits.sum())
        report.closest_ligand = float(distances.min())
        if hits.any():
            residue_index = int(np.unravel_index(np.argmin(distances), distances.shape)[0])
            chain, number = labels[residue_index]
            report.worst.append(f"{chain}{number}-ligand {distances.min():.2f}A")
    return report


def redesign_spec(near: Sequence[Tuple[str, int]],
                  fixed_positions: Dict[str, List[int]]) -> Tuple[List[str], List[str]]:
    """(residues LigandMPNN may change, symmetry groups tying the two copies).

    A position is redesigned only if it is near the ligand in EITHER copy and is
    not scaffolded. Tying the copies means a position redesigned in one must be
    redesigned in the other, so the union is taken rather than the intersection
    -- otherwise the two chains could not stay identical.
    """
    excluded = {number for numbers in fixed_positions.values() for number in numbers}
    numbers = sorted({number for _, number in near if number not in excluded})
    residues = [f"{chain}{number}" for number in numbers for chain in COPY_CHAINS]
    groups = [",".join(f"{chain}{number}" for chain in COPY_CHAINS) for number in numbers]
    return residues, groups


# ---------------------------------------------------------------------------
# Locating the inputs
# ---------------------------------------------------------------------------

def best_model_member(names: Sequence[str], protein_id: str,
                      sequence_id: str, ranking: Dict[str, Dict[int, float]]) -> str:
    """The top-ranked AF3 model for one sequence, from the archive's own members."""
    prefix = f"{protein_id}/{jp.af3_job_name(sequence_id)}/"
    candidates = [name for name in names
                  if name.startswith(prefix) and name.endswith("model.cif")]
    if not candidates:
        raise PrepareError(f"no model cif for {sequence_id} in the transferred archive")

    scores = ranking.get(sequence_id, {})
    best: Optional[Tuple[float, str]] = None
    for name in candidates:
        parent = Path(name).parent.name
        index = None
        if "sample-" in parent:
            try:
                index = int(parent.rsplit("sample-", 1)[1])
            except ValueError:
                index = None
        score = scores.get(index) if index is not None else None
        key = score if score is not None else float("-inf")
        if best is None or key > best[0]:
            best = (key, name)
    return best[1]


def ranking_from_archive(archive: tarfile.TarFile,
                         names: Sequence[str]) -> Dict[str, Dict[int, float]]:
    """{sequence_id: {sample index: ranking score}} from the archived csvs."""
    import csv as csv_module
    table: Dict[str, Dict[int, float]] = {}
    for name in names:
        if not name.endswith("ranking_scores.csv"):
            continue
        job_dir = Path(name).parent.name
        sequence_id = job_dir[: -len("_monomer")] if job_dir.endswith("_monomer") else job_dir
        handle = archive.extractfile(name)
        if handle is None:
            continue
        rows = csv_module.DictReader(handle.read().decode("utf-8").splitlines())
        entry: Dict[int, float] = {}
        for row in rows:
            try:
                entry[int(row["sample"])] = float(row["ranking_score"])
            except (KeyError, TypeError, ValueError):
                continue
        if entry:
            table[sequence_id] = entry
    return table


def reference_bundle(stage: Path, experiment: str, group_key: str,
                     protein_id: str, scratch: Path,
                     reference_root: Optional[Path] = None
                     ) -> Tuple[gemmi.Structure, dict]:
    """The stage-02 design, ligand and all.

    reference_root points at an archived stage-02 run instead of this project's
    own 02_geometry_filtering/sorted_clean/. The 16k production run was filtered
    by the pre-rewrite geometry code, so its results are not filed under the
    current stage 02 -- reading them from where they are keeps that distinction
    honest rather than implying they cleared the current filter.
    """
    if reference_root is not None:
        archive_path = reference_root / "passed" / f"{group_key}.tar.gz"
    else:
        archive_path = jp.sorted_archive_path(
            jp.stage02_root(stage), experiment, "passed", group_key
        )
    if not archive_path.is_file():
        raise PrepareError(f"no stage 02 passed archive at {archive_path}")
    payload: dict = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        names = archive.getnames()
        wanted = [name for name in names
                  if Path(name).stem == protein_id
                  and Path(name).suffix.lower() in (".cif", ".pdb")]
        if not wanted:
            raise PrepareError(
                f"{protein_id} has no structure in {archive_path.name}; stage 02 passed "
                f"it on, so this archive should hold it"
            )
        path = extract_member(archive, wanted[0], scratch)

        # The metadata json travels beside the structure. It is what makes a run
        # generated before stage 01 kept its own record still usable: the design
        # it was built from is recorded inside it.
        metadata = [name for name in names
                    if Path(name).stem == protein_id
                    and Path(name).suffix.lower() == ".json"]
        if metadata:
            handle = archive.extractfile(metadata[0])
            if handle is not None:
                try:
                    payload = json.loads(handle.read().decode("utf-8"))
                except json.JSONDecodeError:
                    payload = {}

    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    path.unlink(missing_ok=True)
    return structure, payload


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def prepare_group(stage: Path, experiment: str, group_key: str,
                  report: PrepareReport, cutoff: float = CUTOFF,
                  reference_root: Optional[Path] = None,
                  chain_clash: float = CHAIN_CLASH,
                  ligand_clash: float = LIGAND_CLASH) -> None:
    archive_path = jp.input_archive_path(stage, experiment, group_key)
    if not archive_path.is_file():
        raise PrepareError(f"no input archive: {archive_path}")

    records = read_records(jp.fixed_residues_path(jp.stage01_root(stage), experiment))
    out_dir = jp.complex_dir(stage, experiment, group_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = jp.ligand_work_dir(stage, experiment, group_key)
    work_dir.mkdir(parents=True, exist_ok=True)
    scratch = work_dir / "_scratch"
    spec_path = jp.redesign_spec_path(stage, experiment, group_key)

    done: set = set()
    if spec_path.is_file():
        with spec_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    done.add(json.loads(line)["sequence_id"])

    fresh: List[dict] = []
    with tarfile.open(archive_path, "r:gz") as archive:
        names = archive.getnames()
        ranking = ranking_from_archive(archive, names)
        pairs = sorted({
            (parts[0], parts[1][: -len("_monomer")])
            for parts in (name.split("/") for name in names)
            if len(parts) > 2 and parts[1].endswith("_monomer")
        })

        for protein_id, sequence_id in pairs:
            if sequence_id in done and jp.complex_path(
                stage, experiment, group_key, sequence_id).is_file():
                report.already += 1
                continue
            try:
                member = best_model_member(names, protein_id, sequence_id, ranking)
                model_path = extract_member(archive, member, scratch)
                model_structure = gemmi.read_structure(str(model_path))
                model_structure.setup_entities()
                model_path.unlink(missing_ok=True)

                reference, reference_meta = reference_bundle(
                    stage, experiment, group_key, protein_id, scratch, reference_root
                )
                protein_chains, ligand_chains = classify_chains(reference)
                if len(protein_chains) != 2:
                    raise PrepareError(
                        f"the reference has {len(protein_chains)} protein chain(s) "
                        f"{protein_chains}; two are needed to place both copies"
                    )
                if not ligand_chains:
                    raise PrepareError(
                        "the reference carries no ligand chain, so there is no "
                        "polysaccharide to redesign against"
                    )

                mapped = records.get(protein_id)
                if not mapped:
                    # No stage-01 record: this structure predates the record
                    # writer. Derive the same answer from its own metadata --
                    # 'specification' is the design config RFD3 was given, and
                    # diffused_index_map pairs seed residues with generated ones.
                    specification = reference_meta.get("specification") or {}
                    diffused = reference_meta.get("diffused_index_map") or {}
                    if not specification or not diffused:
                        raise PrepareError(
                            f"no fixed residues for {protein_id}: stage 01 has no record, "
                            f"and its metadata json has no "
                            f"{'specification' if not specification else 'diffused_index_map'}"
                        )
                    mapped, unmapped = fixed_residue_map(specification, diffused)
                    if unmapped:
                        _log(f"[prepare] {protein_id}: {len(unmapped)} scaffolded "
                             f"residue(s) {unmapped} absent from diffused_index_map")
                fixed_positions = generated_positions(mapped, protein_chains)

                complex_structure, rmsds = build_complex(
                    model_structure, reference, fixed_positions,
                    protein_chains, ligand_chains,
                )
                near = shell_residues(
                    complex_structure, COPY_CHAINS,
                    [chain.name for chain in complex_structure[0]
                     if chain.name not in COPY_CHAINS],
                    cutoff,
                )
                site_positions = {
                    chain: sorted({number for numbers in fixed_positions.values()
                                   for number in numbers})
                    for chain in COPY_CHAINS
                }
                residues, groups = redesign_spec(near, site_positions)
                if not residues:
                    raise PrepareError(
                        f"no redesignable residue within {cutoff} A of the ligand "
                        f"(every one nearby is scaffolded)"
                    )

                ligand_names = [chain.name for chain in complex_structure[0]
                                if chain.name not in COPY_CHAINS]
                clashes = find_clashes(
                    complex_structure, COPY_CHAINS, ligand_names,
                    chain_clash, ligand_clash,
                )
                if clashes.clashing:
                    # Written where it can be looked at, but no spec line, so
                    # LigandMPNN never sees it.
                    reject_dir = jp.rejected_complex_dir(stage, experiment, group_key)
                    reject_dir.mkdir(parents=True, exist_ok=True)
                    reject_path = reject_dir / f"{sequence_id}.pdb"
                    complex_structure.write_pdb(str(reject_path))
                    report.clashing += 1
                    report.clash_rows.append({
                        "sequence_id": sequence_id,
                        "protein_id": protein_id,
                        "experiment_name": experiment,
                        "group": group_key,
                        "chain_clashes": clashes.chain_pairs,
                        "ligand_clashes": clashes.ligand_pairs,
                        "closest_chain": ("" if clashes.closest_chain is None
                                          else f"{clashes.closest_chain:.3f}"),
                        "closest_ligand": ("" if clashes.closest_ligand is None
                                           else f"{clashes.closest_ligand:.3f}"),
                        "fit_rmsd": f"{rmsds[0]:.4f}",
                        "worst": "; ".join(clashes.worst[:3]),
                    })
                    _log(f"[prepare] {sequence_id}: REJECTED -- "
                         f"{clashes.chain_pairs} chain-chain and "
                         f"{clashes.ligand_pairs} chain-ligand CA clash(es); "
                         f"{'; '.join(clashes.worst[:2])} -> {reject_path.name}")
                    continue

                out_path = jp.complex_path(stage, experiment, group_key, sequence_id)
                complex_structure.write_pdb(str(out_path))
                fresh.append({
                    "sequence_id": sequence_id,
                    "protein_id": protein_id,
                    "group": group_key,
                    "pdb": out_path.name,
                    "redesigned_residues": residues,
                    "symmetry_residues": groups,
                    "fixed_site": sorted(site_positions[COPY_CHAINS[0]]),
                    "fit_rmsd": [round(value, 4) for value in rmsds],
                })
                report.built += 1
                report.shell_sizes.append(len(groups))
                _log(f"[prepare] {sequence_id}: {len(groups)} shell residue(s) "
                     f"redesignable, {len(site_positions[COPY_CHAINS[0]])} scaffolded held, "
                     f"fit rmsd {rmsds[0]:.3f}/{rmsds[1]:.3f} -> {out_path.name}")
            except (PrepareError, DesignError, RuntimeError, ValueError, KeyError,
                    OSError) as exc:
                report.problems.append(f"{experiment}/{sequence_id}: {exc}")
                _log(f"[prepare] {sequence_id}: FAILED ({exc})")

    if fresh:
        with spec_path.open("a", encoding="utf-8") as handle:
            for entry in fresh:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")


CLASH_FIELDS = [
    "sequence_id", "protein_id", "experiment_name", "group",
    "chain_clashes", "ligand_clashes", "closest_chain", "closest_ligand",
    "fit_rmsd", "worst",
]


def write_clash_table(stage: Path, experiment: str, rows: Sequence[dict]) -> Path:
    """Append the rejected complexes to the experiment's clash table.

    Appended rather than rewritten: a later run over a different group must not
    erase what an earlier one recorded.
    """
    import csv
    path = jp.clash_table_path(stage, experiment)
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLASH_FIELDS)
        if fresh:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CLASH_FIELDS})
    _log(f"[prepare] {len(rows)} clash(es) recorded -> {path}")
    return path


def run_prepare(stage: Path, experiment: Optional[str] = None,
                cutoff: float = CUTOFF,
                reference_root: Optional[Path] = None,
                chain_clash: float = CHAIN_CLASH,
                ligand_clash: float = LIGAND_CLASH) -> PrepareReport:
    report = PrepareReport()
    names = [experiment] if experiment else jp.inputs_experiments(stage)
    if not names:
        _log(f"[prepare] nothing handed over under {jp.inputs_root(stage)}")
        return report
    for experiment_name in names:
        directory = jp.inputs_root(stage) / experiment_name
        if not directory.is_dir():
            continue
        before = len(report.clash_rows)
        for archive_path in sorted(directory.glob("*.tar.gz")):
            group_key = archive_path.name[: -len(".tar.gz")]
            try:
                prepare_group(stage, experiment_name, group_key, report, cutoff,
                              reference_root, chain_clash, ligand_clash)
            except PrepareError as exc:
                report.problems.append(f"{experiment_name}/{group_key}: {exc}")
                _log(f"[prepare] {experiment_name}/{group_key}: FAILED ({exc})")
        if len(report.clash_rows) > before:
            write_clash_table(stage, experiment_name, report.clash_rows[before:])
    return report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--cutoff", type=float, default=CUTOFF,
                        help=f"shell radius in angstroms (default: {CUTOFF})")
    parser.add_argument("--chain-clash", type=float, default=CHAIN_CLASH,
                        help=f"CA-CA between chains below this is a clash "
                             f"(default: {CHAIN_CLASH})")
    parser.add_argument("--ligand-clash", type=float, default=LIGAND_CLASH,
                        help=f"protein CA to ligand atom below this is a clash "
                             f"(default: {LIGAND_CLASH})")
    parser.add_argument("--reference-archives", type=Path, default=None,
                        help="a directory holding passed/<group>.tar.gz from an "
                             "archived stage-02 run, instead of this project's "
                             "02_geometry_filtering/sorted_clean/")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    report = run_prepare(
        args.stage.resolve(), args.experiment, args.cutoff,
        args.reference_archives.expanduser().resolve() if args.reference_archives else None,
        args.chain_clash, args.ligand_clash,
    )
    _log(f"[done] {report.summary()}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

