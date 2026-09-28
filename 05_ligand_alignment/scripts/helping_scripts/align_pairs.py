#!/usr/bin/env python3
"""
Stage 05, step 1: put two copies of each validated fold back on the fibril.

    inputs/<experiment>/<group>.tar.gz              <- stage 04's passed folds,
                                                       the reference complex and
                                                       the redesign spec
    inputs_prepared/<experiment>/<group>/<sequence_id>.pdb   <- open these
    tables/stage_05_alignment_<experiment>.csv

AlphaFold folded one chain with the fibril present, which is what makes the
prediction trustworthy -- but the fibril it returns is its own guess. The fibril
used here is the one from the input complex, in the geometry the design was
diffused against, because that is the fibre the linker has to route around and
the pose the design was built for. AF3's protein, the design's sugar.

The fold is superposed onto BOTH protein chains of its reference complex, on the
scaffolded residues only. Those residues are the binding site; they were held
fixed from stage 01 onward, and aligning on them puts the site where it belongs
even when the rest of the fold has drifted. All atoms the two copies of a
residue share are used, not just CA, because a handful of CA atoms barely
constrain a rotation.

The result is one sequence occupying both seed sides -- and then the fibre is
dropped, because by that point it has done both of its jobs. It placed the
copies, and it was the thing their clearance was measured against. What is
written is one pdb per sequence holding two protein chains and nothing else:
the linker is generated on protein alone, and a fibre wanted later comes back
by re-running this stage, whose input archives still carry the reference
complexes.

Both fibres go, not just one. AlphaFold predicts its own, which was never used
here -- the reference's is what the design was diffused against -- and neither
is kept.

NOTHING IS REJECTED HERE. Clashes are still measured, on all heavy atoms, and
every pair gets a row in the table; no pair is turned away on them. Two reasons,
and the second is the one that matters:

    AlphaFold folds slightly differently from the design. A contact measured
    between two copies of ITS prediction is partly its own deviation, so
    rejecting on it discards designs for an error that is not theirs.

    The question has already been asked where it is objective. Stage 02 filters
    the designed coordinates at zero protein-protein contacts under 1.8 A and
    zero protein-ligand under 2.2 A. Anything that reached here passed that on
    the geometry RFD3 was actually given.

So the columns are worth having -- a pair whose copies interpenetrate badly is
worth looking at before spending GPU time linking it -- and the verdict is
yours, not the code's.
"""
from __future__ import annotations

import argparse
import csv
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
from structures import classify_chains, drop_chains  # noqa: E402

STAGE = Path(__file__).resolve().parents[2]

MIN_FIT_ATOMS = 6     # below this a superposition is not meaningfully constrained
COPY_CHAINS = ("A", "B")

# Heavy-atom cutoffs. 2.5 A is below the 2.8-3.0 A of a hydrogen bond and well
# below the 3.4 A of ordinary carbon packing, so a real interface survives and
# only genuine interpenetration is flagged.
PROTEIN_CLASH = 2.5
LIGAND_CLASH = 2.5

ALIGNMENT_FIELDS = [
    "sequence_id", "protein_id", "experiment_name", "group",
    "fit_rmsd_a", "fit_rmsd_b",
    "chain_contacts", "ligand_contacts", "closest_chain", "closest_ligand",
    "closest_pair",
]


class AlignError(Exception):
    """One fold could not be placed."""


def _log(*parts: object) -> None:
    print(*parts, flush=True)


@dataclass
class AlignReport:
    built: int = 0
    already: int = 0
    touching: int = 0     # pairs with a heavy-atom contact under the cutoff
    fit_rmsds: List[float] = field(default_factory=list)
    rows: List[dict] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        if not self.built:
            return f"align: nothing new ({self.already} already built)"
        rmsds = self.fit_rmsds
        fit = (f", fit rmsd {min(rmsds):.2f}-{max(rmsds):.2f} "
               f"(median {sorted(rmsds)[len(rmsds) // 2]:.2f})") if rmsds else ""
        # "noted" rather than "rejected": all of them went on. Worded this way
        # because a count next to a filter reads as a count of what the filter
        # removed, and nothing was removed.
        return (f"align: {self.built} pair(s) built, {self.already} already"
                f"{fit}, {self.touching} with heavy-atom contacts noted "
                f"(none rejected)")


# ---------------------------------------------------------------------------
# Superposition
# ---------------------------------------------------------------------------

def kabsch(moving: np.ndarray,
           target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
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
                 target_residues: Sequence[gemmi.Residue]
                 ) -> Tuple[np.ndarray, np.ndarray]:
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


def polymer_residues(chain: gemmi.Chain) -> List[gemmi.Residue]:
    out: List[gemmi.Residue] = []
    for residue in chain:
        info = gemmi.find_tabulated_residue(residue.name)
        if info is not None and info.is_amino_acid():
            out.append(residue)
    return out


def build_pair(model_structure: gemmi.Structure, reference: gemmi.Structure,
               fixed_positions: Dict[str, List[int]],
               reference_protein: Sequence[str],
               reference_ligand: Sequence[str]) -> Tuple[gemmi.Structure, List[float]]:
    """Two superposed copies of the AF3 chain, plus the reference's fibril."""
    # gemmi returns a structure with no model at all when a cif is missing the
    # columns it builds one from, and indexing that raises a bare IndexError
    # naming nothing. Say which file and what was wrong with it instead.
    for label, structure in (("AF3 model", model_structure), ("reference", reference)):
        if len(structure) == 0:
            raise AlignError(
                f"the {label} parsed to zero models -- the file is truncated, or its "
                f"atom_site loop is missing the columns gemmi builds a model from "
                f"(label_asym_id, pdbx_PDB_model_num)"
            )
    model_chains = [chain for chain in model_structure[0] if polymer_residues(chain)]
    if len(model_chains) != 1:
        raise AlignError(
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
            raise AlignError(
                f"chain {reference_name} has {len(reference_residues)} residues but the "
                f"AF3 model has {len(model_residues)} -- they should be the same design, "
                f"so the numbering cannot be transferred safely"
            )

        # Residue i of the prediction is residue i of the reference: AF3 folded
        # exactly the sequence LigandMPNN wrote for this backbone, so the two run
        # one-to-one. The length check above is what makes that safe to assume.
        numbering = [residue.seqid.num for residue in reference_residues]
        positions = fixed_positions.get(reference_name) or []
        index_of = {number: index for index, number in enumerate(numbering)}
        fixed_indices = [index_of[number] for number in positions if number in index_of]
        if not fixed_indices:
            raise AlignError(
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
            raise AlignError(
                f"only {len(moving_points)} atom(s) shared across the scaffolded "
                f"residues of chain {reference_name}; need at least {MIN_FIT_ATOMS} "
                f"to place the fold against the fibril"
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


# ---------------------------------------------------------------------------
# Clashes, on every heavy atom
# ---------------------------------------------------------------------------

def heavy_atoms(chain: gemmi.Chain) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
    """(coordinates, (residue name, number) per atom), hydrogens excluded."""
    points: List[List[float]] = []
    labels: List[Tuple[str, int]] = []
    for residue in chain:
        for atom in residue:
            if atom.element == gemmi.Element("H"):
                continue
            points.append([atom.pos.x, atom.pos.y, atom.pos.z])
            labels.append((residue.name, residue.seqid.num))
    array = np.array(points) if points else np.zeros((0, 3))
    return array, labels


@dataclass
class ClashReport:
    chain_pairs: int = 0
    ligand_pairs: int = 0
    closest_chain: Optional[float] = None
    closest_ligand: Optional[float] = None
    worst: List[str] = field(default_factory=list)

    @property
    def clashing(self) -> bool:
        return bool(self.chain_pairs or self.ligand_pairs)


def _closest(first: np.ndarray, second: np.ndarray,
             labels_a: Sequence[Tuple[str, int]], labels_b: Sequence[Tuple[str, int]],
             cutoff: float, tag_a: str, tag_b: str) -> Tuple[int, Optional[float], List[str]]:
    if first.size == 0 or second.size == 0:
        return 0, None, []
    distances = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
    hits = distances < cutoff
    closest = float(distances.min())
    worst: List[str] = []
    if hits.any():
        rows, cols = np.nonzero(hits)
        index = int(np.argmin(distances[hits]))
        one, two = labels_a[rows[index]], labels_b[cols[index]]
        worst.append(f"{tag_a}{one[0]}{one[1]}-{tag_b}{two[0]}{two[1]} "
                     f"{distances[hits][index]:.2f}A")
    return int(hits.sum()), closest, worst


def find_clashes(structure: gemmi.Structure, protein_chains: Sequence[str],
                 ligand_chains: Sequence[str],
                 protein_cutoff: float = PROTEIN_CLASH,
                 ligand_cutoff: float = LIGAND_CLASH) -> ClashReport:
    """Heavy-atom overlaps: chain against chain, and each chain against the fibril."""
    report = ClashReport()
    coords = {name: heavy_atoms(structure[0][name]) for name in protein_chains}
    ligands = {name: heavy_atoms(structure[0][name]) for name in ligand_chains}

    for index, first in enumerate(protein_chains):
        for second in protein_chains[index + 1:]:
            points_a, labels_a = coords[first]
            points_b, labels_b = coords[second]
            count, closest, worst = _closest(
                points_a, points_b, labels_a, labels_b, protein_cutoff, first, second
            )
            report.chain_pairs += count
            if closest is not None and (report.closest_chain is None
                                        or closest < report.closest_chain):
                report.closest_chain = closest
            report.worst.extend(worst)

    for name in protein_chains:
        points_a, labels_a = coords[name]
        for ligand_name in ligand_chains:
            points_b, labels_b = ligands[ligand_name]
            count, closest, worst = _closest(
                points_a, points_b, labels_a, labels_b, ligand_cutoff, name, "lig "
            )
            report.ligand_pairs += count
            if closest is not None and (report.closest_ligand is None
                                        or closest < report.closest_ligand):
                report.closest_ligand = closest
            report.worst.extend(worst)
    return report


# ---------------------------------------------------------------------------
# Reading stage 04's handover
# ---------------------------------------------------------------------------

def ranking_from_archive(archive: tarfile.TarFile,
                         names: Sequence[str]) -> Dict[str, Dict[int, float]]:
    """{sequence_id: {sample index: ranking score}} from the archived csvs."""
    table: Dict[str, Dict[int, float]] = {}
    ending = f"_{jp.LIGAND_SUFFIX}"
    for name in names:
        if not name.endswith("ranking_scores.csv"):
            continue
        job_dir = Path(name).parent.name
        sequence_id = job_dir[: -len(ending)] if job_dir.endswith(ending) else job_dir
        handle = archive.extractfile(name)
        if handle is None:
            continue
        entry: Dict[int, float] = {}
        for row in csv.DictReader(handle.read().decode("utf-8").splitlines()):
            try:
                entry[int(row["sample"])] = float(row["ranking_score"])
            except (KeyError, TypeError, ValueError):
                continue
        if entry:
            table[sequence_id] = entry
    return table


def best_model_member(names: Sequence[str], protein_id: str, sequence_id: str,
                      ranking: Dict[str, Dict[int, float]]) -> str:
    """The top-ranked AF3 model for one sequence, from the archive's own members."""
    prefix = f"{protein_id}/{jp.af3_job_name(sequence_id)}/"
    candidates = [name for name in names
                  if name.startswith(prefix) and name.endswith("model.cif")]
    if not candidates:
        raise AlignError(f"no model cif for {sequence_id} in the transferred archive")

    scores = ranking.get(sequence_id, {})
    best: Optional[Tuple[float, str]] = None
    for name in candidates:
        parent = Path(name).parent.name
        index: Optional[int] = None
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


def scaffolded_positions(spec: Optional[dict],
                         protein_chains: Sequence[str]) -> Dict[str, List[int]]:
    """{chain: held residue numbers} from the redesign spec stage 03 wrote.

    The spec records one list, because the two copies are tied and share a
    numbering; it is applied to both reference chains.
    """
    if not spec:
        return {}
    held = [int(number) for number in spec.get("held", [])]
    return {name: list(held) for name in protein_chains}


def archived_sequences(stage: Path, experiment: str, group_key: str) -> set:
    """The sequence ids already inside this group's passed archive.

    The archive is the record of what has been done, which is what lets the
    loose pdb be deleted as soon as it is safely in there. A missing archive is
    not an error: it simply means nothing has been aligned for this group yet.
    """
    path = jp.sorted_archive_path(stage, experiment, "passed", group_key)
    if not path.is_file():
        return set()
    with tarfile.open(path, "r:gz") as archive:
        return {Path(name).stem for name in archive.getnames()
                if name.endswith(".pdb")}


def align_group(stage: Path, experiment: str, group_key: str, report: AlignReport,
                protein_cutoff: float = PROTEIN_CLASH,
                ligand_cutoff: float = LIGAND_CLASH) -> None:
    archive_path = jp.input_archive_path(stage, experiment, group_key)
    if not archive_path.is_file():
        raise AlignError(f"no input archive: {archive_path}")

    out_dir = jp.pair_dir(stage, experiment, group_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "_scratch"

    # What this group has already produced, read from the ARCHIVE rather than
    # from the loose files. The loose pdb is transient now -- cleanup removes it
    # once it is archived, so its absence says nothing about whether the work
    # was done. Checking the file would re-align every sequence on every run.
    done = archived_sequences(stage, experiment, group_key)

    with tarfile.open(archive_path, "r:gz") as archive:
        names = archive.getnames()
        ranking = ranking_from_archive(archive, names)

        references: Dict[str, str] = {}
        specs: Dict[str, str] = {}
        pairs: List[Tuple[str, str]] = []
        for name in sorted(names):
            parts = (name[2:] if name.startswith("./") else name).split("/")
            if len(parts) == 2 and parts[1] == f"{parts[0]}.pdb":
                references[parts[0]] = name
            elif len(parts) == 2 and parts[1].endswith("_redesign.json"):
                specs[parts[0]] = name
            elif len(parts) > 2 and parts[1].endswith(f"_{jp.LIGAND_SUFFIX}"):
                sequence_id = parts[1][: -len(f"_{jp.LIGAND_SUFFIX}")]
                if (parts[0], sequence_id) not in pairs:
                    pairs.append((parts[0], sequence_id))

        for protein_id, sequence_id in pairs:
            if (sequence_id in done
                    or jp.pair_path(stage, experiment, group_key,
                                    sequence_id).is_file()):
                report.already += 1
                continue
            try:
                if protein_id not in references:
                    raise AlignError(
                        f"no reference complex for {protein_id} in "
                        f"{archive_path.name}; stage 04 should have sent it"
                    )
                model_path = extract_member(
                    archive, best_model_member(names, protein_id, sequence_id, ranking),
                    scratch,
                )
                model_structure = gemmi.read_structure(str(model_path))
                model_structure.setup_entities()
                model_path.unlink(missing_ok=True)

                reference_path = extract_member(archive, references[protein_id], scratch)
                reference = gemmi.read_structure(str(reference_path))
                reference.setup_entities()
                reference_path.unlink(missing_ok=True)

                reference_protein, reference_ligand = classify_chains(reference)
                if len(reference_protein) != 2:
                    raise AlignError(
                        f"the reference has {len(reference_protein)} protein chain(s) "
                        f"{reference_protein}; two are needed to place both copies"
                    )
                if not reference_ligand:
                    raise AlignError("the reference carries no fibril to place against")

                spec = None
                if protein_id in specs:
                    handle = archive.extractfile(specs[protein_id])
                    if handle is not None:
                        spec = json.loads(handle.read().decode("utf-8"))
                fixed_positions = scaffolded_positions(spec, reference_protein)
                if not fixed_positions:
                    raise AlignError(
                        "no redesign spec for this protein, so the scaffolded "
                        "residues to superpose on are unknown"
                    )

                pair, rmsds = build_pair(
                    model_structure, reference, fixed_positions,
                    reference_protein, reference_ligand,
                )
                clashes = find_clashes(
                    pair, list(COPY_CHAINS),
                    [chain.name for chain in pair[0]
                     if chain.name not in COPY_CHAINS],
                    protein_cutoff, ligand_cutoff,
                )
            except (AlignError, RuntimeError, ValueError, KeyError, OSError,
                    json.JSONDecodeError) as exc:
                report.problems.append(f"{experiment}/{sequence_id}: {exc}")
                _log(f"[align] {sequence_id}: FAILED ({exc})")
                continue

            # The fibril has done its job by here and does not survive it. It
            # was needed twice -- to place the copies, and to measure how close
            # they came to it -- and both are done, so it comes out before the
            # file is written and nothing downstream carries it. The linker is
            # generated on protein alone; a fibre needed later is regenerable by
            # re-running this stage, whose input archives still hold the
            # reference complexes.
            fibril_chains = [chain.name for chain in pair[0]
                             if chain.name not in COPY_CHAINS]
            if fibril_chains:
                drop_chains(pair, fibril_chains)

            # Every pair is kept. The contact numbers go in the table beside it
            # so you can sort on them; they decide nothing here.
            out_path = jp.pair_path(stage, experiment, group_key, sequence_id)
            pair.write_pdb(str(out_path))
            report.built += 1
            report.fit_rmsds.extend(rmsds)
            report.touching += int(clashes.clashing)
            report.rows.append({
                "sequence_id": sequence_id,
                "protein_id": protein_id,
                "experiment_name": experiment,
                "group": group_key,
                "fit_rmsd_a": f"{rmsds[0]:.4f}",
                "fit_rmsd_b": f"{rmsds[1]:.4f}",
                "chain_contacts": clashes.chain_pairs,
                "ligand_contacts": clashes.ligand_pairs,
                "closest_chain": ("" if clashes.closest_chain is None
                                  else f"{clashes.closest_chain:.3f}"),
                "closest_ligand": ("" if clashes.closest_ligand is None
                                   else f"{clashes.closest_ligand:.3f}"),
                "closest_pair": "; ".join(clashes.worst[:3]),
            })
            note = ""
            if clashes.clashing:
                note = (f", {clashes.chain_pairs} chain-chain and "
                        f"{clashes.ligand_pairs} chain-fibril contact(s) under "
                        f"{protein_cutoff:.1f} A")
            _log(f"[align] {sequence_id}: pair built, fit rmsd "
                 f"{rmsds[0]:.3f}/{rmsds[1]:.3f}{note} -> {out_path.name}")


    if scratch.exists() and not any(scratch.iterdir()):
        scratch.rmdir()


def write_alignment_table(stage: Path, experiment: str,
                          rows: Sequence[dict]) -> Path:
    """Append every pair built to the experiment's alignment table."""
    path = jp.alignment_table_path(stage, experiment)
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ALIGNMENT_FIELDS)
        if fresh:
            writer.writeheader()
        for row in rows:
            writer.writerow({field_: row.get(field_, "") for field_ in ALIGNMENT_FIELDS})
    _log(f"[align] {len(rows)} pair(s) recorded -> {path}")
    return path


def run_align(stage: Path, experiment: Optional[str] = None,
              protein_cutoff: float = PROTEIN_CLASH,
              ligand_cutoff: float = LIGAND_CLASH) -> AlignReport:
    report = AlignReport()
    names = [experiment] if experiment else jp.inputs_experiments(stage)
    if not names:
        _log(f"[align] nothing handed over under {jp.inputs_root(stage)}")
        return report

    for experiment_name in names:
        directory = jp.inputs_root(stage) / experiment_name
        if not directory.is_dir():
            continue
        before = len(report.rows)
        for archive_path in sorted(directory.glob("*.tar.gz")):
            group_key = archive_path.name[: -len(".tar.gz")]
            try:
                align_group(stage, experiment_name, group_key, report,
                            protein_cutoff, ligand_cutoff)
            except AlignError as exc:
                report.problems.append(f"{experiment_name}/{group_key}: {exc}")
                _log(f"[align] {experiment_name}/{group_key}: FAILED ({exc})")
        if len(report.rows) > before:
            write_alignment_table(stage, experiment_name, report.rows[before:])
    return report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--protein-clash", type=float, default=PROTEIN_CLASH,
                        help=f"heavy atom to heavy atom between the two copies, "
                             f"below this is a clash (default: {PROTEIN_CLASH})")
    parser.add_argument("--ligand-clash", type=float, default=LIGAND_CLASH,
                        help=f"protein heavy atom to fibril heavy atom (default: "
                             f"{LIGAND_CLASH})")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    report = run_align(args.stage.resolve(), args.experiment,
                       args.protein_clash, args.ligand_clash)
    _log(f"[done] {report.summary()}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
