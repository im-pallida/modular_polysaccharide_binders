#!/usr/bin/env python3
"""
Stage 04, step 3: score every folded monomer and sort it passed or rejected.

    outputs/<sequence_id>_ligand/                       <- AF3 wrote
    inputs/<experiment>/<group>.tar.gz                  <- the reference complex,
                                                           sent by stage 03
    sorted_clean/<experiment>/<passed|rejected>/<group>.tar.gz
    tables/stage_04_results_<experiment>.csv

The metric is the one used before: a ChimeraX matchmaker-style RMSD. Pairwise
sequence alignment (BLOSUM62) establishes which residue corresponds to which,
CA atoms are superposed by Kabsch, and the fit is iterated with outlier pruning
at PRUNE_CUTOFF. matched_fraction is what survived pruning over what the
alignment paired, so it says how much of the chain the RMSD actually describes.
A sequence passes when rmsd < RMSD_THRESHOLD and matched_fraction > FRACTION.

Of AF3's samples, only the one AF3 ranks first is scored -- that is the
prediction you would use, so the verdict should be about it.

The reference is the complex stage 03 designed onto, which travels in the input
archive beside the sequences. Only its protein is used here -- one chain of the
tied dimer, since the prediction is a monomer. The fibril in both structures is
ignored by the RMSD: every polysaccharide residue maps to 'X', which
protein_chains() excludes, so the ligand can be present in the fold without
entering the metric.

site_rmsd is a diagnostic, never a gate. After superposing on the pruned core,
it measures the deviation at the scaffolded residues alone -- the "ALL" fixed
side chains stage 01 recorded. A design can have an excellent whole-chain RMSD
while the binding site itself has drifted, and that column is where you would
see it. Blank when stage 01 recorded no fixed residues for that structure.

The contact columns answer a question RMSD cannot. RMSD is measured after
superposition, so it says whether the FOLD is right and nothing at all about
where that fold ended up relative to the fibre it was given: a design that folds
beautifully and drifts off into solvent scores exactly as well as one sitting on
the cellulose. So the prediction is also measured in its own coordinates --

    site_contacts / site_contact_fraction   how many of the ALL-fixed residues
                                            are still within SITE_CONTACT_CUTOFF
                                            of the fibre, heavy atom to heavy atom
    site_closest                            the nearest of them
    chain_closest                           the nearest approach of the whole
                                            monomer, which is the blunt "is it
                                            on the fibre at all" number

-- and site_contact_fraction is a gate when --min-site-contact is above zero.
It is zero by default because the right value is a property of these seeds, not
of the method: site_contacts.py measures it and prints the number to pass.
"""
from __future__ import annotations

import argparse
import sys
import tarfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from Bio import Align
    from Bio.Align import substitution_matrices
except ImportError:  # pragma: no cover - environment problem, not a code path
    raise SystemExit(
        "ERROR: biopython is not installed in this interpreter.\n"
        "       pip install biopython, into the same environment as gemmi and numpy."
    )

try:
    import gemmi
except ImportError:  # pragma: no cover - environment problem, not a code path
    raise SystemExit(
        "ERROR: gemmi is not installed in this interpreter.\n"
        "       pip install gemmi, into the same environment as biopython and numpy."
    )

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import job_paths as jp  # noqa: E402
from archives import load_table, merge_members_into_archive, save_table  # noqa: E402
from designs import generated_positions, read_records  # noqa: E402
from structures import (  # noqa: E402
    SITE_CONTACT_CUTOFF,
    StructureError,
    chain_approach,
    classify_chains,
    most_contacted,
    site_approaches,
)

STAGE = Path(__file__).resolve().parents[2]

RMSD_THRESHOLD = 2.5
FRACTION_THRESHOLD = 0.8
PRUNE_CUTOFF = 2.0
MAX_ITERATIONS = 5

# What fraction of the seed's ALL-fixed residues must still be within
# SITE_CONTACT_CUTOFF of the fibre in the prediction. Zero means the columns are
# filled in but nothing is rejected on them.
#
# Off by default on purpose. The right value depends on whether every ALL-fixed
# residue in these seeds is a binding residue, which only the seeds know --
# site_contacts.py measures it and prints the number to pass. A gate guessed too
# high rejects everything and reads exactly like a stage that ran and found
# nothing, which has already cost time here more than once.
MIN_SITE_CONTACT = 0.0

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

RESULT_FIELDS = [
    "sequence_id", "protein_id", "experiment_name", "group", "time_stamp",
    "rmsd", "matched_pairs", "total_aligned_pairs", "matched_fraction",
    "ranking_score", "site_rmsd", "site_residues",
    "site_contacts", "site_contact_fraction", "site_closest", "chain_closest",
    "status",
]


class ScoreError(Exception):
    """One sequence could not be scored."""


def _log(*parts: object) -> None:
    print(*parts, flush=True)


@dataclass
class Chain:
    """One chain's CA atoms, in residue order."""
    chain_id: str
    residue_numbers: List[int]
    sequence: str
    coords: np.ndarray

    def index_of(self, residue_number: int) -> Optional[int]:
        try:
            return self.residue_numbers.index(residue_number)
        except ValueError:
            return None


@dataclass
class ScoreReport:
    evaluated: int = 0
    passed: int = 0
    rejected: int = 0
    not_folded: int = 0
    already: int = 0
    no_site: int = 0
    off_fibre: int = 0
    tables: List[Path] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        if not self.evaluated:
            return (f"score: nothing new ({self.already} already scored, "
                    f"{self.not_folded} not folded yet)")
        return (f"score: {self.evaluated} evaluated (PASSED={self.passed}, "
                f"REJECTED={self.rejected}), {self.already} already scored, "
                f"{self.not_folded} not folded yet")


# ---------------------------------------------------------------------------
# Reading structures
# ---------------------------------------------------------------------------

def parse_mmcif_atom_site(cif_text: str) -> List[Dict[str, str]]:
    """The _atom_site loop as dicts, read by column header rather than position."""
    lines = cif_text.splitlines()
    columns: List[str] = []
    data_start = None
    for index, line in enumerate(lines):
        if line.strip() != "loop_":
            continue
        cursor = index + 1
        peek: List[str] = []
        while cursor < len(lines) and lines[cursor].strip().startswith("_atom_site."):
            peek.append(lines[cursor].strip()[len("_atom_site."):])
            cursor += 1
        if peek:
            columns, data_start = peek, cursor
            break
    if not columns or data_start is None:
        raise ScoreError("no _atom_site loop_ in this mmCIF")

    position = {name: index for index, name in enumerate(columns)}
    rows: List[Dict[str, str]] = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("_") or stripped == "loop_":
            break
        fields = stripped.split()
        if fields[0] not in ("ATOM", "HETATM"):
            break
        rows.append({name: fields[position[name]] for name in columns
                     if position[name] < len(fields)})
    return rows


def chains_from_mmcif(cif_text: str) -> List[Chain]:
    by_chain: Dict[str, Dict[int, Tuple[str, Tuple[float, float, float]]]] = {}
    for row in parse_mmcif_atom_site(cif_text):
        if row.get("label_atom_id") != "CA":
            continue
        chain_id = row.get("auth_asym_id") or row.get("label_asym_id")
        if chain_id is None:
            continue
        by_chain.setdefault(chain_id, {})[int(row["auth_seq_id"])] = (
            row.get("label_comp_id", "UNK"),
            (float(row["Cartn_x"]), float(row["Cartn_y"]), float(row["Cartn_z"])),
        )
    return [_build_chain(chain_id, residues) for chain_id, residues in sorted(by_chain.items())]


def chains_from_pdb(pdb_text: str) -> List[Chain]:
    by_chain: Dict[str, Dict[int, Tuple[str, Tuple[float, float, float]]]] = {}
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if line[12:16].strip() != "CA":
            continue
        chain_id = line[21].strip() or " "
        by_chain.setdefault(chain_id, {})[int(line[22:26])] = (
            line[17:20].strip(),
            (float(line[30:38]), float(line[38:46]), float(line[46:54])),
        )
    return [_build_chain(chain_id, residues) for chain_id, residues in sorted(by_chain.items())]


def _build_chain(chain_id, residues) -> Chain:
    ordered = sorted(residues.items())
    return Chain(
        chain_id=chain_id,
        residue_numbers=[number for number, _ in ordered],
        sequence="".join(AA3TO1.get(name, "X") for _, (name, _) in ordered),
        coords=(np.array([xyz for _, (_, xyz) in ordered], dtype=float)
                if ordered else np.zeros((0, 3))),
    )


def protein_chains(chains: Sequence[Chain], minimum: int = 10) -> List[Chain]:
    """Chains long enough to be protein, so a stray HETATM chain cannot be picked.

    Length rather than residue type because this reads text, not gemmi: a
    polysaccharide chain has a handful of residues that all map to 'X'.
    """
    return [chain for chain in chains
            if len(chain.sequence) >= minimum and chain.sequence.count("X") < len(chain.sequence)]


# ---------------------------------------------------------------------------
# Matchmaker-style superposition
# ---------------------------------------------------------------------------

@dataclass
class Fit:
    rmsd: float
    matched_pairs: int
    total_pairs: int
    rotation: np.ndarray
    moving_center: np.ndarray
    target_center: np.ndarray

    @property
    def matched_fraction(self) -> float:
        return self.matched_pairs / self.total_pairs if self.total_pairs else 0.0

    def apply(self, coords: np.ndarray) -> np.ndarray:
        """Put arbitrary moving-structure coordinates into the target's frame."""
        return (coords - self.moving_center) @ self.rotation.T + self.target_center


def correspondence(seq_moving: str, seq_target: str) -> List[Tuple[int, int]]:
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(seq_moving, seq_target)[0]
    pairs: List[Tuple[int, int]] = []
    for (a_start, a_end), (b_start, b_end) in zip(*alignment.aligned):
        for offset in range(a_end - a_start):
            pairs.append((a_start + offset, b_start + offset))
    return pairs


def kabsch(moving: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """(rmsd, rotation, moving centroid, target centroid). No reflections."""
    moving_center = moving.mean(axis=0)
    target_center = target.mean(axis=0)
    a = moving - moving_center
    b = target - target_center
    u, _, vt = np.linalg.svd(a.T @ b)
    # Force a proper rotation: an unguarded SVD can return a reflection, which
    # would superpose a mirror image and report a flatteringly low rmsd.
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    deviation = a @ rotation.T - b
    rmsd = float(np.sqrt((deviation ** 2).sum(axis=1).mean()))
    return rmsd, rotation, moving_center, target_center


def matchmaker(moving: Chain, target: Chain,
               prune_cutoff: float = PRUNE_CUTOFF,
               max_iterations: int = MAX_ITERATIONS) -> Tuple[Fit, List[Tuple[int, int]]]:
    """Align by sequence, superpose CAs, prune outliers, repeat."""
    pairs = correspondence(moving.sequence, target.sequence)
    if len(pairs) < 3:
        raise ScoreError(f"only {len(pairs)} aligned residue pair(s); need at least 3")

    moving_coords = np.array([moving.coords[i] for i, _ in pairs])
    target_coords = np.array([target.coords[j] for _, j in pairs])

    keep = np.ones(len(pairs), dtype=bool)
    rmsd, rotation, moving_center, target_center = kabsch(moving_coords, target_coords)
    for _ in range(max_iterations):
        rmsd, rotation, moving_center, target_center = kabsch(
            moving_coords[keep], target_coords[keep]
        )
        transformed = (moving_coords - moving_center) @ rotation.T + target_center
        within = np.linalg.norm(transformed - target_coords, axis=1) <= prune_cutoff
        # Pruning stops when nothing more would be dropped, or when dropping
        # more would leave too few atoms to define a superposition at all.
        if (within == keep).all() or within.sum() < 3:
            break
        keep = within

    fit = Fit(rmsd, int(keep.sum()), len(pairs), rotation, moving_center, target_center)
    return fit, pairs


def site_pairs(pairs: Sequence[Tuple[int, int]], target: Chain,
               site_residues: Sequence[int]) -> List[Tuple[int, int]]:
    """The aligned pairs that land on the scaffolded residues.

    Shared by the two site measurements, because they have to be talking about
    the same residues or the columns beside each other in the table mean
    different things. Selection is by residue number in the REFERENCE and then
    carried across the alignment, since AlphaFold renumbers its output from 1
    and the seed does not: matching the numbers directly would silently select
    the wrong residues whenever the seed's numbering did not start at 1.
    """
    wanted = set(site_residues)
    target_indices = {index for index, number in enumerate(target.residue_numbers)
                      if number in wanted}
    return [(i, j) for i, j in pairs if j in target_indices]


def site_deviation(fit: Fit, selected: Sequence[Tuple[int, int]], moving: Chain,
                   target: Chain) -> Tuple[Optional[float], int]:
    """RMSD at the scaffolded residues under the whole-chain superposition.

    Not its own fit: superposing five atoms on five atoms would fit almost
    anything. The question is where the site landed once the fold as a whole
    was aligned.
    """
    if not selected:
        return None, 0
    moved = fit.apply(np.array([moving.coords[i] for i, _ in selected]))
    reference = np.array([target.coords[j] for _, j in selected])
    deviation = moved - reference
    return float(np.sqrt((deviation ** 2).sum(axis=1).mean())), len(selected)


# ---------------------------------------------------------------------------
# Did the prediction stay on the fibre?
# ---------------------------------------------------------------------------

@dataclass
class Contacts:
    """How much of the designed interface AlphaFold kept."""
    contacting: int
    measured: int
    site_closest: Optional[float]
    chain_closest: Optional[float]

    @property
    def fraction(self) -> float:
        return self.contacting / self.measured if self.measured else 0.0


def contact_report(cif_path: Path, residue_numbers: Sequence[int],
                   cutoff: float = SITE_CONTACT_CUTOFF) -> Contacts:
    """Where the prediction sits relative to the fibre it was folded with.

    Measured in AlphaFold's OWN coordinates, not after superposing onto the
    seed. That is the whole point of the check: superpose first and the answer
    is almost the RMSD again, because a fold that matches the design is by
    construction sitting where the design sat. Asking it in the prediction's
    own frame asks the different and more useful question -- given the fibre
    right there, did AlphaFold keep the protein on it, or fold a perfectly good
    domain floating in solvent?
    """
    structure = gemmi.read_structure(str(cif_path))
    structure.setup_entities()
    protein_chains, ligand_chains = classify_chains(structure)
    if not protein_chains:
        raise ScoreError(f"{cif_path.name}: no protein chain")
    if not ligand_chains:
        raise ScoreError(
            f"{cif_path.name}: the prediction has no ligand chain, so it was "
            f"folded without the fibre -- check the json that produced it"
        )
    ligand, _ = most_contacted(structure, protein_chains, ligand_chains)
    chain = protein_chains[0]

    approaches = site_approaches(structure, chain, residue_numbers, ligand)
    contacting = sum(1 for distance in approaches.values() if distance <= cutoff)
    return Contacts(
        contacting=contacting,
        measured=len(approaches),
        site_closest=min(approaches.values()) if approaches else None,
        chain_closest=chain_approach(structure, chain, ligand),
    )


def verdict(rmsd: float, matched_fraction: float, contacts: Optional[Contacts],
            rmsd_threshold: float, fraction_threshold: float,
            min_site_contact: float) -> Tuple[bool, bool]:
    """(passed, on_fibre) -- the whole pass decision, in one place.

    contacts is None when stage 01 recorded no fixed residues for this design.
    Such a fold is judged on the rest rather than rejected, because a gap in a
    record is not evidence about a structure.
    """
    on_fibre = contacts is None or contacts.fraction >= min_site_contact
    return (rmsd < rmsd_threshold
            and matched_fraction > fraction_threshold
            and on_fibre), on_fibre


# ---------------------------------------------------------------------------
# AF3 output discovery
# ---------------------------------------------------------------------------

def model_cif_in(directory: Path, job_name: str) -> Optional[Path]:
    """The model mmCIF in one directory, whichever way AF3 named it.

    AF3 has shipped two conventions and both are in use here: the cluster's
    3.0.1 image prefixes every file with the job name
    ('<job>_seed-1_sample-0_model.cif'), while the local checkout writes a bare
    'model.cif'. A single glob cannot cover both -- '*_model.cif' silently
    matches nothing against 'model.cif', which is exactly how a folder full of
    perfectly good predictions came back as "no model cif".
    """
    named = [
        directory / f"{job_name}_{directory.name}_model.cif",
        directory / "model.cif",
        directory / f"{job_name}_model.cif",
    ]
    for candidate in named:
        if candidate.is_file():
            return candidate
    remaining = sorted(path for path in directory.glob("*model.cif") if path.is_file())
    return remaining[0] if remaining else None


def describe(directory: Path, limit: int = 12) -> str:
    """What is actually in a directory, for an error message worth reading."""
    if not directory.is_dir():
        return "the directory does not exist"
    entries = sorted(directory.iterdir())
    shown = []
    for entry in entries[:limit]:
        if entry.is_dir():
            inner = sorted(path.name for path in entry.iterdir())[:6]
            shown.append(f"{entry.name}/ -> {inner}")
        else:
            shown.append(entry.name)
    if len(entries) > limit:
        shown.append(f"... and {len(entries) - limit} more")
    return "; ".join(shown) if shown else "it is empty"


def ranked_samples(stage: Path, job_name: str) -> List[Tuple[int, Path]]:
    """[(sample index, model cif)] for one job, lowest index first."""
    job_dir = jp.af3_output_dir(stage, job_name)
    found: List[Tuple[int, Path]] = []
    for sample_dir in sorted(job_dir.glob("seed-*_sample-*")):
        cif_path = model_cif_in(sample_dir, job_name)
        if cif_path is None:
            continue
        found.append((int(sample_dir.name.rsplit("sample-", 1)[1]), cif_path))
    if found:
        return sorted(found)

    # No per-sample directories: some AF3 versions write a single model at the
    # top level. One sample means no ranking to do, which best_sample handles.
    top_level = model_cif_in(job_dir, job_name)
    return [(0, top_level)] if top_level is not None else []


def ranking_scores(stage: Path, job_name: str) -> Dict[int, float]:
    import csv
    path = jp.af3_ranking_path(stage, job_name)
    if not path.is_file():
        candidates = sorted(jp.af3_output_dir(stage, job_name).glob("*ranking_scores.csv"))
        if not candidates:
            return {}
        path = candidates[0]
    scores: Dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                scores[int(row["sample"])] = float(row["ranking_score"])
            except (KeyError, TypeError, ValueError):
                continue
    return scores


def best_sample(stage: Path, job_name: str) -> Tuple[Path, Optional[float]]:
    """AF3's own top-ranked sample, or sample 0 if the ranking file is unusable."""
    job_dir = jp.af3_output_dir(stage, job_name)
    samples = ranked_samples(stage, job_name)
    if not samples:
        raise ScoreError(
            f"no model cif under {job_dir} -- it contains: {describe(job_dir)}"
        )
    scores = ranking_scores(stage, job_name)
    scored = [(index, path) for index, path in samples if index in scores]
    if not scored:
        _log(f"  [warn] {job_name}: no usable ranking_scores.csv, using sample-{samples[0][0]}")
        return samples[0][1], None
    index, path = max(scored, key=lambda entry: scores[entry[0]])
    return path, scores[index]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def discover(stage: Path, experiment: str) -> List[Tuple[str, str, str]]:
    """(group, protein_id, sequence_id) for everything stage 03 handed over."""
    found: List[Tuple[str, str, str]] = []
    directory = jp.inputs_root(stage) / experiment
    if not directory.is_dir():
        return found
    for archive_path in sorted(directory.glob("*.tar.gz")):
        group_key = archive_path.name[: -len(".tar.gz")]
        with tarfile.open(archive_path, "r:gz") as archive:
            for member_name in sorted(archive.getnames()):
                if "/" not in member_name or not member_name.endswith(".fa"):
                    continue
                protein_id, filename = member_name.split("/", 1)
                found.append((group_key, protein_id, filename[: -len(".fa")]))
    return found


def reference_text(stage: Path, experiment: str, group_key: str,
                   protein_id: str) -> str:
    """The complex the sequence was designed on, as pdb text.

    Read from the input archive first, because stage 03 sends the complex along
    with the designs precisely so that this does not have to reach across into
    another stage's working directory -- which breaks the moment stage 03 is
    cleaned. The old location is still tried, so archives written before the
    complex travelled can still be scored.
    """
    archive_path = jp.input_archive_path(stage, experiment, group_key)
    if archive_path.is_file():
        wanted = f"{protein_id}/{protein_id}.pdb"
        with tarfile.open(archive_path, "r:gz") as archive:
            for name in archive.getnames():
                if (name[2:] if name.startswith("./") else name) == wanted:
                    handle = archive.extractfile(name)
                    if handle is not None:
                        return handle.read().decode("utf-8")

    legacy = jp.prepared_dir(jp.stage03_root(stage), experiment, group_key) / f"{protein_id}.pdb"
    if legacy.is_file():
        return legacy.read_text(encoding="utf-8")
    raise ScoreError(
        f"no reference complex for {protein_id}: not in {archive_path.name} as "
        f"{protein_id}/{protein_id}.pdb, and not at {legacy}"
    )


def passthrough_member(stage: Path, experiment: str, group_key: str,
                       member: str) -> Optional[bytes]:
    """One member of the input archive, verbatim, or None if it is not there.

    For the small provenance files that stage 03 sends and stage 05 needs, and
    that this stage only has to hand on without reading.
    """
    archive_path = jp.input_archive_path(stage, experiment, group_key)
    if not archive_path.is_file():
        return None
    with tarfile.open(archive_path, "r:gz") as archive:
        for name in archive.getnames():
            if (name[2:] if name.startswith("./") else name) == member:
                handle = archive.extractfile(name)
                if handle is not None:
                    return handle.read()
    return None


def reference_chain(stage: Path, experiment: str, group_key: str,
                    protein_id: str,
                    cache: Optional[Dict[Tuple[str, str], Chain]] = None) -> Chain:
    key = (group_key, protein_id)
    if cache is not None and key in cache:
        return cache[key]
    text = reference_text(stage, experiment, group_key, protein_id)
    chains = protein_chains(chains_from_pdb(text))
    if not chains:
        raise ScoreError(f"{protein_id}: no protein chain in the reference complex")
    # The design is a tied homodimer, so both chains carry the same sequence;
    # the first is the monomer's counterpart. Not hardcoded to 'A' because
    # nothing upstream guarantees what the chains are called. The fibril is not
    # among them: every one of its residues maps to 'X', which protein_chains
    # excludes.
    if cache is not None:
        cache[key] = chains[0]
    return chains[0]


def predicted_chain(cif_path: Path) -> Chain:
    chains = protein_chains(chains_from_mmcif(cif_path.read_text(encoding="utf-8")))
    if len(chains) != 1:
        raise ScoreError(f"{cif_path.name}: expected one protein chain, found {len(chains)}")
    return chains[0]


def score_experiment(stage: Path, experiment: str, report: ScoreReport,
                     rmsd_threshold: float, fraction_threshold: float,
                     min_site_contact: float = MIN_SITE_CONTACT) -> None:
    table_path = jp.results_table_path(stage, experiment)
    by_id = load_table(table_path, key="sequence_id")
    before = len(by_id)

    records = read_records(jp.fixed_residues_path(jp.stage01_root(stage), experiment))
    pending: Dict[Tuple[str, str], List[Tuple[tarfile.TarInfo, bytes]]] = {}
    # One complex serves all of its designs; re-opening the archive per sequence
    # would read the same member three times for nothing.
    references: Dict[Tuple[str, str], Chain] = {}

    for group_key, protein_id, sequence_id in discover(stage, experiment):
        if sequence_id in by_id:
            report.already += 1
            continue
        job_name = jp.af3_job_name(sequence_id, jp.LIGAND_SUFFIX)
        if not jp.af3_output_dir(stage, job_name).is_dir():
            report.not_folded += 1
            continue

        try:
            cif_path, score = best_sample(stage, job_name)
            moving = predicted_chain(cif_path)
            target = reference_chain(stage, experiment, group_key, protein_id,
                                     references)
            fit, pairs = matchmaker(moving, target)

            site_value: Optional[float] = None
            site_count = 0
            contacts: Optional[Contacts] = None
            mapped = records.get(protein_id)
            if mapped:
                positions = generated_positions(mapped, [target.chain_id])[target.chain_id]
                selected = site_pairs(pairs, target, positions)
                site_value, site_count = site_deviation(fit, selected, moving, target)
                # The prediction's own numbering for the same residues: AF3
                # renumbers from 1, so the seed's numbers would select nothing.
                contacts = contact_report(
                    cif_path, [moving.residue_numbers[i] for i, _ in selected]
                )
            if site_value is None:
                report.no_site += 1
        except (ScoreError, StructureError, OSError, ValueError, KeyError) as exc:
            report.problems.append(f"{experiment}/{sequence_id}: {exc}")
            _log(f"[score] {sequence_id}: FAILED ({exc})")
            continue

        passed, on_fibre = verdict(fit.rmsd, fit.matched_fraction, contacts,
                                   rmsd_threshold, fraction_threshold,
                                   min_site_contact)
        status = "PASSED" if passed else "REJECTED"
        report.evaluated += 1
        report.passed += int(passed)
        report.rejected += int(not passed)
        if not on_fibre:
            report.off_fibre += 1

        by_id[sequence_id] = {
            "sequence_id": sequence_id,
            "protein_id": protein_id,
            "experiment_name": experiment,
            "group": group_key,
            "time_stamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rmsd": f"{fit.rmsd:.4f}",
            "matched_pairs": str(fit.matched_pairs),
            "total_aligned_pairs": str(fit.total_pairs),
            "matched_fraction": f"{fit.matched_fraction:.4f}",
            "ranking_score": "" if score is None else f"{score:.4f}",
            "site_rmsd": "" if site_value is None else f"{site_value:.4f}",
            "site_residues": str(site_count),
            "site_contacts": "" if contacts is None else str(contacts.contacting),
            "site_contact_fraction": "" if contacts is None else f"{contacts.fraction:.4f}",
            "site_closest": ("" if contacts is None or contacts.site_closest is None
                             else f"{contacts.site_closest:.2f}"),
            "chain_closest": ("" if contacts is None or contacts.chain_closest is None
                              else f"{contacts.chain_closest:.2f}"),
            "status": status,
        }
        _log(f"[score] {sequence_id}: rmsd={fit.rmsd:.3f} "
             f"matched={fit.matched_pairs}/{fit.total_pairs} "
             f"({fit.matched_fraction:.1%})"
             + (f" site={site_value:.3f} over {site_count}" if site_value is not None else "")
             + (f" on-fibre={contacts.contacting}/{contacts.measured} "
                f"({contacts.fraction:.0%}) closest={contacts.chain_closest:.1f}A"
                if contacts is not None and contacts.chain_closest is not None else "")
             + f" -> {status}")

        outcome = "passed" if passed else "rejected"
        members = pending.setdefault((outcome, group_key), [])
        collect_job(stage, job_name, protein_id, members)

    # The reference complex travels with the passed folds. Stage 05 superposes
    # each fold onto its two protein chains and takes its fibril -- the ligand
    # AF3 predicted is discarded, because the fibre's geometry is the design's,
    # not the prediction's. Sending it here means stage 05 never has to reach
    # back into stage 04's inputs, which cleanup empties.
    for (outcome, group_key), members in sorted(pending.items()):
        if outcome != "passed" or not members:
            continue
        for protein_id in sorted({info.name.split("/")[0] for info, _ in members}):
            name = f"{protein_id}/{protein_id}.pdb"
            if not any(info.name == name for info, _ in members):
                try:
                    payload = reference_text(
                        stage, experiment, group_key, protein_id).encode("utf-8")
                except ScoreError as exc:
                    _log(f"[sort] {protein_id}: reference not carried ({exc})")
                    continue
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                members.append((info, payload))

            # The redesign spec goes too: it names the scaffolded residues, and
            # stage 05 superposes on exactly those.
            spec_name = f"{protein_id}/{protein_id}_redesign.json"
            if not any(info.name == spec_name for info, _ in members):
                payload = passthrough_member(stage, experiment, group_key, spec_name)
                if payload is not None:
                    info = tarfile.TarInfo(spec_name)
                    info.size = len(payload)
                    members.append((info, payload))

    for (outcome, group_key), members in sorted(pending.items()):
        if not members:
            continue
        archive_path = jp.sorted_archive_path(stage, experiment, outcome, group_key)
        existing: set = set()
        if archive_path.is_file():
            with tarfile.open(archive_path, "r:gz") as archive:
                existing = set(archive.getnames())
        fresh = [(info, payload) for info, payload in members if info.name not in existing]
        if not fresh:
            continue
        added, total = merge_members_into_archive(archive_path, fresh)
        _log(f"[sort] {outcome}/{group_key}: +{added} file(s), {total} total "
             f"-> {archive_path}")

    if len(by_id) != before:
        save_table(by_id, table_path, RESULT_FIELDS,
                   lambda row: (row.get("protein_id", ""), row.get("sequence_id", "")))
        report.tables.append(table_path)


def collect_job(stage: Path, job_name: str, protein_id: str,
                members: List[Tuple[tarfile.TarInfo, bytes]]) -> None:
    """Everything AF3 wrote for one job, nested <protein_id>/<job_name>/..."""
    job_dir = jp.af3_output_dir(stage, job_name)
    for file_path in sorted(job_dir.rglob("*")):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(job_dir.parent).as_posix()
        payload = file_path.read_bytes()
        info = tarfile.TarInfo(f"{protein_id}/{relative}")
        info.size = len(payload)
        members.append((info, payload))


def run_scoring(stage: Path, experiment: Optional[str] = None,
                rmsd_threshold: float = RMSD_THRESHOLD,
                fraction_threshold: float = FRACTION_THRESHOLD,
                min_site_contact: float = MIN_SITE_CONTACT) -> ScoreReport:
    report = ScoreReport()
    names = [experiment] if experiment else jp.inputs_experiments(stage)
    if not names:
        _log(f"[score] nothing handed over under {jp.inputs_root(stage)}")
        return report
    for experiment_name in names:
        score_experiment(stage, experiment_name, report, rmsd_threshold,
                         fraction_threshold, min_site_contact)
    if report.no_site:
        _log(f"[score] {report.no_site} sequence(s) have no site_rmsd: stage 01 recorded "
             f"no fixed residues for them, so there was nothing to measure")
    # Said every run, not only when it bites: a gate silently set to zero looks
    # exactly like a gate that nothing failed.
    if report.evaluated:
        if min_site_contact > 0:
            _log(f"[score] site contact gate at {min_site_contact:.0%} within "
                 f"{SITE_CONTACT_CUTOFF:.1f} A: {report.off_fibre} rejected on it")
        else:
            _log(f"[score] site contact RECORDED but NOT gated (--min-site-contact 0). "
                 f"The site_contacts / site_contact_fraction / chain_closest columns "
                 f"are filled in; run site_contacts.py to choose a threshold.")
    return report


def test_gate() -> None:
    """Every branch of the pass decision, including the ones that cost GPU time.

    Written because each of these has a way of being wrong that looks like
    success: a gate left at zero looks like a gate nothing failed, a gate set
    above what the seeds support looks like a stage that ran and found nothing,
    and an off-by-one on >= rejects exactly the folds that reproduce the design.
    """
    good = Contacts(contacting=8, measured=10, site_closest=3.1, chain_closest=2.8)
    poor = Contacts(contacting=1, measured=10, site_closest=9.4, chain_closest=8.9)

    # the fold is good and on the fibre
    assert verdict(1.9, 0.95, good, 2.5, 0.8, 0.8)[0]
    # the same fold, folded off the fibre: rejected on contacts alone
    passed, on_fibre = verdict(1.9, 0.95, poor, 2.5, 0.8, 0.8)
    assert not passed and not on_fibre
    # with the gate off it passes again, which is what "record only" means
    assert verdict(1.9, 0.95, poor, 2.5, 0.8, 0.0)[0]
    # boundary: a prediction that matches the gate exactly is kept, not dropped
    assert verdict(1.9, 0.95, good, 2.5, 0.8, 0.8)[0], "0.8 >= 0.8 must pass"
    assert not verdict(1.9, 0.95, good, 2.5, 0.8, 0.81)[0]
    # contacts unknown: judged on the rest rather than failed for a missing record
    assert verdict(1.9, 0.95, None, 2.5, 0.8, 1.0)[0]
    # the RMSD threshold still bites, and 2.5 is the threshold
    assert not verdict(2.6, 0.95, good, 2.5, 0.8, 0.8)[0]
    assert verdict(2.49, 0.95, good, 2.5, 0.8, 0.8)[0]
    # and so does matched_fraction
    assert not verdict(1.9, 0.5, good, 2.5, 0.8, 0.8)[0]

    assert RMSD_THRESHOLD == 2.5, RMSD_THRESHOLD
    assert SITE_CONTACT_CUTOFF == 6.0, SITE_CONTACT_CUTOFF
    print(f"[self-test] rmsd < {RMSD_THRESHOLD}, matched > {FRACTION_THRESHOLD}, "
          f"site contact within {SITE_CONTACT_CUTOFF} A; every branch ok")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--max-rmsd", type=float, default=RMSD_THRESHOLD)
    parser.add_argument("--min-fraction", type=float, default=FRACTION_THRESHOLD)
    parser.add_argument("--min-site-contact", type=float, default=MIN_SITE_CONTACT,
                        help=f"reject a fold unless this fraction of the seed's "
                             f"ALL-fixed residues is still within "
                             f"{SITE_CONTACT_CUTOFF:.1f} A of the fibre "
                             f"(default: {MIN_SITE_CONTACT}, meaning record only). "
                             f"site_contacts.py measures what these seeds support")
    parser.add_argument("--self-test", action="store_true",
                        help="check the pass decision against its own branches "
                             "and exit, touching nothing on disk")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        test_gate()
        return 0
    report = run_scoring(args.stage.resolve(), args.experiment,
                         args.max_rmsd, args.min_fraction, args.min_site_contact)
    _log(f"[done] {report.summary()}")
    for table in report.tables:
        _log(f"[done] results -> {table}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

