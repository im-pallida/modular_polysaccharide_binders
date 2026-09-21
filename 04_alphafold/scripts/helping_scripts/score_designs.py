#!/usr/bin/env python3
"""
Stage 04, step 3: score every folded monomer and sort it passed or rejected.

    outputs/<sequence_id>_monomer/                      <- AF3 wrote
    ../03_protein_mpnn/inputs_prepared/<experiment>/<group>/<protein_id>.pdb
                                                        <- the reference
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

The reference is read straight out of stage 03's inputs_prepared/, the exact
backbone ProteinMPNN designed the sequence onto. No copy is kept here: the old
transfer step duplicated those files into inputs/reference/, and the duplicate
went stale the moment stage 03 changed its layout.

site_rmsd is a diagnostic, never a gate. After superposing on the pruned core,
it measures the deviation at the scaffolded residues alone -- the "ALL" fixed
side chains stage 01 recorded. A design can have an excellent whole-chain RMSD
while the binding site itself has drifted, and that column is where you would
see it. Blank when stage 01 recorded no fixed residues for that structure.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import job_paths as jp  # noqa: E402
from archives import load_table, merge_members_into_archive, save_table  # noqa: E402
from designs import generated_positions, read_records  # noqa: E402

STAGE = Path(__file__).resolve().parents[2]

RMSD_THRESHOLD = 5.0
FRACTION_THRESHOLD = 0.8
PRUNE_CUTOFF = 2.0
MAX_ITERATIONS = 5

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

RESULT_FIELDS = [
    "sequence_id", "protein_id", "experiment_name", "group", "time_stamp",
    "rmsd", "matched_pairs", "total_aligned_pairs", "matched_fraction",
    "ranking_score", "site_rmsd", "site_residues", "status",
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


def site_deviation(fit: Fit, pairs: Sequence[Tuple[int, int]], moving: Chain,
                   target: Chain, site_residues: Sequence[int]) -> Tuple[Optional[float], int]:
    """RMSD at the scaffolded residues under the whole-chain superposition.

    Not its own fit: superposing five atoms on five atoms would fit almost
    anything. The question is where the site landed once the fold as a whole
    was aligned.
    """
    wanted = set(site_residues)
    target_indices = {index for index, number in enumerate(target.residue_numbers)
                      if number in wanted}
    selected = [(i, j) for i, j in pairs if j in target_indices]
    if not selected:
        return None, 0
    moved = fit.apply(np.array([moving.coords[i] for i, _ in selected]))
    reference = np.array([target.coords[j] for _, j in selected])
    deviation = moved - reference
    return float(np.sqrt((deviation ** 2).sum(axis=1).mean())), len(selected)


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


def reference_chain(stage: Path, experiment: str, group_key: str,
                    protein_id: str) -> Chain:
    path = jp.prepared_dir(jp.stage03_root(stage), experiment, group_key) / f"{protein_id}.pdb"
    if not path.is_file():
        raise ScoreError(f"no reference backbone at {path}")
    chains = protein_chains(chains_from_pdb(path.read_text(encoding="utf-8")))
    if not chains:
        raise ScoreError(f"{path.name}: no protein chain found")
    # The design is a tied homodimer, so both chains carry the same sequence;
    # the first is the monomer's counterpart. Not hardcoded to 'A' because
    # nothing upstream guarantees what the chains are called.
    return chains[0]


def predicted_chain(cif_path: Path) -> Chain:
    chains = protein_chains(chains_from_mmcif(cif_path.read_text(encoding="utf-8")))
    if len(chains) != 1:
        raise ScoreError(f"{cif_path.name}: expected one protein chain, found {len(chains)}")
    return chains[0]


def score_experiment(stage: Path, experiment: str, report: ScoreReport,
                     rmsd_threshold: float, fraction_threshold: float) -> None:
    table_path = jp.results_table_path(stage, experiment)
    by_id = load_table(table_path, key="sequence_id")
    before = len(by_id)

    records = read_records(jp.fixed_residues_path(jp.stage01_root(stage), experiment))
    pending: Dict[Tuple[str, str], List[Tuple[tarfile.TarInfo, bytes]]] = {}

    for group_key, protein_id, sequence_id in discover(stage, experiment):
        if sequence_id in by_id:
            report.already += 1
            continue
        job_name = jp.af3_job_name(sequence_id)
        if not jp.af3_output_dir(stage, job_name).is_dir():
            report.not_folded += 1
            continue

        try:
            cif_path, score = best_sample(stage, job_name)
            moving = predicted_chain(cif_path)
            target = reference_chain(stage, experiment, group_key, protein_id)
            fit, pairs = matchmaker(moving, target)

            site_value: Optional[float] = None
            site_count = 0
            mapped = records.get(protein_id)
            if mapped:
                positions = generated_positions(mapped, [target.chain_id])[target.chain_id]
                site_value, site_count = site_deviation(fit, pairs, moving, target, positions)
            if site_value is None:
                report.no_site += 1
        except (ScoreError, OSError, ValueError, KeyError) as exc:
            report.problems.append(f"{experiment}/{sequence_id}: {exc}")
            _log(f"[score] {sequence_id}: FAILED ({exc})")
            continue

        passed = fit.rmsd < rmsd_threshold and fit.matched_fraction > fraction_threshold
        status = "PASSED" if passed else "REJECTED"
        report.evaluated += 1
        report.passed += int(passed)
        report.rejected += int(not passed)

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
            "status": status,
        }
        _log(f"[score] {sequence_id}: rmsd={fit.rmsd:.3f} "
             f"matched={fit.matched_pairs}/{fit.total_pairs} "
             f"({fit.matched_fraction:.1%})"
             + (f" site={site_value:.3f} over {site_count}" if site_value is not None else "")
             + f" -> {status}")

        outcome = "passed" if passed else "rejected"
        members = pending.setdefault((outcome, group_key), [])
        collect_job(stage, job_name, protein_id, members)

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
                fraction_threshold: float = FRACTION_THRESHOLD) -> ScoreReport:
    report = ScoreReport()
    names = [experiment] if experiment else jp.inputs_experiments(stage)
    if not names:
        _log(f"[score] nothing handed over under {jp.inputs_root(stage)}")
        return report
    for experiment_name in names:
        score_experiment(stage, experiment_name, report, rmsd_threshold, fraction_threshold)
    if report.no_site:
        _log(f"[score] {report.no_site} sequence(s) have no site_rmsd: stage 01 recorded "
             f"no fixed residues for them, so there was nothing to measure")
    return report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--max-rmsd", type=float, default=RMSD_THRESHOLD)
    parser.add_argument("--min-fraction", type=float, default=FRACTION_THRESHOLD)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    report = run_scoring(args.stage.resolve(), args.experiment,
                         args.max_rmsd, args.min_fraction)
    _log(f"[done] {report.summary()}")
    for table in report.tables:
        _log(f"[done] results -> {table}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
