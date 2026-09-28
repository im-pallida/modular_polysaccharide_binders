#!/usr/bin/env python3
"""
Stage 08, step 3: measure each folded construct against the pair it was made from.

    outputs/<sequence_id>_holo/                     <- AF3 wrote, with cellulose
    ../06_.../inputs_prepared/<exp>/<group>/<sid>.pdb   <- the two copies, unlinked
    sorted_clean/<experiment>/<passed|rejected>/<group>.tar.gz
    tables/stage_08_results_<experiment>.csv   <- every measurement, both verdicts
    tables/stage_08_passed_<experiment>.csv    <- the deliverable: id, unit, linker

THE QUESTION

Fusing two copies can pull them out of position. The reference is therefore not
the construct RFD3 built -- that would ask whether AlphaFold reproduced RFD3's
guess -- but the two copies AS THEY STOOD BEFORE THE LINKER EXISTED: the aligned
pair from stage 05, each copy on its own seed side, in the arrangement the whole
pipeline was built around. A construct that folds beautifully but has swung its
lobes apart has lost the thing it was for.

WHAT IS MEASURED, AND WHAT IS NOT

    the linker      excluded. It has no counterpart in the reference -- it did
                    not exist there -- and it is the part allowed to be new.
    the cellulose   excluded. Every polysaccharide residue maps to 'X', which
                    protein_chains() drops, so AF3's predicted fibre can be
                    present in the fold without entering the metric.
    everything else CA atoms, superposed and measured.

ONE SUPERPOSITION, NOT TWO

This is the part that makes the number mean what it should. The obvious
implementation fits each copy onto its counterpart separately and reports the
average, and that number is nearly blind to the failure this stage exists to
catch: two lobes can each be perfect in isolation while the angle between them
has opened by thirty degrees, and per-lobe fitting reports two excellent RMSDs
and averages them into a pass.

So both copies are treated as ONE rigid body. A single Kabsch fit over every
non-linker CA in the construct, against the same atoms in the reference, and one
deviation. A hinge between the lobes has nowhere to hide in that number.

A construct passes when rmsd < RMSD_THRESHOLD over at least FRACTION of the
non-linker residues. Of AF3's samples, only the one AF3 ranks first is scored --
that is the prediction you would use, so the verdict should be about it.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tarfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import job_paths as jp  # noqa: E402
from archives import load_table, merge_members_into_archive, save_table  # noqa: E402

STAGE = Path(__file__).resolve().parents[2]

RMSD_THRESHOLD = 5.0
FRACTION_THRESHOLD = 0.8

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

RESULT_FIELDS = [
    "sequence_id", "design_name", "experiment_name", "group", "time_stamp",
    "rmsd", "matched_residues", "reference_residues", "matched_fraction",
    "ranking_score", "linker_length", "linker_sequence", "unit_sequence",
    "status",
]

# The deliverable, beside the results table. The two copies are identical --
# stage 07 ties them -- so one unit column says everything two would.
PASSED_FIELDS = ["protein_id", "unit_sequence", "linker_sequence"]


class ScoreError(Exception):
    """One construct could not be scored."""


def _log(*parts: object) -> None:
    print(*parts, flush=True)


@dataclass
class ScoreReport:
    evaluated: int = 0
    passed: int = 0
    rejected: int = 0
    not_folded: int = 0
    already: int = 0
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
# Reading structures, CA only
# ---------------------------------------------------------------------------

@dataclass
class Chain:
    """One chain's CA atoms in residue order, with the residue numbers."""
    chain_id: str
    numbers: List[int]
    sequence: str
    coords: np.ndarray


def parse_atom_site(cif_text: str) -> List[Dict[str, str]]:
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


def _assemble(by_chain) -> List[Chain]:
    out: List[Chain] = []
    for chain_id, residues in sorted(by_chain.items()):
        ordered = sorted(residues.items())
        out.append(Chain(
            chain_id=chain_id,
            numbers=[number for number, _ in ordered],
            sequence="".join(AA3TO1.get(name, "X") for _, (name, _) in ordered),
            coords=(np.array([xyz for _, (_, xyz) in ordered], dtype=float)
                    if ordered else np.zeros((0, 3))),
        ))
    return out


def chains_from_mmcif(text: str) -> List[Chain]:
    by_chain: Dict[str, Dict[int, Tuple[str, Tuple[float, float, float]]]] = {}
    for row in parse_atom_site(text):
        if row.get("label_atom_id") != "CA":
            continue
        chain_id = row.get("auth_asym_id") or row.get("label_asym_id")
        if chain_id is None:
            continue
        by_chain.setdefault(chain_id, {})[int(row["auth_seq_id"])] = (
            row.get("label_comp_id", "UNK"),
            (float(row["Cartn_x"]), float(row["Cartn_y"]), float(row["Cartn_z"])),
        )
    return _assemble(by_chain)


def chains_from_pdb(text: str) -> List[Chain]:
    by_chain: Dict[str, Dict[int, Tuple[str, Tuple[float, float, float]]]] = {}
    for line in text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if line[12:16].strip() != "CA":
            continue
        chain_id = line[21].strip() or " "
        by_chain.setdefault(chain_id, {})[int(line[22:26])] = (
            line[17:20].strip(),
            (float(line[30:38]), float(line[38:46]), float(line[46:54])),
        )
    return _assemble(by_chain)


def protein_chains(chains: Sequence[Chain], minimum: int = 10) -> List[Chain]:
    """Chains long enough to be protein. Length rather than residue type because
    this reads text: a polysaccharide chain is a handful of residues all mapping
    to 'X', so the cellulose drops out here and never enters the metric."""
    return [chain for chain in chains
            if len(chain.sequence) >= minimum
            and chain.sequence.count("X") < len(chain.sequence)]


# ---------------------------------------------------------------------------
# One rigid body, one fit
# ---------------------------------------------------------------------------

def kabsch(moving: np.ndarray, target: np.ndarray) -> float:
    """RMSD after optimal superposition. No reflections."""
    moving_center = moving.mean(axis=0)
    target_center = target.mean(axis=0)
    a = moving - moving_center
    b = target - target_center
    u, _, vt = np.linalg.svd(a.T @ b)
    # An unguarded SVD can return a reflection, which superposes a mirror image
    # and reports a flatteringly low rmsd.
    determinant = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, determinant]) @ u.T
    deviation = a @ rotation.T - b
    return float(np.sqrt((deviation ** 2).sum(axis=1).mean()))


def reference_units(pair: Sequence[Chain]) -> List[Tuple[str, int]]:
    """(chain, residue number) for every CA of the unlinked pair, in order."""
    return [(chain.chain_id, number)
            for chain in pair for number in chain.numbers]


def construct_positions(construct: Chain, reference_total: int,
                        linker: Sequence[int]) -> List[int]:
    """The construct's non-linker residue numbers, in order.

    The linker is named by its residue numbers in the construct, which stage 07
    recorded from RFD3's own index map. Everything else is a copy residue, and
    the count must equal the reference's or the two are not the same molecule.
    """
    excluded = set(linker)
    kept = [number for number in construct.numbers if number not in excluded]
    if len(kept) != reference_total:
        raise ScoreError(
            f"the construct has {len(kept)} non-linker residue(s) but the "
            f"unlinked pair has {reference_total}; they are not the same two "
            f"copies, so there is nothing meaningful to superpose"
        )
    return kept


def whole_body_rmsd(construct: Chain, kept: Sequence[int],
                    pair: Sequence[Chain]) -> Tuple[float, int]:
    """(rmsd, atoms used) over both copies fitted as a single rigid body.

    Positional correspondence, not sequence alignment: the construct is the two
    copies in order with a linker spliced in, so removing the linker leaves the
    reference's residues in the reference's order. Aligning by sequence instead
    would be free to slide one copy against the other and hide exactly the
    displacement being measured.
    """
    index = {number: position for position, number in enumerate(construct.numbers)}
    moving = np.array([construct.coords[index[number]] for number in kept])
    target = np.vstack([chain.coords for chain in pair if chain.coords.size])
    if len(moving) != len(target):
        raise ScoreError(
            f"{len(moving)} construct atoms against {len(target)} reference atoms"
        )
    if len(moving) < 3:
        raise ScoreError(f"only {len(moving)} atom(s) to superpose")
    return kabsch(moving, target), len(moving)


def per_lobe_rmsd(construct: Chain, kept: Sequence[int],
                  pair: Sequence[Chain]) -> List[float]:
    """Each copy fitted on its own. Recorded, never gated.

    Here to be compared against the whole-body number rather than used instead
    of it: when the two lobes are individually good and the construct as a whole
    is not, the gap between these columns is the hinge.
    """
    index = {number: position for position, number in enumerate(construct.numbers)}
    out: List[float] = []
    cursor = 0
    for chain in pair:
        count = len(chain.numbers)
        slice_numbers = kept[cursor:cursor + count]
        cursor += count
        if len(slice_numbers) != count or count < 3:
            continue
        moving = np.array([construct.coords[index[number]] for number in slice_numbers])
        out.append(kabsch(moving, chain.coords))
    return out


# ---------------------------------------------------------------------------
# AF3 output discovery
# ---------------------------------------------------------------------------

def model_cif_in(directory: Path, job_name: str) -> Optional[Path]:
    """The model mmCIF in one directory, whichever way AF3 named it."""
    for candidate in (directory / f"{job_name}_{directory.name}_model.cif",
                      directory / "model.cif",
                      directory / f"{job_name}_model.cif"):
        if candidate.is_file():
            return candidate
    remaining = sorted(path for path in directory.glob("*model.cif") if path.is_file())
    return remaining[0] if remaining else None


def ranking_scores(stage: Path, job_name: str) -> Dict[int, float]:
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
    found: List[Tuple[int, Path]] = []
    for sample_dir in sorted(job_dir.glob("seed-*_sample-*")):
        cif_path = model_cif_in(sample_dir, job_name)
        if cif_path is not None:
            found.append((int(sample_dir.name.rsplit("sample-", 1)[1]), cif_path))
    if not found:
        top_level = model_cif_in(job_dir, job_name)
        if top_level is None:
            raise ScoreError(f"no model cif under {job_dir}")
        return top_level, None
    found.sort()
    scores = ranking_scores(stage, job_name)
    scored = [(index, path) for index, path in found if index in scores]
    if not scored:
        return found[0][1], None
    index, path = max(scored, key=lambda entry: scores[entry[0]])
    return path, scores[index]


# ---------------------------------------------------------------------------
# What stage 07 handed over
# ---------------------------------------------------------------------------

def linker_records(stage: Path, experiment: str, group_key: str) -> Dict[str, List[int]]:
    """{design name: linker residue numbers}, from stage 07's own record.

    Read rather than recomputed. Stage 07 identified the linker from RFD3's
    diffused_index_map and wrote it down; deriving it a second time here, by a
    second method, is how the two quietly disagree.
    """
    path = jp.linker_spec_path(jp.stage07_root(stage), experiment, group_key)
    found: Dict[str, List[int]] = {}
    if not path.is_file():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        numbers = []
        for key in payload.get("linker", []):
            digits = "".join(character for character in str(key) if character.isdigit())
            if digits:
                numbers.append(int(digits))
        if payload.get("id"):
            found[str(payload["id"])] = sorted(numbers)
    return found


def split_sequence(construct: Chain, linker: Sequence[int]) -> Tuple[str, str]:
    """(the linker's sequence, the unit's sequence).

    ONE unit, because the two copies are identical: stage 07 mirrors the
    linker's 8 A shell into both copies and ties each position to its twin, so
    what comes back is one designed unit written twice.

    That is CHECKED here, not assumed. Stage 07's pre-flight proves the tying
    on one construct per group; this is the only place every construct is
    looked at. A construct whose halves differ is not two copies of one binder
    -- reporting its first half as "the unit" would put a sequence in the table
    that only half the protein has -- so it is refused and lands in the
    problems list with the fold left on disk.
    """
    excluded = set(linker)
    letters = dict(zip(construct.numbers, construct.sequence))
    linker_seq = "".join(letters[number] for number in construct.numbers
                         if number in excluded)
    if not excluded:
        return linker_seq, ""

    first, last = min(excluded), max(excluded)
    unit_a = "".join(letters[number] for number in construct.numbers
                     if number < first)
    unit_b = "".join(letters[number] for number in construct.numbers
                     if number > last)
    if len(unit_a) != len(unit_b):
        raise ScoreError(
            f"the copies flanking the linker are {len(unit_a)} and "
            f"{len(unit_b)} residues, so this is not two copies of one unit"
        )
    if unit_a != unit_b:
        differ = [index for index in range(len(unit_a))
                  if unit_a[index] != unit_b[index]]
        shown = ", ".join(f"{index + 1}: {unit_a[index]} vs {unit_b[index]}"
                          for index in differ[:5])
        raise ScoreError(
            f"the two units differ at {len(differ)} position(s) ({shown}), so "
            f"stage 07's tying did not hold for this construct and there is no "
            f"single unit sequence to record"
        )
    return linker_seq, unit_a


def save_passed_table(by_id: Dict[str, dict], table_path: Path) -> Optional[Path]:
    """The round's deliverable: one row per construct that passed.

    Three columns. Which construct it is, the unit it is two copies of, and the
    linker holding them together -- that is the design. Every measurement
    behind the verdict, and every rejection, stays in the results table beside
    it; nothing is lost by keeping this one narrow.
    """
    rows = {
        row["sequence_id"]: {
            "protein_id": row["sequence_id"],
            "unit_sequence": row.get("unit_sequence", ""),
            "linker_sequence": row.get("linker_sequence", ""),
        }
        for row in by_id.values() if row.get("status") == "PASSED"
    }
    if not rows:
        return None
    save_table(rows, table_path, PASSED_FIELDS, lambda row: (row["protein_id"],))
    return table_path


def score_experiment(stage: Path, experiment: str, report: ScoreReport,
                     rmsd_threshold: float, fraction_threshold: float) -> None:
    table_path = jp.results_table_path(stage, experiment)
    by_id = load_table(table_path, key="sequence_id")
    before = len(by_id)
    pending: Dict[Tuple[str, str], List[Tuple[tarfile.TarInfo, bytes]]] = {}

    directory = jp.inputs_root(stage) / experiment
    if not directory.is_dir():
        return

    for archive_path in sorted(directory.glob("*.tar.gz")):
        group_key = archive_path.name[: -len(".tar.gz")]
        linkers = linker_records(stage, experiment, group_key)
        references: Dict[str, List[Chain]] = {}

        with tarfile.open(archive_path, "r:gz") as archive:
            sequences: List[Tuple[str, str]] = []
            for name in sorted(archive.getnames()):
                cleaned = name[2:] if name.startswith("./") else name
                parts = cleaned.split("/")
                if len(parts) == 2 and parts[-1].endswith(".fa"):
                    sequences.append((parts[0], Path(parts[-1]).stem))

        for design_name, sequence_id in sequences:
            if sequence_id in by_id:
                report.already += 1
                continue
            job_name = jp.af3_job_name(sequence_id, jp.HOLO_SUFFIX)
            if not jp.af3_output_dir(stage, job_name).is_dir():
                report.not_folded += 1
                continue

            try:
                cif_path, ranking = best_sample(stage, job_name)
                folded = protein_chains(
                    chains_from_mmcif(cif_path.read_text(encoding="utf-8")))
                if len(folded) != 1:
                    raise ScoreError(
                        f"expected one protein chain in the fold, found {len(folded)}")
                construct = folded[0]

                pair_id = jp.sequence_id_from_design(design_name)
                if pair_id not in references:
                    pair_path = jp.construct_pair_path(stage, experiment,
                                                       group_key, pair_id)
                    if not pair_path.is_file():
                        raise ScoreError(
                            f"no unlinked pair at {pair_path}; stage 06 keeps "
                            f"these, so either it has not run for this group or "
                            f"they were cleared by hand"
                        )
                    references[pair_id] = protein_chains(
                        chains_from_pdb(pair_path.read_text(encoding="utf-8")))
                pair = references[pair_id]
                if len(pair) != 2:
                    raise ScoreError(
                        f"the unlinked reference has {len(pair)} protein chain(s), "
                        f"expected the two copies")

                linker = linkers.get(design_name, [])
                if not linker:
                    raise ScoreError(
                        f"stage 07 recorded no linker for {design_name}, so the "
                        f"generated residues cannot be excluded from the metric")

                total = sum(len(chain.numbers) for chain in pair)
                kept = construct_positions(construct, total, linker)
                rmsd, matched = whole_body_rmsd(construct, kept, pair)
                lobes = per_lobe_rmsd(construct, kept, pair)
                linker_seq, unit_seq = split_sequence(construct, linker)
            except (ScoreError, OSError, ValueError, KeyError) as exc:
                report.problems.append(f"{experiment}/{sequence_id}: {exc}")
                _log(f"[score] {sequence_id}: FAILED ({exc})")
                continue

            fraction = matched / total if total else 0.0
            passed = rmsd < rmsd_threshold and fraction > fraction_threshold
            status = "PASSED" if passed else "REJECTED"
            report.evaluated += 1
            report.passed += int(passed)
            report.rejected += int(not passed)

            by_id[sequence_id] = {
                "sequence_id": sequence_id,
                "design_name": design_name,
                "experiment_name": experiment,
                "group": group_key,
                "time_stamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "rmsd": f"{rmsd:.4f}",
                "matched_residues": str(matched),
                "reference_residues": str(total),
                "matched_fraction": f"{fraction:.4f}",
                "ranking_score": "" if ranking is None else f"{ranking:.4f}",
                "linker_length": str(len(linker)),
                "linker_sequence": linker_seq,
                "unit_sequence": unit_seq,
                "status": status,
            }
            lobe_note = (f" lobes {'/'.join(f'{value:.2f}' for value in lobes)}"
                         if lobes else "")
            _log(f"[score] {sequence_id}: rmsd={rmsd:.3f} over {matched} CA"
                 f"{lobe_note} linker={len(linker)} -> {status}")

            # Both outcomes are archived, as at every other stage. A rejected
            # fold is the evidence for its own rejection -- a 7 A RMSD is worth
            # opening, and the linker that caused it is worth looking at before
            # the next round of lengths. It also has to be archived for cleanup
            # to clear it: nothing under outputs/ is deleted until its files
            # are verified inside a sorted_clean tarball, so archiving only the
            # passed folds left every rejected one sitting there as a loose
            # directory, which is what it did.
            outcome = "passed" if passed else "rejected"
            members = pending.setdefault((outcome, group_key), [])
            collect_fold(stage, job_name, sequence_id, members)

    for (outcome, group_key), members in sorted(pending.items()):
        if not members:
            continue
        archive_path = jp.sorted_archive_path(stage, experiment, outcome, group_key)
        existing: set = set()
        if archive_path.is_file():
            with tarfile.open(archive_path, "r:gz") as archive:
                existing = set(archive.getnames())
        fresh = [(info, payload) for info, payload in members
                 if info.name not in existing]
        if fresh:
            added, total_members = merge_members_into_archive(archive_path, fresh)
            _log(f"[sort] {outcome}/{group_key}: +{added} file(s), "
                 f"{total_members} total -> {archive_path}")

    if len(by_id) != before:
        save_table(by_id, table_path, RESULT_FIELDS,
                   lambda row: (row.get("group", ""), row.get("sequence_id", "")))
        report.tables.append(table_path)

    # Rewritten every run, not only when something new was scored, so it
    # appears the first time this stage runs with a results table already on
    # disk. It is derived from that table rather than accumulated alongside
    # it, which is what keeps the two from drifting apart.
    passed_path = save_passed_table(by_id, jp.passed_table_path(stage, experiment))
    if passed_path is not None:
        report.tables.append(passed_path)


def collect_fold(stage: Path, job_name: str, sequence_id: str,
                 members: List[Tuple[tarfile.TarInfo, bytes]]) -> None:
    """Everything AF3 wrote for one construct, cellulose included.

    The fold is archived as it came out, with its ligand: the whole point of
    this stage is the complex, and a structure stripped of the fibre would be
    the one thing nobody wants to open.
    """
    job_dir = jp.af3_output_dir(stage, job_name)
    for file_path in sorted(job_dir.rglob("*")):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(job_dir.parent).as_posix()
        payload = file_path.read_bytes()
        info = tarfile.TarInfo(f"{sequence_id}/{relative}")
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
        score_experiment(stage, experiment_name, report,
                         rmsd_threshold, fraction_threshold)
    return report


def _hinged_pair(degrees: float, radius: float = 12.0, separation: float = 40.0,
                 count: int = 93, seed: int = 0):
    """(reference pair, matching construct, hinged construct).

    Two domains the size of the copies this pipeline actually builds -- about
    93 residues, so roughly a 12 A radius -- separated across where the fibre
    would be, with the second rotated about the midpoint between them.
    """
    rng = np.random.default_rng(seed)
    lobe_a = rng.normal(size=(count, 3)) * radius
    lobe_b = rng.normal(size=(count, 3)) * radius + np.array([separation, 0.0, 0.0])
    pivot = np.array([separation / 2.0, 0.0, 0.0])
    angle = np.deg2rad(degrees)
    spin = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                     [np.sin(angle), np.cos(angle), 0.0],
                     [0.0, 0.0, 1.0]])
    hinged_b = (lobe_b - pivot) @ spin.T + pivot

    reference = [Chain("A", list(range(1, count + 1)), "A" * count, lobe_a),
                 Chain("B", list(range(1, count + 1)), "A" * count, lobe_b)]

    # The linker sits BETWEEN the copies, which is where RFD3 puts it: copy A,
    # then the generated residues, then copy B, renumbered continuously. Putting
    # it at the end would make "the residues before the linker" mean both
    # copies, and the unit sequence would come out twice as long as it is.
    linker_numbers = [count + 1, count + 2]
    numbers = (list(range(1, count + 1)) + linker_numbers
               + list(range(count + 3, 2 * count + 3)))
    middle = rng.normal(size=(2, 3)) + np.array([separation / 2.0, 0.0, 0.0])
    matching = Chain("A", numbers, "A" * (2 * count + 2),
                     np.vstack([lobe_a, middle, lobe_b]))
    hinged = Chain("A", numbers, "A" * (2 * count + 2),
                   np.vstack([lobe_a, middle, hinged_b]))
    return reference, matching, hinged, linker_numbers


def test_whole_body() -> None:
    """A hinge must show up in the number, and per-lobe fitting must not see it.

    This is the assertion the stage exists for. Each lobe is rotated as a rigid
    body, so each still superposes onto its counterpart EXACTLY -- per-lobe
    RMSD stays at zero however far the construct opens. Only the whole-body fit
    moves, because no single rigid transform puts both copies back at once.
    """
    reference, matching, hinged, linker = _hinged_pair(40.0)
    kept = construct_positions(matching, 186, linker)
    assert len(kept) == 186 and not set(kept) & set(linker), kept[:5]

    good, count = whole_body_rmsd(matching, kept, reference)
    assert count == 186 and good < 1e-6, (good, count)

    bad, _ = whole_body_rmsd(hinged, kept, reference)
    lobes = per_lobe_rmsd(hinged, kept, reference)
    assert max(lobes) < 1e-6, f"each lobe should still be exact, got {lobes}"
    assert bad > RMSD_THRESHOLD, (
        f"a 40 degree hinge gave {bad:.2f} A, under the {RMSD_THRESHOLD} A "
        f"threshold -- it would have passed")

    # ... and the response is monotonic, so the threshold means one angle
    # rather than depending on which way the construct happens to open.
    previous = 0.0
    for degrees in (5, 10, 20, 30, 40):
        value, _ = whole_body_rmsd(_hinged_pair(degrees)[2], kept, reference)
        assert value > previous, f"{degrees} deg scored {value}, below {previous}"
        previous = value

    labelled = Chain("A", matching.numbers, "M" * 93 + "GS" + "K" * 93,
                     matching.coords)
    linker_seq, unit = split_sequence(labelled, linker)
    assert linker_seq == "GS", linker_seq
    assert unit == "M" * 93, f"unit is {len(unit)} residues, expected 93"

    print(f"[self-test] a 40 degree hinge scores {bad:.1f} A whole-body while "
          f"each lobe scores {max(lobes):.2f} A; response is monotonic; "
          f"sequences split correctly")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--max-rmsd", type=float, default=RMSD_THRESHOLD)
    parser.add_argument("--min-fraction", type=float, default=FRACTION_THRESHOLD)
    parser.add_argument("--self-test", action="store_true",
                        help="check the whole-body metric against a known hinge "
                             "and exit, touching nothing on disk")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        test_whole_body()
        return 0
    report = run_scoring(args.stage.resolve(), args.experiment,
                         args.max_rmsd, args.min_fraction)
    _log(f"[done] {report.summary()}")
    for table in report.tables:
        _log(f"[done] results -> {table}")
    for problem in report.problems[:10]:
        _log(f"[done]   {problem}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
