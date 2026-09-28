#!/usr/bin/env python3
"""
Stage 04, diagnostic: how close the seeds' fixed residues sit to the fibre.

    inputs/<experiment>/<group>.tar.gz   <- <protein_id>/<protein_id>.pdb
    ../01_.../job_runs/fixed_residues_<experiment>.jsonl

Run this once, before setting --min-site-contact, and read the number off the
bottom of the output.

Stage 04 can gate on whether a prediction kept its binding site on the fibre:
the residues stage 01 fixed "ALL" should still be within SITE_CONTACT_CUTOFF of
the ligand after folding. What that gate should be set to is not a number worth
guessing, because it depends on something only these files know -- whether every
ALL-fixed residue in these particular seeds is a binding residue, or whether
some were fixed for structural reasons and never touch the fibre at all.

So measure it in the seeds, where the answer is by definition correct: the seed
IS the design, and a residue that is not on the fibre there was never going to
be on it in a prediction either. Set the gate at or below what the seeds show,
never above, or nothing can pass.

Nothing is written. This only reads.

Usage:
    ./site_contacts.py --stage ~/1cbh_clear/04_alphafold
    ./site_contacts.py --stage ~/1cbh_clear/04_alphafold --cutoff 5.0 --verbose
"""
from __future__ import annotations

import argparse
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import gemmi

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import job_paths as jp  # noqa: E402
from designs import read_records, split_map_key  # noqa: E402
from structures import (  # noqa: E402
    SITE_CONTACT_CUTOFF,
    StructureError,
    chain_approach,
    classify_chains,
    most_contacted,
    site_approaches,
)

STAGE = Path(__file__).resolve().parents[2]


def _log(*parts: object) -> None:
    print(*parts, flush=True)


@dataclass
class SeedResult:
    protein_id: str
    group: str
    fixed: int           # residues stage 01 fixed "ALL"
    present: int         # of those, how many are in the pdb
    contacting: int      # of those present, how many are within the cutoff
    closest: float       # closest approach over the whole chain
    far: List[Tuple[int, float]] = field(default_factory=list)

    @property
    def fraction(self) -> float:
        return self.contacting / self.present if self.present else 0.0


def seed_complexes(stage: Path, experiment: str):
    """(group, protein_id, pdb text) for every complex stage 03 sent."""
    directory = jp.inputs_root(stage) / experiment
    if not directory.is_dir():
        return
    for archive_path in sorted(directory.glob("*.tar.gz")):
        group_key = archive_path.name[: -len(".tar.gz")]
        with tarfile.open(archive_path, "r:gz") as archive:
            for name in sorted(archive.getnames()):
                cleaned = name[2:] if name.startswith("./") else name
                parts = cleaned.split("/")
                if len(parts) != 2 or not parts[1].endswith(".pdb"):
                    continue
                if parts[1] != f"{parts[0]}.pdb":
                    continue          # a fibril or some other carried pdb
                handle = archive.extractfile(name)
                if handle is not None:
                    yield group_key, parts[0], handle.read().decode("utf-8")


def measure(pdb_text: str, positions: List[int], cutoff: float
            ) -> Tuple[int, int, float, List[Tuple[int, float]]]:
    """(present, contacting, closest approach, [(residue, distance) too far])."""
    structure = gemmi.read_pdb_string(pdb_text)
    structure.setup_entities()

    protein_chains, ligand_chains = classify_chains(structure)
    if not protein_chains:
        raise StructureError("no protein chain")
    if not ligand_chains:
        raise StructureError("no ligand chain -- this seed carries no fibre")
    ligand, _ = most_contacted(structure, protein_chains, ligand_chains)

    # One chain of the tied dimer, the same one stage 04 scores against.
    chain = protein_chains[0]
    approaches = site_approaches(structure, chain, positions, ligand)
    far = sorted((number, distance) for number, distance in approaches.items()
                 if distance > cutoff)
    closest = chain_approach(structure, chain, ligand)
    contacting = sum(1 for distance in approaches.values() if distance <= cutoff)
    return len(approaches), contacting, (closest if closest is not None else -1.0), far


def run(stage: Path, experiment: Optional[str], cutoff: float,
        verbose: bool) -> List[SeedResult]:
    names = [experiment] if experiment else jp.inputs_experiments(stage)
    results: List[SeedResult] = []
    for experiment_name in names:
        records = read_records(
            jp.fixed_residues_path(jp.stage01_root(stage), experiment_name)
        )
        seen: set = set()
        for group_key, protein_id, pdb_text in seed_complexes(stage, experiment_name):
            if protein_id in seen:
                continue
            seen.add(protein_id)
            mapped = records.get(protein_id)
            if not mapped:
                _log(f"[site] {protein_id}: stage 01 recorded no fixed residues, skipped")
                continue
            # The generated side of stage 01's record: the seed pdb is the
            # design, so it carries the design's numbering, not the motif's.
            positions = sorted({split_map_key(value)[1] for value in mapped.values()})
            try:
                present, contacting, closest, far = measure(pdb_text, positions, cutoff)
            except (StructureError, RuntimeError, ValueError, KeyError) as exc:
                _log(f"[site] {protein_id}: FAILED ({exc})")
                continue
            result = SeedResult(protein_id, group_key, len(positions), present,
                                contacting, closest, far)
            results.append(result)
            line = (f"[site] {protein_id}: {contacting}/{present} fixed residue(s) "
                    f"within {cutoff:.1f} A ({result.fraction:.0%}), "
                    f"chain closest {closest:.2f} A")
            if present != len(positions):
                line += f", {len(positions) - present} not in the pdb"
            _log(line)
            if verbose and far:
                _log("         beyond the cutoff: "
                     + ", ".join(f"{number}@{distance:.1f}" for number, distance in far[:8]))
    return results


def summarise(results: List[SeedResult], cutoff: float) -> None:
    if not results:
        _log("\n[site] nothing measured -- no seed carried both a fibre and a "
             "stage 01 record")
        return
    fractions = sorted(result.fraction for result in results)
    lowest = fractions[0]
    _log(f"\n[site] {len(results)} seed(s) measured at {cutoff:.1f} A")
    _log(f"[site]   lowest  {lowest:.0%}   median {fractions[len(fractions) // 2]:.0%}"
         f"   highest {fractions[-1]:.0%}")

    # Cumulative, and said so: an earlier version printed these as if each row
    # were its own band, which made the last row read "3 seeds below 60%" when
    # it meant "3 seeds in total".
    _log("[site]   seeds that would survive each gate:")
    _log("[site]     " + "   ".join(
        f"{threshold:.0%}: {sum(1 for fraction in fractions if fraction >= threshold)}"
        for threshold in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
    ) + f"   (of {len(results)})")

    _log("")
    if lowest >= 1.0:
        _log("[site] Every fixed residue of every seed is on the fibre, so the "
             "whole set is a binding site and the gate can be strict:")
        _log("[site]     --min-site-contact 1.0")
    else:
        # Below the worst seed, never at it: a prediction is allowed to be no
        # worse than its own design, and a gate set at the minimum would reject
        # a fold that reproduced that design exactly.
        suggested = max(0.0, (int(lowest * 10) - 1) / 10.0)
        _log(f"[site] The worst seed has {lowest:.0%} of its fixed residues on the "
             f"fibre, so some were fixed for reasons other than binding.")
        _log(f"[site] Setting the gate at {lowest:.0%} would reject a prediction that "
             f"reproduced that seed exactly, so go one step below it:")
        _log(f"[site]     --min-site-contact {suggested:.1f}")
        if len(fractions) > 1 and fractions[len(fractions) // 2] - lowest >= 0.2:
            _log(f"[site] That is set by one outlier, though -- the median seed is at "
                 f"{fractions[len(fractions) // 2]:.0%}. The table above says what a "
                 f"stricter gate would cost you in seeds; --verbose names the "
                 f"residues that are off the fibre, which is worth a look before "
                 f"letting the worst seed set the threshold for all of them.")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--cutoff", type=float, default=SITE_CONTACT_CUTOFF,
                        help=f"contact distance in angstroms "
                             f"(default: {SITE_CONTACT_CUTOFF})")
    parser.add_argument("--verbose", action="store_true",
                        help="name the fixed residues that are beyond the cutoff")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    results = run(args.stage.resolve(), args.experiment, args.cutoff, args.verbose)
    summarise(results, args.cutoff)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

