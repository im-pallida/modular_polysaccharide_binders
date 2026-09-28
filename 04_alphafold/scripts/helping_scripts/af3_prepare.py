#!/usr/bin/env python3
"""
Stage 04, step 1: build an AlphaFold3 json per design, ligand included.

    inputs/<experiment>/<group>.tar.gz   <- stage 03: <protein_id>/<sequence_id>.fa
                                            and <protein_id>/<protein_id>.pdb
    job_runs/af3/<experiment>/<group>/<sequence_id>_ligand.json

One monomer, folded in the presence of the fibril it was designed against. The
pipeline used to fold the design alone here and only meet the ligand two stages
later, after superposing the prediction back onto a reference -- which is where
the fit error and the clashes came from. Designing and folding both against the
ligand as diffused removes that step and the whole class of error with it.

The ligand is read out of the complex, never assumed:

    which chain   the ligand chain the protein has most contacts with, counted
                  per structure. A seed can carry several fibril strands and the
                  design only touches one; folding against a strand it never
                  approaches would be a different question than the one asked.
    ccdCodes      the residue names, checked before use. A name and a CCD code
                  are different namespaces, and a seed naming its glucose BGL
                  once built twenty detergent molecules here instead. Pass
                  --ccd-code BGC (cellulose) or NAG (chitin) when the seed's
                  naming cannot be changed.
    bonds         the atom pair under BOND_CUTOFF between each consecutive pair
                  of residues, measured rather than hardcoded, so any linkage
                  comes out right. For a beta-1,4 glucan that is C1 to O4; which
                  of the two carries the lower residue number depends on which
                  end the seed was numbered from, so the direction found is
                  printed. A pair with no such contact is reported: a json with
                  a silently disconnected ligand folds fine and means nothing.

One protein chain. The two copies are tied and identical, so a dimer prediction
would answer a question about association, not about binding -- and the linker
stage takes the association up later. Folding one chain also asks the harder
question: whether the sequence holds its designed fold without its partner.

The json's shape is copied field for field from a file that ran correctly on
this cluster: dialect alphafold3 version 2, protein id as a bare string,
unpairedMsa empty so AF3 folds the sequence alone without searching, and the
ligand keeping the seed's own chain name so the result can be diffed against a
hand-written json from the same seed. See common/af3_json.py.
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
from archives import extract_member, load_table  # noqa: E402
from af3_json import (  # noqa: E402
    CHAIN_SEPARATOR,
    Af3JsonError,
    bond_direction,
    build_json,
    ligand_block,
)
from structures import classify_chains, most_contacted  # noqa: E402

STAGE = Path(__file__).resolve().parents[2]



class PrepareError(Exception):
    """A design's json could not be built."""


def _log(*parts: object) -> None:
    print(*parts, flush=True)


@dataclass
class PrepareReport:
    written: int = 0
    already: int = 0
    seen: int = 0
    ligand_sizes: List[int] = field(default_factory=list)
    odd_linkages: List[str] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        if not self.written:
            return f"prepare: nothing new ({self.already} already built or folded)"
        sizes = self.ligand_sizes
        return (f"prepare: {self.seen} design(s) seen, {self.written} json(s) written, "
                f"{self.already} already done, ligand "
                f"{min(sizes)}-{max(sizes)} residues")


# ---------------------------------------------------------------------------
# Reading the ligand out of the complex
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def first_chain(text: str) -> str:
    """The designed sequence, one chain of LigandMPNN's ':'-joined pair."""
    lines = [line for line in text.splitlines() if line.strip()]
    body = "".join(line for line in lines if not line.startswith(">"))
    chain = body.split(CHAIN_SEPARATOR)[0]
    if not chain:
        raise PrepareError("empty designed sequence")
    return chain


def prepare_group(stage: Path, experiment: str, group_key: str,
                  report: PrepareReport, scored: Optional[set] = None,
                  ccd_code: Optional[str] = None) -> None:
    archive_path = jp.input_archive_path(stage, experiment, group_key)
    if not archive_path.is_file():
        raise PrepareError(f"no input archive: {archive_path}")

    json_dir = jp.af3_json_dir(stage, experiment, group_key)
    json_dir.mkdir(parents=True, exist_ok=True)
    scratch = json_dir / "_scratch"

    with tarfile.open(archive_path, "r:gz") as archive:
        names = archive.getnames()
        designs: Dict[str, List[str]] = {}
        complexes: Dict[str, str] = {}
        for name in names:
            parts = (name[2:] if name.startswith("./") else name).split("/")
            if len(parts) < 2:
                continue
            if parts[-1].endswith(".fa"):
                designs.setdefault(parts[0], []).append(name)
            elif parts[-1].endswith(".pdb"):
                complexes[parts[0]] = name

        for protein_id, members in sorted(designs.items()):
            outstanding = []
            for member in sorted(members):
                sequence_id = Path(member).stem
                job_name = jp.af3_job_name(sequence_id, jp.LIGAND_SUFFIX)
                report.seen += 1
                if scored and sequence_id in scored:
                    report.already += 1
                    continue
                if jp.af3_json_path(stage, experiment, group_key, sequence_id,
                                    jp.LIGAND_SUFFIX).is_file():
                    report.already += 1
                    continue
                if jp.af3_output_dir(stage, job_name).is_dir():
                    report.already += 1
                    continue
                outstanding.append((sequence_id, job_name, member))
            if not outstanding:
                continue

            # The ligand is a property of the complex, so it is read once for
            # all of that complex's designs rather than once per design.
            try:
                if protein_id not in complexes:
                    raise PrepareError(
                        f"no complex pdb for {protein_id} in {archive_path.name}; "
                        f"stage 03 should have sent it with the designs"
                    )
                path = extract_member(archive, complexes[protein_id], scratch)
                structure = gemmi.read_structure(str(path))
                structure.setup_entities()
                path.unlink(missing_ok=True)
                protein_chains, ligand_chains = classify_chains(structure)
                chosen, contacts = most_contacted(
                    structure, protein_chains, ligand_chains
                )
                ligand, bonds, size, odd, why = ligand_block(
                    structure, chosen, ccd_code=ccd_code
                )
            except (PrepareError, Af3JsonError, RuntimeError, ValueError, OSError) as exc:
                report.problems.append(f"{experiment}/{protein_id}: {exc}")
                _log(f"[prepare] {protein_id}: FAILED ({exc})")
                continue

            if len(ligand_chains) > 1:
                _log(f"[prepare] {protein_id}: {len(ligand_chains)} ligand chain(s) "
                     f"{ligand_chains}, folding with {chosen} ({contacts} contacts)")

            for sequence_id, job_name, member in outstanding:
                handle = archive.extractfile(member)
                if handle is None:
                    report.problems.append(f"{experiment}/{sequence_id}: unreadable")
                    continue
                try:
                    sequence = first_chain(handle.read().decode("utf-8"))
                except PrepareError as exc:
                    report.problems.append(f"{experiment}/{sequence_id}: {exc}")
                    continue
                payload = build_json(job_name, sequence, ligand, bonds)
                (json_dir / f"{job_name}.json").write_text(
                    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
                )
                report.written += 1
                report.ligand_sizes.append(size)
            _log(f"[prepare] {protein_id}: {len(outstanding)} json(s), "
                 f"ligand {chosen} -> chain {ligand['id']}, {size} x "
                 f"{ligand['ccdCodes'][0]}, {len(bonds)} bond(s) "
                 f"{bond_direction(bonds)}")
            # Said every time, not only when it is surprising: this is the
            # decision that once put twenty detergent molecules in the box.
            _log(f"[prepare] {protein_id}: ligand identified as {why}")
            if odd:
                report.odd_linkages.append(f"{protein_id}: {', '.join(odd[:3])}")
                _log(f"[prepare] {protein_id}: WARNING {len(odd)} linkage(s) are not "
                     f"C-O, which is not how a sugar chain bonds: {', '.join(odd[:3])}")

    if scratch.exists() and not any(scratch.iterdir()):
        scratch.rmdir()


def run_prepare(stage: Path, experiment: Optional[str] = None,
                ccd_code: Optional[str] = None) -> PrepareReport:
    report = PrepareReport()
    names = [experiment] if experiment else jp.inputs_experiments(stage)
    if not names:
        _log(f"[prepare] nothing handed over under {jp.inputs_root(stage)}")
        return report
    for experiment_name in names:
        directory = jp.inputs_root(stage) / experiment_name
        if not directory.is_dir():
            continue
        scored = set(load_table(jp.results_table_path(stage, experiment_name),
                                key="sequence_id"))
        for archive_path in sorted(directory.glob("*.tar.gz")):
            group_key = archive_path.name[: -len(".tar.gz")]
            try:
                prepare_group(stage, experiment_name, group_key, report, scored,
                              ccd_code)
            except PrepareError as exc:
                report.problems.append(f"{experiment_name}/{group_key}: {exc}")
                _log(f"[prepare] {experiment_name}/{group_key}: FAILED ({exc})")
    return report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--ccd-code", default=None,
                        help="force this CCD code for every ligand unit instead "
                             "of taking the residue name (cellulose is BGC, "
                             "chitin NAG)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    report = run_prepare(args.stage.resolve(), args.experiment, args.ccd_code)
    _log(f"[done] {report.summary()}")
    for problem in report.problems[:10]:
        _log(f"[done]   {problem}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

