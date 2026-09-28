#!/usr/bin/env python3
"""
Stage 08, step 1: build one AlphaFold3 json per linked construct, with cellulose.

    inputs/<experiment>/<group>.tar.gz   <- stage 07: <design>/<id>.fa and
                                            <design>/<design>.pdb
    job_runs/af3/<experiment>/<group>/<sequence_id>_holo.json

One fold, with the fibre. The question this stage asks is whether the construct
still binds cellulose once its two copies are fused, and a fold without the
fibre answers a different one.

WHERE THE CELLULOSE COMES FROM

Nothing has carried a fibril since stage 05, which drops both of them after the
alignment -- AlphaFold's predicted one and the reference one it superposed
against. That is not a problem here, because the json does not need a fibre's
COORDINATES: a ligand is given to AF3 as CCD codes plus the bonds between them,
and AF3 places it. What is needed is a structure to read the composition and
connectivity out of.

So it is read from the seed, exactly as stage 04 reads it from the complex it
was handed: the same ligand_block(), the same geometric bond search, the same
identification from atoms rather than residue names. The seed is recovered the
way job_paths documents -- the group key is the stem of the stage 01 json that
generated the group, and that json's "input" field names the seed it was
diffused from. Nothing is recorded anywhere for this; the path is derivable.

--seed overrides it, for a group whose stage 01 json has been moved or renamed.

The construct is one chain. RFD3 fused the two copies into a single continuous
sequence and ProteinMPNN wrote the linker, so what folds here is one protein and
one ligand -- unlike stage 04, which folded one copy of a tied pair.
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
from af3_json import (  # noqa: E402
    CHAIN_SEPARATOR,
    Af3JsonError,
    bond_direction,
    build_json,
    ligand_block,
)
from archives import load_table  # noqa: E402
from structures import classify_chains, most_contacted  # noqa: E402

STAGE = Path(__file__).resolve().parents[2]


class PrepareError(Exception):
    """A construct's json could not be built."""


def _log(*parts: object) -> None:
    print(*parts, flush=True)


@dataclass
class PrepareReport:
    written: int = 0
    already: int = 0
    seen: int = 0
    seeds: Dict[str, str] = field(default_factory=dict)
    problems: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        if not self.written:
            return f"prepare: nothing new ({self.already} already built or folded)"
        return (f"prepare: {self.seen} construct(s) seen, {self.written} json(s) "
                f"written, {self.already} already done")


def first_chain(text: str) -> str:
    """The designed sequence. The construct is one chain, so a separator here
    means the file is not what this stage expects; the first part is taken
    regardless and the fold will be obviously wrong rather than subtly so."""
    lines = [line for line in text.splitlines() if line.strip()]
    body = "".join(line for line in lines if not line.startswith(">"))
    chain = body.split(CHAIN_SEPARATOR)[0].split("/")[0]
    if not chain:
        raise PrepareError("empty designed sequence")
    return chain


def seed_for_group(stage: Path, experiment: str, group_key: str) -> Path:
    """The seed structure this group was diffused from.

    group_key is the stem of the stage 01 json that generated it, and that
    json's "input" names the seed. Both steps can fail for an honest reason --
    a renamed experiment, a moved seed -- so both are reported separately
    rather than as one "not found".
    """
    config_path = jp.stage01_json_path(stage, experiment, group_key)
    if not config_path.is_file():
        raise PrepareError(
            f"no stage 01 json at {config_path}, so the seed this group came "
            f"from cannot be recovered. Pass --seed with the structure holding "
            f"the fibre, or put the json back."
        )
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        entries = jp.design_entries(payload)
        raw = str(next(iter(entries.values())).get("input", ""))
    except (OSError, json.JSONDecodeError, ValueError, StopIteration,
            AttributeError) as exc:
        raise PrepareError(f"cannot read the seed path out of {config_path}: {exc}")

    seed = jp.resolve_seed_path(jp.stage01_root(stage), raw, config_path)
    if seed is None:
        raise PrepareError(
            f"{config_path.name} names the seed {raw!r}, which is not there. "
            f"Pass --seed, or fix the path in that json."
        )
    return seed


def ligand_from_seed(seed: Path, ccd_code: Optional[str] = None
                     ) -> Tuple[dict, List[list], int, str]:
    """(the ligand entry, the bonds, how many units, how it was decided)."""
    structure = gemmi.read_structure(str(seed))
    structure.setup_entities()
    protein_chains, ligand_chains = classify_chains(structure)
    if not ligand_chains:
        raise PrepareError(f"{seed.name} carries no ligand chain to fold against")
    if protein_chains:
        chosen, _ = most_contacted(structure, protein_chains, ligand_chains)
    else:
        chosen = ligand_chains[0]
    ligand, bonds, size, odd, why = ligand_block(structure, chosen,
                                                 ccd_code=ccd_code)
    if odd:
        _log(f"[prepare] {seed.name}: WARNING {len(odd)} linkage(s) are not C-O: "
             f"{', '.join(odd[:3])}")
    return ligand, bonds, size, why


def prepare_group(stage: Path, experiment: str, group_key: str,
                  report: PrepareReport, scored: Optional[set] = None,
                  seed_override: Optional[Path] = None,
                  ccd_code: Optional[str] = None) -> None:
    archive_path = jp.input_archive_path(stage, experiment, group_key)
    if not archive_path.is_file():
        raise PrepareError(f"no input archive: {archive_path}")

    json_dir = jp.af3_json_dir(stage, experiment, group_key)
    json_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as archive:
        designs: Dict[str, List[str]] = {}
        for name in archive.getnames():
            cleaned = name[2:] if name.startswith("./") else name
            parts = cleaned.split("/")
            if len(parts) == 2 and parts[-1].endswith(".fa"):
                designs.setdefault(parts[0], []).append(name)
        if not designs:
            return

        outstanding: List[Tuple[str, str]] = []
        for members in designs.values():
            for member in sorted(members):
                sequence_id = Path(member).stem
                report.seen += 1
                if scored and sequence_id in scored:
                    report.already += 1
                    continue
                job_name = jp.af3_job_name(sequence_id, jp.HOLO_SUFFIX)
                if (jp.af3_json_path(stage, experiment, group_key, sequence_id,
                                     jp.HOLO_SUFFIX).is_file()
                        or jp.af3_output_dir(stage, job_name).is_dir()):
                    report.already += 1
                    continue
                outstanding.append((sequence_id, member))
        if not outstanding:
            return

        # The fibre is a property of the group's seed, so it is read once for
        # the whole group rather than once per sequence.
        try:
            seed = seed_override or seed_for_group(stage, experiment, group_key)
            ligand, bonds, size, why = ligand_from_seed(seed, ccd_code)
        except (PrepareError, Af3JsonError, RuntimeError, ValueError, OSError) as exc:
            report.problems.append(f"{experiment}/{group_key}: {exc}")
            _log(f"[prepare] {group_key}: FAILED ({exc})")
            return
        report.seeds[group_key] = seed.name
        _log(f"[prepare] {group_key}: fibre from {seed.name} -- {size} x "
             f"{ligand['ccdCodes'][0]}, {len(bonds)} bond(s) "
             f"{bond_direction(bonds)}")
        _log(f"[prepare] {group_key}: ligand identified as {why}")

        for sequence_id, member in sorted(outstanding):
            handle = archive.extractfile(member)
            if handle is None:
                report.problems.append(f"{experiment}/{sequence_id}: unreadable")
                continue
            try:
                sequence = first_chain(handle.read().decode("utf-8"))
            except PrepareError as exc:
                report.problems.append(f"{experiment}/{sequence_id}: {exc}")
                continue
            job_name = jp.af3_job_name(sequence_id, jp.HOLO_SUFFIX)
            destination = jp.af3_json_path(stage, experiment, group_key,
                                           sequence_id, jp.HOLO_SUFFIX)
            destination.write_text(
                json.dumps(build_json(job_name, sequence, ligand, bonds),
                           indent=2) + "\n",
                encoding="utf-8",
            )
            report.written += 1
        _log(f"[prepare] {group_key}: {len(outstanding)} json(s) written")


def run_prepare(stage: Path, experiment: Optional[str] = None,
                seed_override: Optional[Path] = None,
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
                              seed_override, ccd_code)
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
    parser.add_argument("--seed", type=Path, default=None,
                        help="the structure to read the fibre from, instead of "
                             "the seed named by the group's stage 01 json")
    parser.add_argument("--ccd-code", default=None,
                        help="force this CCD code for every fibre unit instead "
                             "of reading it from the atoms")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    report = run_prepare(args.stage.resolve(), args.experiment,
                         args.seed, args.ccd_code)
    _log(f"[done] {report.summary()}")
    for problem in report.problems[:10]:
        _log(f"[done]   {problem}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
