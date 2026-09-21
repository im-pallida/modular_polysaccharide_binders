#!/usr/bin/env python3
"""
Given an experiment (a folder of jsons under json/<experiment>/): 
1. Validates the jsons;
2. Checks each referenced seed structure;
3. Decides how many structures to generate per json
4. Writes a run list;
5. Hands it to whichever dispatcher fits the machine (cluster if sbatch exists, workstation otherwise) and;
6. Saves the results into the archives.

Usage:
    ./stage_01_launcher.py <experiment>                  # prompts for counts
    ./stage_01_launcher.py <experiment> --count 25       # same count for every json
    ./stage_01_launcher.py <experiment> --counts small.json=25,large.json=10

Exits non-zero if any job failed, so it can be used from a wrapper script.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_ROOT = SCRIPT_DIR.parent
HELPING_SCRIPTS_DIR = SCRIPT_DIR / "helping_scripts"
# job_paths.py is shared by every stage and lives at the repo root.
COMMON_DIR = STAGE_ROOT.parent / "common"

sys.path.insert(0, str(COMMON_DIR))
sys.path.insert(0, str(HELPING_SCRIPTS_DIR))

import job_paths as jp  # noqa: E402
import run_cluster  # noqa: E402
import run_workstation  # noqa: E402
from designs import fixed_residues  # noqa: E402
from filter_designs import Stage01InputError, run_filter  # noqa: E402
from run_one_job import archive_experiment, exclusive_lock  # noqa: E402
from symmetry_check import (  # noqa: E402
    StructureError,
    check_symmetry,
    format_translation_spec,
    has_ligand,
    load_structure,
)
from transfer_to_stage02 import TransferError, run_transfer  # noqa: E402

REQUIRED_TOP_LEVEL_KEYS = ("input", "contig", "symmetry")
REQUIRED_SYMMETRY_KEYS = ("id",)


def _log(*parts: object) -> None:
    print(*parts, flush=True)


# Step 1: Input validation


@dataclass
class JsonValidationResult:
    is_valid: bool
    reason: str = ""
    # design name -> seed structure. A flat json (no design name) uses the key ''.
    seed_structures: Dict[str, Path] = field(default_factory=dict)
    seed_was_relative: bool = False


def _missing_keys(data: dict, required_keys: Sequence[str], label: str) -> str:
    """Empty string means everything is present."""
    missing = [key for key in required_keys if key not in data]
    return f"{label} missing required field(s): {missing}" if missing else ""


def validate_json_file(json_path: Path, stage: Path) -> JsonValidationResult:
    """Checks the file parses, has the fields RFD3 needs and that the seed
    structure its "input" field points at actually exists."""
    try:
        text = json_path.read_text(encoding="utf-8")
    except OSError as exc:
        return JsonValidationResult(False, f"could not read file: {exc}")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return JsonValidationResult(False, f"invalid JSON syntax: {exc}")

    try:
        entries = jp.design_entries(data)
    except ValueError as exc:
        return JsonValidationResult(False, str(exc))

    seeds: Dict[str, Path] = {}
    relative = False

    for design, config in entries.items():
        where = f"design {design!r}" if design else "top-level"

        problem = _missing_keys(config, REQUIRED_TOP_LEVEL_KEYS, where)
        if problem:
            return JsonValidationResult(False, problem)

        symmetry = config["symmetry"]
        if not isinstance(symmetry, dict):
            return JsonValidationResult(
                False, f"{where}: 'symmetry' must be an object, got {type(symmetry).__name__}"
            )
        problem = _missing_keys(symmetry, REQUIRED_SYMMETRY_KEYS, f"{where} 'symmetry'")
        if problem:
            return JsonValidationResult(False, problem)

        raw_input = config["input"]
        if not isinstance(raw_input, str) or not raw_input:
            return JsonValidationResult(
                False, f"{where}: 'input' must be a non-empty string, got {raw_input!r}"
            )

        seed_structure = jp.resolve_seed_path(stage, raw_input, json_path)
        if seed_structure is None:
            return JsonValidationResult(
                False,
                f"{where}: 'input' seed structure not found: {raw_input!r} "
                f"(looked relative to {stage} and {json_path.parent})",
            )
        seeds[design] = seed_structure
        relative = relative or not Path(raw_input).is_absolute()

    return JsonValidationResult(True, seed_structures=seeds, seed_was_relative=relative)


def validate_jsons(experiment_dir: Path, json_names: Sequence[str], stage: Path) -> Dict[str, Path]:
    """Validates every json, reporting all problems at once. 
    Returns each json's seed structure path."""
    problems: List[str] = []
    relative_seeds: List[str] = []
    seed_structures: Dict[str, Path] = {}

    for name in json_names:
        result = validate_json_file(experiment_dir / name, stage)
        if not result.is_valid:
            problems.append(f"  {name}: {result.reason}")
            continue
        for design, seed in result.seed_structures.items():
            seed_structures[f"{name}[{design}]" if design else name] = seed
        if result.seed_was_relative:
            relative_seeds.append(name)

    if problems:
        raise SystemExit("ERROR: invalid json file(s):\n" + "\n".join(problems))

    _log(f"[ok] all {len(json_names)} json file(s) passed validation")
    if relative_seeds:
        _log(f"[note] {len(relative_seeds)} json(s) give 'input' as a relative path "
             f"({', '.join(relative_seeds)}). RFD3 resolves it against its own working "
             f"directory on the compute node, which is not this one -- prefer absolute paths.")
    return seed_structures


def warn_designs_without_fixed_side_chains(experiment_dir: Path, json_names: Sequence[str]) -> None:
    """Say at launch which designs cannot go past stage 02.

    Stage 02 aligns on the residues whose side chains were fixed
    (select_fixed_atoms valued "ALL") and stage 03 holds those same residues
    fixed. A design that names none can still be generated -- it just stops at
    stage 02 -- and finding that out now costs nothing, whereas finding it out
    after a few hundred structures costs the GPU time.
    """
    unusable = []
    for name in json_names:
        try:
            data = json.loads((experiment_dir / name).read_text(encoding="utf-8"))
            for design, config in jp.design_entries(data).items():
                if not fixed_residues(config):
                    unusable.append(f"{name}[{design}]" if design else name)
        except (OSError, ValueError, json.JSONDecodeError):
            continue  # validate_jsons already reported anything unreadable
    if unusable:
        _log(f"[note] {len(unusable)} design(s) fix no side chains "
             f"(no select_fixed_atoms entry valued \"ALL\"): {', '.join(unusable)}")
        _log("       these will generate, but stage 02 cannot align them and stage 03 "
             "cannot hold anything fixed, so they stop after generation.")


def check_seed(seed_structure: Path) -> None:
    """Symmetry is a hard gate; a missing ligand is only a notice.
    The structure is parsed once and reused for both checks."""
    try:
        structure = load_structure(seed_structure)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: could not read seed structure {seed_structure}\n  {exc}")

    try:
        result = check_symmetry(structure)
    except StructureError as exc:
        raise SystemExit(f"ERROR: symmetry check failed for {seed_structure}\n  reason: {exc}")

    if not result.is_symmetric:
        raise SystemExit(
            f"ERROR: symmetry check failed for {seed_structure}\n  reason: {result.reason}"
        )
    _log(f"[ok] symmetry check passed: {seed_structure}")
    _log(f"     {result.summary()}")
    if result.transform is not None and result.transform.is_linear:
        # This is what each job will derive and hand to RFD3 as
        # RFD3_<symmetry.id>_TRANSLATIONS, so show it before anything is queued.
        _log(f"     translations: {format_translation_spec(result.transform.displacement)}")
    elif result.transform is not None:
        _log("[note] the two chains are related by a rotation, not a pure translation, "
             "so the translation operator cannot be derived from this seed -- set "
             "RFD3_<symmetry.id>_TRANSLATIONS explicitly if that is intended.")
    if result.altlocs_collapsed:
        _log(f"[note] collapsed {result.altlocs_collapsed} alternate conformer(s) "
             f"(highest occupancy kept)")
 
    ligands = has_ligand(structure)
    if ligands:
        _log(f"[ok] ligand present: {', '.join(ligands)}")
    else:
        _log("[note] no ligand in this seed -- generation runs without one; "
             "it needs to be introduced in a later stage.")


def check_seeds(seed_structures: Dict[str, Path]) -> None:
    """Each distinct seed is checked once, even if several jsons share it."""
    for seed in sorted(set(seed_structures.values())):
        users = sorted(name for name, path in seed_structures.items() if path == seed)
        _log(f"[check] {seed}  (used by: {', '.join(users)})")
        check_seed(seed)


# Step 2. Input search


def experiment_dir_for(stage: Path, experiment: str) -> Path:
    """Experiment folder search."""
    experiment_dir = jp.experiment_json_dir(stage, experiment)
    if not experiment_dir.is_dir():
        raise SystemExit(f"ERROR: no such experiment folder: {experiment_dir}")
    return experiment_dir


def discover_jsons(experiment_dir: Path) -> List[str]:
    """json search."""
    jsons = sorted(path.name for path in experiment_dir.glob("*.json"))
    if not jsons:
        raise SystemExit(f"ERROR: no json files found under {experiment_dir}")
    _log(f"[ok] found {len(jsons)} json file(s) under {experiment_dir}")
    return jsons


# Step 3. Preparation for a run


def prompt_counts(json_names: Sequence[str]) -> Dict[str, int]:
    """Prompts and checks the format of the prompt input."""
    counts: Dict[str, int] = {}
    for name in json_names:
        while True:
            raw = input(f"How many of {name} structures do you want to generate? ").strip()
            # isdecimal(), not isdigit(): isdigit() accepts things like '2'
            # (superscript two) that int() then refuses.
            if raw.isdecimal():
                counts[name] = int(raw)
                break
            _log(f"  ERROR: must be a non-negative integer, got: {raw!r}")
    return counts


def resolve_counts(
    json_names: Sequence[str], count: Optional[int], counts_spec: Optional[str]
) -> Dict[str, int]:
    """Counts from --count, from --counts or interactively."""
    if count is not None:
        return {name: count for name in json_names}

    if counts_spec:
        counts: Dict[str, int] = {name: 0 for name in json_names}
        for item in counts_spec.split(","):
            item = item.strip()
            if not item:
                continue
            name, sep, raw = item.partition("=")
            name, raw = name.strip(), raw.strip()
            if not sep or not raw.isdecimal():
                raise SystemExit(f"ERROR: --counts entry must be NAME=INTEGER, got: {item!r}")
            if name not in counts:
                raise SystemExit(
                    f"ERROR: --counts names {name!r}, which is not in this experiment. "
                    f"Available: {', '.join(json_names)}"
                )
            counts[name] = int(raw)
        return counts

    if not sys.stdin.isatty():
        raise SystemExit(
            "ERROR: no counts given and stdin is not a terminal. "
            "Use --count N or --counts NAME=N,... when running non-interactively."
        )
    return prompt_counts(json_names)


def build_run_list(stage: Path, experiment: str, counts: Dict[str, int]) -> Tuple[Path, int]:
    """Write one '<json_filename><TAB><global_seq>' row per structure, advancing
    the persistent global sequence counter as it goes.

    The counter is read and written under an exclusive lock, so two launches on
    a shared filesystem cannot hand out the same sequence numbers."""
    counter_file = jp.counter_path(stage)
    counter_file.parent.mkdir(parents=True, exist_ok=True)

    rows: List[jp.RunRow] = []
    plan_lines: List[str] = []

    with exclusive_lock(Path(f"{counter_file}.lock")):
        try:
            global_seq = int(counter_file.read_text().strip()) if counter_file.exists() else 0
        except ValueError as exc:
            raise SystemExit(f"ERROR: sequence counter {counter_file} is corrupt: {exc}")

        for json_rel, count in counts.items():
            if count == 0:
                plan_lines.append(f"[plan] {json_rel}: skipped")
                continue
            start = global_seq + 1
            for _ in range(count):
                global_seq += 1
                rows.append((json_rel, global_seq))
            plan_lines.append(
                f"[plan] {json_rel}: {count} structure(s), global seq {start}-{global_seq}"
            )

        if not rows:
            raise SystemExit("Nothing to do - all counts were 0.")

        # Only claim sequence numbers once the plan is known to be non-empty.
        counter_file.write_text(f"{global_seq}\n")

    for line in plan_lines:
        _log(line)

    run_list = jp.run_list_path(stage, experiment)
    jp.write_run_list(run_list, rows)
    return run_list, len(rows)


# Step 4. Dispatching


def dispatch(experiment: str, run_list: Path, stage: Path):
    """Cluster if sbatch is available, workstation otherwise."""
    if shutil.which("sbatch") is not None:
        return run_cluster.dispatch_cluster(experiment, run_list, stage)
    return run_workstation.dispatch_workstation(experiment, run_list, stage)


def archive(stage: Path, experiment: str, rows: Sequence[jp.RunRow]) -> None:
    _log("[archive] updating outputs_clean archives...")
    for result in archive_experiment(stage, experiment, rows):
        total = result.added + result.already_present
        _log(f"[archive] {result.group_key}: +{result.added} new, "
             f"{total} member(s) total -> {result.archive}")
        if result.missing:
            preview = ", ".join(result.missing[:5])
            more = f" (+{len(result.missing) - 5} more)" if len(result.missing) > 5 else ""
            _log(f"[archive] {result.group_key}: {len(result.missing)} structure(s) "
                 f"not on disk, not archived: {preview}{more}")


def filter_structures(stage: Path) -> bool:
    """Sort everything unfiltered in outputs_clean/ into passed/rejected."""
    _log("[filter] sorting structures into passed/rejected...")
    try:
        report = run_filter(stage)
    except (OSError, ValueError, Stage01InputError) as exc:
        _log(f"[filter] FAILED: {exc}")
        _log("[filter] generation and archiving are unaffected -- fix the cause and "
             "re-run scripts/helping_scripts/filter_designs.py")
        return False

    _log(f"[filter] {report.summary()}")
    for table in report.tables:
        _log(f"[filter] results -> {table}")
    if not report.ok:
        _log(f"[filter] {report.counts.get('ERROR', 0)} structure(s) could not be "
             f"evaluated -- see the error_reason column")
    return report.ok


def transfer_structures(stage: Path) -> bool:
    """Hand every passed structure not already in stage 02 across to it."""
    _log("[transfer] handing passed structures to stage 02...")
    try:
        report = run_transfer(stage)
    except (OSError, TransferError) as exc:
        _log(f"[transfer] FAILED: {exc}")
        _log("[transfer] everything up to this point is unaffected -- fix the cause "
             "and re-run scripts/helping_scripts/transfer_to_stage02.py")
        return False

    _log(f"[transfer] {report.summary()}")
    if report.incomplete:
        preview = ", ".join(report.incomplete[:5])
        more = f" (+{len(report.incomplete) - 5} more)" if len(report.incomplete) > 5 else ""
        _log(f"[transfer] {len(report.incomplete)} passed structure(s) could not be "
             f"transferred, missing half their file pair: {preview}{more}")
    return report.ok


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("experiment", help="folder name under json/")
    parser.add_argument(
        "--stage", type=Path, default=STAGE_ROOT,
        help=f"stage root directory (default: {STAGE_ROOT})",
    )
    parser.add_argument(
        "--no-filter", action="store_true",
        help="generate only; skip the geometry filter that normally runs afterwards",
    )
    parser.add_argument(
        "--no-transfer", action="store_true",
        help="skip handing passed structures to stage 02 after filtering",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--count", type=int, default=None,
        help="generate this many structures for every json (non-interactive)",
    )
    group.add_argument(
        "--counts", default=None,
        help="per-json counts, e.g. 'small.json=25,large.json=10' (non-interactive)",
    )
    args = parser.parse_args(argv)
    if args.count is not None and args.count < 0:
        parser.error("--count must be non-negative")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    stage = args.stage.resolve()

    experiment_dir = experiment_dir_for(stage, args.experiment)
    json_names = discover_jsons(experiment_dir)
    seed_structures = validate_jsons(experiment_dir, json_names, stage)
    check_seeds(seed_structures)
    warn_designs_without_fixed_side_chains(experiment_dir, json_names)

    counts = resolve_counts(json_names, args.count, args.counts)
    run_list, total = build_run_list(stage, args.experiment, counts)
    _log(f"[plan] {total} structure(s) queued in {run_list}")

    report = dispatch(args.experiment, run_list, stage)
    archive(stage, args.experiment, report.rows)

    filtered_ok = True
    if not args.no_filter:
        filtered_ok = filter_structures(stage)

    transferred_ok = True
    if not args.no_transfer:
        transferred_ok = transfer_structures(stage)

    _log(f"[done] {report.summary()}")
    return 0 if report.ok and filtered_ok and transferred_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

