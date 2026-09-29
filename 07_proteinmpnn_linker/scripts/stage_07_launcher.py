#!/usr/bin/env python3
"""
Runs stage 07 end to end:

1. Reads each construct RFD3 built, finds the linker from diffused_index_map,
   and opens up the linker plus everything within 8 A of it;
2. Designs those positions with vanilla ProteinMPNN, holding the rest;
3. Scores every sequence and keeps the best three per construct -- by LOWEST
   score, which is the inverse of stage 03's LigandMPNN sort;
4. Hands those to stage 08, which folds each construct twice.

One ProteinMPNN job per group, submitted with sbatch where it exists and run in
place otherwise. Constructs already designed are skipped, so adding more costs
only the new ones.

Usage:
    ./stage_07_launcher.py                      # everything outstanding
    ./stage_07_launcher.py --experiment NAME
    ./stage_07_launcher.py --no-transfer

Needs MPNN_ROOT pointing at the ProteinMPNN checkout. Vanilla ProteinMPNN, not
LigandMPNN: the linker binds nothing, so there is no ligand for the model to use.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_ROOT = SCRIPT_DIR.parent
HELPING_SCRIPTS_DIR = SCRIPT_DIR / "helping_scripts"
COMMON_DIR = STAGE_ROOT.parent / "common"

sys.path.insert(0, str(COMMON_DIR))
sys.path.insert(0, str(HELPING_SCRIPTS_DIR))

# Before importing anything that needs gemmi: if this interpreter cannot, start
# again under one that can. A no-op wherever the environment is already right,
# which is every workstation run.
import bootstrap  # noqa: E402

try:
    bootstrap.ensure()
except bootstrap.BootstrapError as _exc:
    raise SystemExit(f"ERROR: {_exc}")

import job_paths as jp  # noqa: E402
import partitions  # noqa: E402
import site_config as site  # noqa: E402
from cleanup import run_cleanup  # noqa: E402
from linker_prepare import PrepareError, prepare_group  # noqa: E402
from run_mpnn import MpnnError, run_group  # noqa: E402
from select_sequences import SelectionError, run_selection  # noqa: E402
from transfer_to_stage08 import TransferError, run_transfer  # noqa: E402

SBATCH_SCRIPT = HELPING_SCRIPTS_DIR / "run_mpnn.sbatch"


def _log(*parts: object) -> None:
    print(*parts, flush=True)


def outstanding_groups(stage: Path, experiment: Optional[str]) -> List[Tuple[str, str]]:
    """(experiment, group) for everything stage 02 has handed over."""
    names = [experiment] if experiment else jp.inputs_experiments(stage)
    found: List[Tuple[str, str]] = []
    for name in names:
        directory = jp.inputs_root(stage) / name
        if not directory.is_dir():
            continue
        for archive in sorted(directory.glob("*.tar.gz")):
            found.append((name, archive.name[: -len(".tar.gz")]))
    return found


def dispatch_group(stage: Path, experiment: str, group_key: str) -> bool:
    """Design one group: sbatch where it exists, in place otherwise."""
    if site.mode() == "cluster":
        logs = stage / "logs" / experiment
        logs.mkdir(parents=True, exist_ok=True)
        command = [
            "sbatch", "--wait",
            *partitions.sbatch_args(),
            "--chdir", str(stage),
            "--output", str(logs / f"{group_key}-%j.out"),
            "--error", str(logs / f"{group_key}-%j.err"),
            # The stage goes on the command line: a submitted batch script
            # runs from SLURM's spool, not from the checkout.
            str(SBATCH_SCRIPT), experiment, group_key, str(stage),
        ]
        _log("$ " + " ".join(command))
        return subprocess.run(command).returncode == 0

    try:
        result = run_group(stage, experiment, group_key)
    except MpnnError as exc:
        _log(f"[mpnn] {experiment}/{group_key}: FAILED ({exc})")
        return False
    _log(f"[mpnn] {experiment}/{group_key}: {result.proteins} construct(s), "
         f"{result.sequences} sequence(s) -> {result.archive}")
    return True


def design_structures(stage: Path, experiment: Optional[str]) -> bool:
    groups = outstanding_groups(stage, experiment)
    if not groups:
        _log(f"[mpnn] nothing to design under {jp.inputs_root(stage)}")
        return True

    mode = site.mode()
    _log(f"[mpnn] {len(groups)} group(s), mode={mode}, one job per group")
    ok = True
    for experiment_name, group_key in groups:
        try:
            prepared = prepare_group(stage, experiment_name, group_key)
        except PrepareError as exc:
            _log(f"[prepare] {experiment_name}/{group_key}: FAILED ({exc})")
            ok = False
            continue
        for protein_id, reason in sorted(prepared.failed.items()):
            _log(f"[prepare]   {protein_id}: not prepared ({reason})")
        if prepared.failed:
            ok = False
        if prepared.nothing_to_do:
            continue
        _log(f"[prepare] {experiment_name}/{group_key}: "
             f"{len(prepared.prepared)} construct(s) ready"
             + (f", {len(prepared.already)} already designed" if prepared.already else ""))
        ok = dispatch_group(stage, experiment_name, group_key) and ok
    return ok


def select_best(stage: Path, experiment: Optional[str], keep: int) -> bool:
    _log("[select] scoring sequences and keeping the best per protein...")
    try:
        report = run_selection(stage, experiment, keep)
    except (OSError, SelectionError) as exc:
        _log(f"[select] FAILED: {exc}")
        return False
    _log(f"[select] {report.summary()}")
    for table in report.tables:
        _log(f"[select] results -> {table}")
    for problem in report.problems:
        _log(f"[select] {problem}")
    return report.ok


def transfer_structures(stage: Path, experiment: Optional[str]) -> bool:
    _log("[transfer] handing the best linker sequences to stage 08...")
    try:
        report = run_transfer(stage, experiment)
    except (OSError, TransferError) as exc:
        _log(f"[transfer] FAILED: {exc}")
        return False
    _log(f"[transfer] {report.summary()}")
    return report.ok



def tidy(stage: Path, experiment: Optional[str], dry_run: bool) -> bool:
    """Clear what the archives already hold. Runs last, after the transfer, so
    the archives the deletions are checked against are complete."""
    report = run_cleanup(stage, experiment, dry_run)
    _log(f"[clean] {report.summary()}")
    for path in report.unarchived[:5]:
        _log(f"[clean]   kept (not in any archive): {path}")
    return True

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=Path, default=STAGE_ROOT)
    parser.add_argument("--experiment", default=None,
                        help="only this experiment (default: every outstanding one)")
    parser.add_argument("--keep", type=int, default=3,
                        help="sequences to keep per protein (default: 3)")
    parser.add_argument("--no-design", action="store_true",
                        help="skip ProteinMPNN; select from what is already designed")
    parser.add_argument("--no-transfer", action="store_true",
                        help="do not hand anything to stage 08")
    parser.add_argument("--no-clean", action="store_true",
                        help="keep the raw files even once they are archived")
    parser.add_argument("--clean-dry-run", action="store_true",
                        help="report what cleanup would remove, remove nothing")
    parser.add_argument("--partition", default=None,
                        help="submit to this Slurm partition instead of asking "
                             "(cluster only; PIPELINE_PARTITION does the same)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    # Fill in whatever this machine has not exported -- tool locations, and
    # whether there is a queue. Anything already exported is left alone.
    try:
        site.apply()
    except site.SiteError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    # Asked once, here, before anything submits or any thread pool starts.
    if site.mode() == "cluster":
        try:
            partitions.choose(args.partition)
        except partitions.PartitionError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    stage = args.stage.resolve()

    designed_ok = True
    if not args.no_design:
        designed_ok = design_structures(stage, args.experiment)

    selected_ok = select_best(stage, args.experiment, args.keep)

    transferred_ok = True
    if not args.no_transfer:
        transferred_ok = transfer_structures(stage, args.experiment)

    # Last, and only after the transfer: stage 08 has taken its copy by then, and
    # the archives the cleanup checks against are complete.
    if not args.no_clean:
        tidy(stage, args.experiment, args.clean_dry_run)

    return 0 if designed_ok and selected_ok and transferred_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
