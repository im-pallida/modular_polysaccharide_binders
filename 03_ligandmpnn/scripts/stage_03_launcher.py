#!/usr/bin/env python3
"""
Runs stage 03 end to end:

1. Prepares each group stage 02 handed over -- the protein WITH the fibril chain
   it sits on, and the residues LigandMPNN must not redesign;
2. Designs sequences for it, symmetrically: chain A and chain B tied, so one
   sequence is produced and applied to both, with the ligand in context;
3. Scores every sequence and keeps the best three per protein -- by HIGHEST
   overall_confidence, which is the inverse of the ProteinMPNN sort;
4. Hands those to stage 04, which folds one chain of each in the fibril's presence;
5. Clears job_runs/ and inputs_prepared/ for everything the archives already
   hold. LigandMPNN writes sixteen repacked backbones per protein that nothing
   reads; what IS needed lives in a tar.gz.

One LigandMPNN job per group, submitted with sbatch where it exists and run in
place otherwise. Proteins already designed are skipped, so adding structures to
a group costs only the new ones.

Usage:
    ./stage_03_launcher.py                      # everything outstanding
    ./stage_03_launcher.py --experiment NAME
    ./stage_03_launcher.py --no-transfer

Needs LIGANDMPNN_ROOT pointing at the LigandMPNN checkout (the folder holding
run.py), and LIGANDMPNN_CHECKPOINT if the weights are not at the default path.
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
from ligand_prepare import PrepareError, prepare_group  # noqa: E402
from run_ligandmpnn import (  # noqa: E402
    LigandMpnnError,
    RunReport,
    design_group,
)
from cleanup import run_cleanup  # noqa: E402
from select_sequences import SelectionError, run_selection  # noqa: E402
from transfer_to_stage04 import TransferError, run_transfer  # noqa: E402

SBATCH_SCRIPT = HELPING_SCRIPTS_DIR / "run_ligandmpnn.sbatch"


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

    report = RunReport()
    try:
        design_group(stage, experiment, group_key, report)
    except LigandMpnnError as exc:
        _log(f"[ligandmpnn] {experiment}/{group_key}: FAILED ({exc})")
        return False
    for problem in report.problems:
        _log(f"[ligandmpnn]   {problem}")
    _log(f"[ligandmpnn] {experiment}/{group_key}: {report.designed} protein(s), "
         f"{report.sequences} sequence(s)")
    return report.ok


def design_structures(stage: Path, experiment: Optional[str]) -> bool:
    groups = outstanding_groups(stage, experiment)
    if not groups:
        _log(f"[ligandmpnn] nothing to design under {jp.inputs_root(stage)}")
        return True

    mode = site.mode()
    _log(f"[ligandmpnn] {len(groups)} group(s), mode={mode}, one job per group")
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
             f"{len(prepared.prepared)} complex(es) ready"
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
    _log("[transfer] handing the best sequences to stage 04...")
    try:
        report = run_transfer(stage, experiment)
    except (OSError, TransferError) as exc:
        _log(f"[transfer] FAILED: {exc}")
        return False
    _log(f"[transfer] {report.summary()}")
    return report.ok


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
                        help="skip LigandMPNN; select from what is already designed")
    parser.add_argument("--no-transfer", action="store_true",
                        help="do not hand anything to stage 04")
    parser.add_argument("--no-clean", action="store_true",
                        help="keep job_runs/ and inputs_prepared/")
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

    # Last, and only after the transfer: stage 04 has taken its copy by then, and
    # the archives the cleanup checks against are complete.
    if not args.no_clean:
        report = run_cleanup(stage, args.experiment, args.clean_dry_run)
        _log(f"[clean] {report.summary()}")

    return 0 if designed_ok and selected_ok and transferred_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
