#!/usr/bin/env python3
"""
Runs stage 04 end to end:

1. Builds one AlphaFold3 json per designed sequence -- monomer only;
2. Folds each one, four at a time on the cluster, one at a time otherwise;
3. Scores every folded monomer against the backbone ProteinMPNN designed it
   onto, and sorts it passed or rejected;
4. Hands the passed folds to stage 05;
5. Clears outputs/, af3_runs/ and the input jsons for everything the archives
   already hold. What is needed lives in a tar.gz; what is not is removed.

This BLOCKS while AF3 runs. A few hundred sequences at four concurrent
four-hour jobs is days, so start it inside a terminal multiplexer:

    tmux new -s stage04          # or: screen -S stage04
    ./stage_04_launcher.py --stage .
    # detach with Ctrl-b d       # screen: Ctrl-a d
    # come back with: tmux attach -t stage04

Interrupting it is safe. Every step skips what is already done -- a json that
exists, a job whose output exists, a sequence already in the table -- so
re-running picks up where it stopped rather than starting over.

Usage:
    ./stage_04_launcher.py                      # everything outstanding
    ./stage_04_launcher.py --experiment NAME
    ./stage_04_launcher.py --no-fold            # score what is already folded
    ./stage_04_launcher.py --no-fold --no-score # transfer only, fold nothing
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_ROOT = SCRIPT_DIR.parent
HELPING_SCRIPTS_DIR = SCRIPT_DIR / "helping_scripts"
COMMON_DIR = STAGE_ROOT.parent / "common"

sys.path.insert(0, str(COMMON_DIR))
sys.path.insert(0, str(HELPING_SCRIPTS_DIR))

import job_paths as jp  # noqa: E402
from af3_prepare import PrepareError, run_prepare  # noqa: E402
from run_af3 import Af3Error, run_af3  # noqa: E402
from score_designs import run_scoring  # noqa: E402
from cleanup import run_cleanup  # noqa: E402
from transfer_to_stage05 import TransferError, run_transfer  # noqa: E402


def _log(*parts: object) -> None:
    print(*parts, flush=True)


def prepare_jsons(stage: Path, experiment: Optional[str]) -> bool:
    try:
        report = run_prepare(stage, experiment)
    except (PrepareError, OSError) as exc:
        _log(f"[prepare] FAILED: {exc}")
        return False
    _log(f"[prepare] {report.summary()}")
    return report.ok


def fold_sequences(stage: Path, experiment: Optional[str], max_concurrent: int) -> bool:
    try:
        report = run_af3(stage, experiment, max_concurrent)
    except (Af3Error, OSError) as exc:
        _log(f"[af3] FAILED: {exc}")
        return False
    _log(f"[af3] {report.summary()}")
    for failure in report.failed[:10]:
        _log(f"[af3]   failed: {failure}")
    return report.ok


def score_and_sort(stage: Path, experiment: Optional[str],
                   max_rmsd: float, min_fraction: float) -> bool:
    report = run_scoring(stage, experiment, max_rmsd, min_fraction)
    _log(f"[score] {report.summary()}")
    for table in report.tables:
        _log(f"[score] results -> {table}")
    for problem in report.problems[:10]:
        _log(f"[score] {problem}")
    return report.ok


def hand_over(stage: Path, experiment: Optional[str]) -> bool:
    _log("[transfer] handing the passed folds to stage 05...")
    try:
        report = run_transfer(stage, experiment)
    except (OSError, TransferError) as exc:
        _log(f"[transfer] FAILED: {exc}")
        return False
    _log(f"[transfer] {report.summary()}")
    return report.ok


def tidy(stage: Path, experiment: Optional[str], dry_run: bool) -> bool:
    report = run_cleanup(stage, experiment, dry_run)
    _log(f"[clean] {report.summary()}")
    return True


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=Path, default=STAGE_ROOT)
    parser.add_argument("--experiment", default=None,
                        help="only this experiment (default: every outstanding one)")
    parser.add_argument("--max-rmsd", type=float, default=5.0,
                        help="pass below this RMSD (default: 5.0)")
    parser.add_argument("--min-fraction", type=float, default=0.8,
                        help="pass above this matched fraction (default: 0.8)")
    parser.add_argument("--max-concurrent", type=int, default=4,
                        help="cluster jobs in flight (default: 4, the account cap)")
    parser.add_argument("--no-fold", action="store_true",
                        help="skip AlphaFold3; score whatever is already folded")
    parser.add_argument("--no-score", action="store_true",
                        help="fold only; leave scoring and sorting for later")
    parser.add_argument("--no-transfer", action="store_true",
                        help="do not hand anything to stage 05")
    parser.add_argument("--no-clean", action="store_true",
                        help="keep outputs/, af3_runs/ and the input jsons")
    parser.add_argument("--clean-dry-run", action="store_true",
                        help="report what cleanup would remove, remove nothing")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    stage = args.stage.resolve()

    if not jp.inputs_root(stage).is_dir():
        _log(f"[stage 04] nothing handed over yet: {jp.inputs_root(stage)} does not exist")
        return 0

    prepared_ok = prepare_jsons(stage, args.experiment)

    folded_ok = True
    if not args.no_fold:
        if shutil.which("sbatch") is None:
            _log("[af3] no sbatch on PATH -- folding in place, one job at a time")
        folded_ok = fold_sequences(stage, args.experiment, args.max_concurrent)

    scored_ok = True
    if not args.no_score:
        scored_ok = score_and_sort(stage, args.experiment, args.max_rmsd, args.min_fraction)

    # Runs even with --no-score: scoring may have happened on an earlier run,
    # and transferring what already passed should not need a rescore.
    transferred_ok = True
    if not args.no_transfer:
        transferred_ok = hand_over(stage, args.experiment)

    # Last, and only after the transfer: stage 05 has taken its copy by then,
    # and cleanup refuses to touch anything not already inside an archive.
    if not args.no_clean:
        tidy(stage, args.experiment, args.clean_dry_run)

    return 0 if prepared_ok and folded_ok and scored_ok and transferred_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

