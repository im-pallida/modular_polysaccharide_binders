#!/usr/bin/env python3
"""
Runs stage 06 end to end:

1. Lays out the pairs stage 05 handed over;
2. Runs RFD3 on every json whose linker length is set, DESIGNS_PER_JSON
   backbones each. A json still holding LINKER is an ERROR -- reported, and the
   run exits non-zero -- but it does not stop the filled ones from generating;
3. Files the backbones back into their source group's archive;
4. Hands them to stage 07, each cif with the metadata json that says which
   residues are the linker;
5. Clears what the archives already hold.

The jsons are written by stage 05, straight into this stage's json/ directory,
along with the table of how far each linker has to reach. Set the lengths there
and run this; nothing here rewrites a json.

    ../05_ligand_alignment/tables/stage_05_pairs_<experiment>.csv   how far
    ./helping_scripts/set_linker.py --length 14
    ./helping_scripts/set_linker.py --length 12-18 --orientation AB
    ./helping_scripts/run_rfd3.py --list          # what is ready, run nothing

Usage:
    ./stage_06_launcher.py
    ./stage_06_launcher.py --experiment NAME
    ./stage_06_launcher.py --designs 8
    ./stage_06_launcher.py --no-generate          # lay out only
"""
from __future__ import annotations

import argparse
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
from archive_designs import run_archive  # noqa: E402
from cleanup import run_cleanup  # noqa: E402
from transfer_to_stage07 import TransferError, run_transfer  # noqa: E402
from unpack_inputs import run_unpack  # noqa: E402
from run_rfd3 import DESIGNS_PER_JSON, Rfd3Error, run_rfd3  # noqa: E402


def _log(*parts: object) -> None:
    print(*parts, flush=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=Path, default=STAGE_ROOT)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--designs", type=int, default=DESIGNS_PER_JSON,
                        help=f"backbones per json, on consecutive seeds "
                             f"(default: {DESIGNS_PER_JSON})")
    parser.add_argument("--no-generate", action="store_true",
                        help="lay the inputs out but do not run RFD3")
    parser.add_argument("--no-transfer", action="store_true",
                        help="do not hand anything to stage 07")
    parser.add_argument("--no-clean", action="store_true",
                        help="keep outputs_raw and the laid-out pdbs")
    parser.add_argument("--clean-dry-run", action="store_true",
                        help="report what cleanup would remove, remove nothing")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    stage = args.stage.resolve()
    report = run_unpack(stage, args.experiment)
    _log(f"[unpack] {report.summary()}")
    for problem in report.problems[:10]:
        _log(f"[unpack]   {problem}")

    generated_ok = True
    if not args.no_generate:
        _log("")
        try:
            run = run_rfd3(stage, args.experiment, args.designs)
        except Rfd3Error as exc:
            _log(f"[rfd3] {exc}")
            return 1
        _log(f"[rfd3] {run.summary()}")
        for failure in run.failed[:10]:
            _log(f"[rfd3]   {failure}")
        for unfilled in run.unfilled[:10]:
            _log(f"[rfd3]   {unfilled}")
        generated_ok = run.ok
        if run.unfilled:
            _log("")
            _log("=" * 72)
            _log(f"  {len(run.unfilled)} json(s) still hold LINKER and did NOT run.")
            _log("")
            _log("  Everything else generated; this is not a failed batch, it is a")
            _log("  list of the ones still waiting on you. They are reported rather")
            _log("  than skipped so an unset length cannot pass for a finished pair.")
            _log("")
            _log("  How far each has to reach:")
            _log(f"    {jp.stage05_root(stage) / 'tables'}")
            _log("  min_residues is a floor at 3.4 A per residue fully extended -- a")
            _log("  linker that length is a taut string, so allow slack above it.")
            _log("")
            _log("  Then:")
            _log(f"    {SCRIPT_DIR / 'helping_scripts' / 'set_linker.py'} --length 14")
            _log("  and run this launcher again. What already generated is archived,")
            _log("  so a second run picks up only the new ones.")
            _log("=" * 72)

    # Archiving runs even when generation reported problems: the designs that
    # DID come out are finished work, and leaving them loose because a sibling
    # json was unfilled would mean the next run had to redo them.
    archive = run_archive(stage, args.experiment)
    _log(f"[archive] {archive.summary()}")
    for entry in (archive.incomplete + archive.unplaced)[:10]:
        _log(f"[archive]   {entry}")

    transferred_ok = True
    if not args.no_transfer:
        _log("")
        try:
            moved = run_transfer(stage, args.experiment)
        except (OSError, TransferError) as exc:
            _log(f"[transfer] FAILED: {exc}")
            transferred_ok = False
        else:
            _log(f"[transfer] {moved.summary()}")
            transferred_ok = moved.ok

    if not args.no_clean:
        cleaned = run_cleanup(stage, args.experiment, args.clean_dry_run)
        _log(f"[clean] {cleaned.summary()}")

    return 0 if (report.ok and generated_ok and archive.ok
                 and transferred_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
