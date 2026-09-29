#!/usr/bin/env python3
"""
Runs stage 02 end to end:

1. Geometry-filters everything stage 01 handed over that has not been filtered
   yet -- aligns each design onto the seed it was generated against, carries
   the seed's polysaccharide across, and sorts on clashes and contacts;
2. Hands the passed structures to stage 03.

Both steps sweep every experiment rather than one, so a batch left behind by
an interrupted run is picked up by the next launch. Both skip whatever is
already done, so re-running costs a directory listing when there is nothing
new.

Usage:
    ./stage_02_launcher.py                      # everything outstanding
    ./stage_02_launcher.py --experiment NAME    # just this one
    ./stage_02_launcher.py --no-transfer        # filter only

Exits non-zero if any structure could not be evaluated or moved on, so it can
be used from a wrapper script.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_ROOT = SCRIPT_DIR.parent
HELPING_SCRIPTS_DIR = SCRIPT_DIR / "helping_scripts"
# job_paths.py and the shared archive/transfer machinery live at the repo root.
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

from cleanup import run_cleanup  # noqa: E402
from archives import ArchiveError  # noqa: E402
from geometry_filter import GeometryError, run_geometry_filter  # noqa: E402
from transfer_to_stage03 import TransferError, run_transfer  # noqa: E402


def _log(*parts: object) -> None:
    print(*parts, flush=True)


def filter_structures(stage: Path, experiment: Optional[str]) -> bool:
    """Align and sort everything not already in a results table."""
    _log("[geometry] sorting structures into passed/rejected...")
    try:
        report = run_geometry_filter(stage, experiment)
    except (OSError, ArchiveError, GeometryError) as exc:
        _log(f"[geometry] FAILED: {exc}")
        _log("[geometry] nothing has been lost -- fix the cause and re-run "
             "scripts/helping_scripts/geometry_filter.py")
        return False

    _log(f"[geometry] {report.summary()}")
    for table in report.tables:
        _log(f"[geometry] results -> {table}")
    skipped = report.counts.get("SKIPPED", 0)
    if skipped:
        _log(f"[geometry] {skipped} structure(s) skipped: their stage-01 seed has "
             f"no ligand chain, so there is nothing to check the design against")
    if not report.ok:
        _log(f"[geometry] {report.counts.get('ERROR', 0)} structure(s) could not be "
             f"evaluated -- see the skip_reason column")
    return report.ok


def transfer_structures(stage: Path, experiment: Optional[str]) -> bool:
    """Hand every passed structure not already in stage 03 across to it.

    Runs whatever the filter reported: a SKIPPED or ERROR structure never
    reaches passed/, and the archive merge is atomic, so a filter problem
    cannot leave a half-written source archive for this to read.
    """
    _log("[transfer] handing passed structures to stage 03...")
    try:
        report = run_transfer(stage, experiment)
    except (OSError, ArchiveError, TransferError) as exc:
        _log(f"[transfer] FAILED: {exc}")
        _log("[transfer] filtering is unaffected -- fix the cause and re-run "
             "scripts/helping_scripts/transfer_to_stage03.py")
        return False

    _log(f"[transfer] {report.summary()}")
    if report.incomplete:
        preview = ", ".join(report.incomplete[:5])
        more = f" (+{len(report.incomplete) - 5} more)" if len(report.incomplete) > 5 else ""
        _log(f"[transfer] {len(report.incomplete)} passed structure(s) could not be "
             f"transferred, missing half their file pair: {preview}{more}")
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
    parser.add_argument(
        "--stage", type=Path, default=STAGE_ROOT,
        help=f"stage root directory (default: {STAGE_ROOT})",
    )
    parser.add_argument(
        "--experiment", default=None,
        help="only process this one experiment (default: every outstanding one)",
    )
    parser.add_argument(
        "--no-filter", action="store_true",
        help="skip the geometry filter; transfer whatever already passed",
    )
    parser.add_argument(
        "--no-transfer", action="store_true",
        help="filter only; do not hand anything to stage 03",
    )
    parser.add_argument("--no-clean", action="store_true",
                        help="keep the raw files even once they are archived")
    parser.add_argument("--clean-dry-run", action="store_true",
                        help="report what cleanup would remove, remove nothing")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    stage = args.stage.resolve()

    filtered_ok = True
    if not args.no_filter:
        filtered_ok = filter_structures(stage, args.experiment)

    transferred_ok = True
    if not args.no_transfer:
        transferred_ok = transfer_structures(stage, args.experiment)

    # Last, and only after the transfer: sorted_raw is cleared on the strength
    # of the archives, so those archives have to be complete first.
    if not args.no_clean:
        tidy(stage, args.experiment, args.clean_dry_run)

    return 0 if filtered_ok and transferred_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
