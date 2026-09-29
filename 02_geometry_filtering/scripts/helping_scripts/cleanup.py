#!/usr/bin/env python3
"""
Stage 02, last step: delete the routed structures the archives already hold.

    sorted_raw/<experiment>/<outcome>/<group>/<protein_id>.pdb   <- removed
    sorted_raw/<experiment>/<outcome>/<group>/<protein_id>.json      once inside
    sorted_raw/<experiment>/_scratch/                             <- scratch
                       sorted_clean/<experiment>/<outcome>/<group>.tar.gz

The filter writes every structure it judges into sorted_raw and then folds them
into per-group tarballs. Nothing removed the loose copies afterwards, so each
run left a second copy of everything it sorted.

WHAT IS NOT TOUCHED

    inputs/        stage 01's handover. Stage 02 re-reads it on a re-run.
    sorted_clean/  the archives themselves.
    tables/        the results.

Nothing downstream reads sorted_raw: stage 03 is handed its structures through
inputs/, from the archives.

Usage:
    ./cleanup.py --dry-run
    ./cleanup.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import job_paths as jp  # noqa: E402
from raw_cleanup import CleanReport, clean_tree, drop_tree, remove_empty_dirs  # noqa: E402

STAGE = Path(__file__).resolve().parents[2]


def _log(*parts: object) -> None:
    print(*parts, flush=True)


def archive_for(stage: Path, path: Path) -> Optional[Path]:
    """sorted_raw/<experiment>/<outcome>/<group>/<file> -> its archive."""
    root = stage / jp.SORTED_RAW_DIRNAME
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    parts = relative.parts
    if len(parts) < 3 or parts[1] not in jp.OUTCOMES:
        return None
    if len(parts) == 4:
        group_key = parts[2]
    else:
        try:
            group_key = jp.group_key_from_job_name(path.stem)
        except (ValueError, AttributeError):
            return None
    return jp.sorted_archive_path(stage, parts[0], parts[1], group_key)


def run_cleanup(stage: Path, experiment: Optional[str] = None,
                dry_run: bool = False) -> CleanReport:
    report = CleanReport()
    root = stage / jp.SORTED_RAW_DIRNAME
    names = [experiment] if experiment else sorted(
        directory.name for directory in root.glob("*") if directory.is_dir()
    )
    if not names:
        _log(f"[clean] nothing under {root}")
    for experiment_name in names:
        clean_tree(root / experiment_name,
                   lambda path: archive_for(stage, path),
                   report, dry_run, log=_log)
        drop_tree(jp.sorted_scratch_dir(stage, experiment_name), report, dry_run)
        _log(f"[clean] {experiment_name}: {report.removed} cleared so far")
    remove_empty_dirs([root], report, dry_run)
    return report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would go; delete nothing")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    report = run_cleanup(args.stage.resolve(), args.experiment, args.dry_run)
    _log(f"[done] {report.summary()}")
    for path in report.unarchived[:10]:
        _log(f"[done]   kept (not in any archive): {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
