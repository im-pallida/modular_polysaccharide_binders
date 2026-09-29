#!/usr/bin/env python3
"""
Stage 01, last step: delete the raw structures the archives already hold.

    outputs_raw/<experiment>/<group>/<name>.cif   <- removed once inside
    outputs_raw/<experiment>/<group>/<name>.json     outputs_clean/<exp>/<group>.tar.gz
    outputs_raw/.tmp/                             <- scratch, removed
    sorted_raw/<experiment>/<outcome>/            <- removed once inside
                                                     sorted_clean/<exp>/<outcome>/<group>.tar.gz

This stage generates more than any other -- a 16k production run leaves 32k
loose files here -- and it was the only generating stage with no cleanup at
all, so they simply accumulated.

WHAT IS NOT TOUCHED, and why

    json/        the design configs. Stage 02 reads them to find each group's
                 seed, and stage 08 reads them to recover the fibre. They are
                 configuration, not output.
    job_runs/    fixed_residues_<experiment>.jsonl, which stages 02, 03 and 04
                 all read to learn what was held fixed. Deleting it would
                 strand three later stages.
    tables/      the results.

SAFE TO RUN, AND WHY IT IS NOW SAFE

run_one_job and run_cluster used to decide "already generated?" by looking for
the raw cif, so clearing outputs_raw meant diffusing the whole experiment
again. Both now check the group's archive as well, so a cleaned structure is
skipped rather than regenerated. That change landed with this file and the two
belong together: this script is not safe against an older stage 01.

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


def outputs_archive_for(stage: Path, path: Path) -> Optional[Path]:
    """outputs_raw/<experiment>/<group>/<file> -> its clean archive."""
    try:
        relative = path.relative_to(jp.raw_root(stage))
    except ValueError:
        return None
    parts = relative.parts
    if len(parts) != 3 or parts[0] == ".tmp":
        return None
    return jp.clean_archive_path(stage, parts[0], parts[1])


def sorted_archive_for(stage: Path, path: Path) -> Optional[Path]:
    """sorted_raw/<experiment>/<outcome>[/<group>]/<file> -> its sorted archive.

    The group is the directory when the filter routed into one, and otherwise
    has to come out of the file's own name -- which is how the filter files a
    structure it wrote directly into the outcome folder.
    """
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
    names = [experiment] if experiment else sorted(
        directory.name for directory in (jp.raw_root(stage)).glob("*")
        if directory.is_dir() and directory.name != ".tmp"
    )
    if not names:
        _log(f"[clean] nothing under {jp.raw_root(stage)}")

    for experiment_name in names:
        clean_tree(jp.raw_root(stage) / experiment_name,
                   lambda path: outputs_archive_for(stage, path),
                   report, dry_run, log=_log)
        clean_tree(stage / jp.SORTED_RAW_DIRNAME / experiment_name,
                   lambda path: sorted_archive_for(stage, path),
                   report, dry_run, log=_log)
        _log(f"[clean] {experiment_name}: {report.removed} cleared so far")

    # Interrupted jobs leave half-written work here; it is re-created on demand
    # and nothing reads it between runs.
    drop_tree(jp.raw_root(stage) / ".tmp", report, dry_run)
    remove_empty_dirs([jp.raw_root(stage), stage / jp.SORTED_RAW_DIRNAME],
                      report, dry_run)
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
