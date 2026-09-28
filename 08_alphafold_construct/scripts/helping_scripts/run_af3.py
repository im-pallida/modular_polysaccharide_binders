#!/usr/bin/env python3
"""
Stage 08, step 2: fold every prepared json, holo and apo.

    job_runs/af3/<experiment>/<group>/<sequence_id>_holo.json  <- read
    job_runs/af3/<experiment>/<group>/<sequence_id>_apo.json   <- read
    outputs/<sequence_id>_holo/  outputs/<sequence_id>_apo/    <- written

Two passes over the same dispatcher, one per suffix, rather than one pass over
a mixed queue. The dispatcher's canary runs a single job before the batch, and
a canary is only worth anything if it is the same KIND of job as the rest: a
holo fold that works says nothing about whether the apo jsons are well formed.
One canary each.

Everything else -- four in flight, sbatch --wait, skip anything already folded
-- is common/af3_dispatch.py, the same code stage 04 folds with.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import job_paths as jp  # noqa: E402
from af3_dispatch import MAX_CONCURRENT, Af3Error, RunReport, run_af3 as _run_af3

STAGE = Path(__file__).resolve().parents[2]

__all__ = ["Af3Error", "RunReport", "MAX_CONCURRENT", "run_af3"]


def _log(*parts: object) -> None:
    print(*parts, flush=True)


def run_af3(stage: Path, experiment: Optional[str] = None,
            max_concurrent: int = MAX_CONCURRENT) -> RunReport:
    combined = RunReport()
    for suffix in jp.FOLD_SUFFIXES:
        _log(f"[af3] {suffix} fold(s)...")
        report = _run_af3(stage, experiment, max_concurrent, suffix)
        _log(f"[af3] {suffix}: {report.summary()}")
        combined.submitted += report.submitted
        combined.succeeded += report.succeeded
        combined.skipped += report.skipped
        combined.failed.extend(report.failed)
    return combined


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT,
                        help=f"cluster jobs in flight (default: {MAX_CONCURRENT})")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = run_af3(args.stage.resolve(), args.experiment, args.max_concurrent)
    except Af3Error as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    _log(f"[done] {report.summary()}")
    for failure in report.failed[:10]:
        _log(f"[done]   failed: {failure}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
