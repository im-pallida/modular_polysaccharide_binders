#!/usr/bin/env python3
"""
Stage 04, step 2: fold every prepared json with AlphaFold3, monomer plus fibril.

    job_runs/af3/<experiment>/<group>/<sequence_id>_ligand.json   <- read
    outputs/<sequence_id>_ligand/                                 <- written

The dispatching itself -- one sbatch --wait per job, four in flight, a canary
before the batch, skip anything already folded -- lives in common/af3_dispatch.py.
This file is only the stage's half of that: the suffix, and a command line.

The suffix is '_ligand', not '_monomer', even though one chain is folded. It
names what is in the box rather than what is not, and it keeps these outputs
from being confused with the monomer-only folds of the previous pipeline, which
may still be sitting in outputs/ on a machine that ran it.

Blocking by design. A few hundred sequences is days of queue, so run it under
tmux or screen; interrupting and re-running is safe.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import job_paths as jp  # noqa: E402
from af3_dispatch import (  # noqa: E402
    MAX_CONCURRENT,
    Af3Error,
    RunReport,
    pending_jobs as _pending_jobs,
    run_af3 as _run_af3,
)

STAGE = Path(__file__).resolve().parents[2]

__all__ = ["Af3Error", "RunReport", "MAX_CONCURRENT", "pending_jobs", "run_af3"]


def pending_jobs(stage: Path, experiment: Optional[str],
                 report: RunReport) -> List:
    return _pending_jobs(stage, experiment, report, jp.LIGAND_SUFFIX)


def run_af3(stage: Path, experiment: Optional[str] = None,
            max_concurrent: int = MAX_CONCURRENT) -> RunReport:
    return _run_af3(stage, experiment, max_concurrent, jp.LIGAND_SUFFIX)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT,
                        help=f"cluster jobs in flight (default: {MAX_CONCURRENT}, "
                             f"the account cap)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = run_af3(args.stage.resolve(), args.experiment, args.max_concurrent)
    except Af3Error as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"[done] {report.summary()}", flush=True)
    for failure in report.failed[:10]:
        print(f"[done]   failed: {failure}", flush=True)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
