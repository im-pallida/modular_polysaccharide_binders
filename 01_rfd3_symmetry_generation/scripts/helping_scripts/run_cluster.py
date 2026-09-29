#!/usr/bin/env python3
"""
Stage 01's half of the cluster dispatch: the runner, and this stage's sampler.

Everything else -- the canary, the submission, the polling, the failure
diagnosis, the archive-aware output check -- is in common/rfd3_dispatch.py,
because stage 06 diffuses too and there is no version of that worth maintaining
twice. This file is the same shape as 04_alphafold/.../run_af3.py, which sits
over common/af3_dispatch.py for the same reason.

What is stage 01's and stays here:

    RUNNER      run_one_job.py, which lives in this stage and is used by both
    OVERRIDES   the SYMMETRY sampler. Stage 06 must not inherit it: a linker
                runs from one copy's C-terminus to the other's N-terminus and
                no symmetry operation maps that onto itself.

Imported and called by the launcher; not useful on its own.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import rfd3_dispatch  # noqa: E402

from run_one_job import DispatchReport, RFD3_SAMPLER_OVERRIDES  # noqa: E402

RUNNER = Path(__file__).resolve().parent / "run_one_job.py"

# Re-exported so the module keeps its old surface: the launcher and the tests
# import these names from here.
SBATCH_SCRIPT = rfd3_dispatch.SBATCH_SCRIPT
POLL_INTERVAL_S = rfd3_dispatch.POLL_INTERVAL_S
MAX_CONSECUTIVE_SQUEUE_ERRORS = rfd3_dispatch.MAX_CONSECUTIVE_SQUEUE_ERRORS
# Was private here before the move; still imported by name elsewhere.
_missing_outputs = rfd3_dispatch.missing_outputs

__all__ = ["dispatch_cluster", "DispatchReport", "RUNNER", "SBATCH_SCRIPT"]


def dispatch_cluster(experiment: str, run_list: Path, stage: Path,
                     overrides: Sequence[str] = RFD3_SAMPLER_OVERRIDES) -> DispatchReport:
    return rfd3_dispatch.dispatch_cluster(experiment, run_list, stage, RUNNER, overrides)
