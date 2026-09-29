#!/usr/bin/env python3
"""
Stage 01's half of the workstation dispatch: the runner, and this stage's sampler.

The loop itself is in common/rfd3_dispatch.py -- see run_cluster.py beside this
for why. Imported and called by the launcher; not useful on its own.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import rfd3_dispatch  # noqa: E402

from run_one_job import DispatchReport, RFD3_SAMPLER_OVERRIDES  # noqa: E402

RUNNER = Path(__file__).resolve().parent / "run_one_job.py"

__all__ = ["dispatch_workstation", "DispatchReport", "RUNNER"]


def dispatch_workstation(experiment: str, run_list: Path, stage: Path,
                         overrides: Sequence[str] = RFD3_SAMPLER_OVERRIDES) -> DispatchReport:
    return rfd3_dispatch.dispatch_workstation(experiment, run_list, stage, RUNNER, overrides)
