#!/usr/bin/env python3
"""
Hand stage 02's passed structures to stage 03.

    sorted_clean/<experiment>/passed/<group>.tar.gz
    ../03_ligandmpnn/inputs/<experiment>/<group>.tar.gz

The mechanics -- incremental, no transfer log, cumulative destination
archives, both halves of a file pair or neither -- live in common/transfer.py
and are shared with every other stage. All this file supplies is where the
output goes.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import job_paths as jp  # noqa: E402
from transfer import (  # noqa: E402
    TransferError,
    TransferReport,
    cli,
    run_transfer as _run_transfer,
)

STAGE = Path(__file__).resolve().parents[2]
LABEL = "stage 03"


def run_transfer(stage: Path, experiment: Optional[str] = None) -> TransferReport:
    return _run_transfer(stage, jp.stage03_archive_path, LABEL, experiment)


def main(argv: Optional[List[str]] = None) -> int:
    return cli(jp.stage03_archive_path, LABEL, STAGE, __doc__, argv)


if __name__ == "__main__":
    raise SystemExit(main())
