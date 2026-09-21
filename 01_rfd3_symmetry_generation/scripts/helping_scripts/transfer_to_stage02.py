#!/usr/bin/env python3
"""
Hand stage 01's passed structures to stage 02.
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
LABEL = "stage 02"


def run_transfer(stage: Path, experiment: Optional[str] = None) -> TransferReport:
    return _run_transfer(stage, jp.stage02_archive_path, LABEL, experiment)


def main(argv: Optional[List[str]] = None) -> int:
    return cli(jp.stage02_archive_path, LABEL, STAGE, __doc__, argv)


if __name__ == "__main__":
    raise SystemExit(main())

