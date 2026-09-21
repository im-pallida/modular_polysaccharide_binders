#!/usr/bin/env python3
"""
Hand stage 04's validated folds to stage 05.

    sorted_clean/<experiment>/passed/<group>.tar.gz
    ../05_ligandmpnn/inputs/<experiment>/<group>.tar.gz

The mechanics are shared with every other stage. Only two things are local: the
destination, and what one unit's files look like -- here a whole AF3 job folder
per sequence rather than a json and a structure, because stage 05 needs both the
model and the ranking file that says which sample to trust.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import job_paths as jp  # noqa: E402
from transfer import (  # noqa: E402
    TransferError,
    TransferReport,
    cli,
    run_transfer as _run_transfer,
)

STAGE = Path(__file__).resolve().parents[2]
LABEL = "stage 05"


def job_members(member_names: Sequence[str]) -> Dict[str, List[str]]:
    """{protein_id: [<protein_id>/<sequence_id>_monomer/..., ...]}.

    Keyed on protein_id, like every other transfer, so a protein moves as a
    unit. Everything AF3 wrote travels: stage 05 re-picks the top-ranked sample
    from the archive rather than trusting a filename.
    """
    grouped: Dict[str, List[str]] = {}
    for name in member_names:
        parts = name.split("/")
        if len(parts) > 2 and parts[1].endswith("_monomer"):
            grouped.setdefault(parts[0], []).append(name)
    return grouped


def run_transfer(stage: Path, experiment: Optional[str] = None) -> TransferReport:
    return _run_transfer(stage, jp.stage05_archive_path, LABEL, experiment, job_members)


def main(argv: Optional[List[str]] = None) -> int:
    return cli(jp.stage05_archive_path, LABEL, STAGE, __doc__, argv, job_members)


if __name__ == "__main__":
    raise SystemExit(main())

