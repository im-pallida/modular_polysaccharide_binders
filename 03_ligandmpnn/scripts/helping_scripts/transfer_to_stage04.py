#!/usr/bin/env python3
"""
Hand stage 03's best sequences to stage 04.

    sorted_clean/<experiment>/passed/<group>.tar.gz
    ../04_alphafold/inputs/<experiment>/<group>.tar.gz

The mechanics are shared with every other stage. Only two things are local: the
destination, and what one protein's files look like -- here the sequences kept
for it plus the complex they were designed on, which is the shape stage 04's
json builder reads.

The complex travels because stage 04 needs the fibril to fold against and the
backbone to measure against, and neither is recoverable from a fasta. Carrying
it means stage 03's inputs_prepared/ can be cleaned without stranding stage 04.
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
LABEL = "stage 04"


def design_members(member_names: Sequence[str]) -> Dict[str, List[str]]:
    """{protein_id: [<sequence_id>.fa ..., <pid>.pdb, <pid>_redesign.json]}.

    A protein moves with every sequence kept for it and with the complex those
    sequences were designed on. There is no completeness rule to apply: whatever
    survived selection is what stage 04 should fold.
    """
    grouped: Dict[str, List[str]] = {}
    for name in member_names:
        if "/" in name and name.endswith((".fa", ".pdb", ".json")):
            grouped.setdefault(name.split("/")[0], []).append(name)
    return grouped


def run_transfer(stage: Path, experiment: Optional[str] = None) -> TransferReport:
    return _run_transfer(stage, jp.stage04_archive_path, LABEL, experiment, design_members)


def main(argv: Optional[List[str]] = None) -> int:
    return cli(jp.stage04_archive_path, LABEL, STAGE, __doc__, argv, design_members)


if __name__ == "__main__":
    raise SystemExit(main())
