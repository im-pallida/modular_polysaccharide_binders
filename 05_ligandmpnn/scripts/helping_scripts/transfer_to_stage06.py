#!/usr/bin/env python3
"""
Hand stage 05's redesigns to stage 06.

    sorted_clean/<experiment>/passed/<group>.tar.gz
    ../06_rfd3_linker/inputs/<experiment>/<group>.tar.gz

Keyed on the stage-04 sequence the redesigns came from, so a complex moves with
every design kept for it.
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
LABEL = "stage 06"


def _components(name: str) -> list:
    """Path components, with the leading './' that `tar czf ... -C dir .` writes
    normalised away. archives.py does the same on the way in; a member rule that
    skips it sees every component shifted by one and matches nothing."""
    cleaned = name[2:] if name.startswith("./") else name
    return cleaned.rstrip("/").split("/")


def design_members(member_names: Sequence[str]) -> Dict[str, List[str]]:
    """{sequence_id: [<sequence_id>/<design_id>.fa, ..., <sequence_id>.pdb]}.

    The complex pdb travels with its designs: stage 06 reads the ligand out of
    it, and it is the reference anything downstream aligns against.
    """
    grouped: Dict[str, List[str]] = {}
    for name in member_names:
        parts = _components(name)
        if len(parts) >= 2 and parts[-1].endswith((".fa", ".pdb")):
            grouped.setdefault(parts[0], []).append(name)
    return grouped


def run_transfer(stage: Path, experiment: Optional[str] = None) -> TransferReport:
    return _run_transfer(stage, jp.stage06_archive_path, LABEL, experiment, design_members)


def main(argv: Optional[List[str]] = None) -> int:
    return cli(jp.stage06_archive_path, LABEL, STAGE, __doc__, argv, design_members)


if __name__ == "__main__":
    raise SystemExit(main())

