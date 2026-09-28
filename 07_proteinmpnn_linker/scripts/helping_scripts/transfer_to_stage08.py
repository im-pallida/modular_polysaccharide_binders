#!/usr/bin/env python3
"""
Hand stage 07's best linker sequences to stage 08.

    sorted_clean/<experiment>/passed/<group>.tar.gz
    ../08_alphafold_construct/inputs/<experiment>/<group>.tar.gz

The mechanics are shared with every other stage. What travels is the sequences
kept for each construct and the RFD3 backbone they were designed onto, which is
what stage 08 measures its RMSD against.

No fibril. Stage 05 stopped setting one aside when both fibres were dropped
after the alignment, so a holo fold at stage 08 would have to regenerate it by
re-running stage 05, whose inputs still hold the reference complexes.
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
LABEL = "stage 08"


def construct_members(member_names: Sequence[str]) -> Dict[str, List[str]]:
    """{design: [<design>/<id>.fa ..., <design>.pdb]}."""
    grouped: Dict[str, List[str]] = {}
    for name in member_names:
        cleaned = name[2:] if name.startswith("./") else name
        if "/" in cleaned and cleaned.endswith((".fa", ".pdb")):
            grouped.setdefault(cleaned.split("/")[0], []).append(name)
    return grouped


def run_transfer(stage: Path, experiment: Optional[str] = None) -> TransferReport:
    return _run_transfer(stage, jp.stage08_archive_path, LABEL, experiment,
                         construct_members)


def main(argv: Optional[List[str]] = None) -> int:
    return cli(jp.stage08_archive_path, LABEL, STAGE, __doc__, argv, construct_members)


if __name__ == "__main__":
    raise SystemExit(main())
