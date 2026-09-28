#!/usr/bin/env python3
"""
Hand stage 05's prepared pairs to stage 06.

    sorted_clean/<experiment>/passed/<group>.tar.gz
    ../06_rfd3_linker_generation/inputs/<experiment>/<group>.tar.gz

The mechanics are shared with every other stage. Only two things are local: the
destination, and what one unit's files look like -- here the stripped pair RFD3
diffuses on, the fibril it must not see, and the two jsons whose linker length
you fill in.

Everything passes. Stage 05 filters nothing, so this hands on every pair it
built; the alignment table records how close each one's copies came without
turning any of them away.
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


def pair_members(member_names: Sequence[str]) -> Dict[str, List[str]]:
    """{sequence_id: [<sequence_id>.pdb]}.

    One file per sequence and a flat archive, so the key is the filename's stem
    rather than a directory. Members that are not pdbs are ignored by name:
    archives written before the jsons moved out still carry them, and those must
    not be dragged into stage 06 on top of a length already set.
    """
    grouped: Dict[str, List[str]] = {}
    for name in member_names:
        cleaned = name[2:] if name.startswith("./") else name
        if not cleaned.endswith(".pdb"):
            continue
        grouped.setdefault(Path(cleaned).stem, []).append(name)
    return grouped


def run_transfer(stage: Path, experiment: Optional[str] = None) -> TransferReport:
    return _run_transfer(stage, jp.stage06_archive_path, LABEL, experiment, pair_members)


def main(argv: Optional[List[str]] = None) -> int:
    return cli(jp.stage06_archive_path, LABEL, STAGE, __doc__, argv, pair_members)


if __name__ == "__main__":
    raise SystemExit(main())
