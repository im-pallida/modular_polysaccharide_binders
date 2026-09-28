#!/usr/bin/env python3
"""
Hand stage 04's validated folds to stage 05, which aligns them.

    sorted_clean/<experiment>/passed/<group>.tar.gz
    ../05_ligand_alignment/inputs/<experiment>/<group>.tar.gz

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


def _components(name: str) -> list:
    """Path components, with the leading './' that `tar czf ... -C dir .` writes
    normalised away. archives.py does the same on the way in; a member rule that
    skips it sees every component shifted by one and matches nothing."""
    cleaned = name[2:] if name.startswith("./") else name
    return cleaned.rstrip("/").split("/")


def job_members(member_names: Sequence[str]) -> Dict[str, List[str]]:
    """{protein_id: [<protein_id>/<sequence_id>_ligand/..., ...]}.

    Keyed on protein_id, like every other transfer, so a protein moves as a
    unit. Everything AF3 wrote travels: the next stage re-picks the top-ranked
    sample from the archive rather than trusting a filename.

    The suffix comes from job_paths, not a literal. It was spelled out here
    once, and when the fold gained its ligand and became '_ligand' this rule
    matched nothing at all and the transfer reported "nothing new" -- which
    reads exactly like success with an empty queue.
    """
    grouped: Dict[str, List[str]] = {}
    for name in member_names:
        parts = _components(name)
        # The reference complex sits at <protein_id>/<protein_id>.pdb, one
        # level shallower than the job folders, and stage 05 cannot align
        # without it.
        if len(parts) == 2 and parts[1].startswith(parts[0]) \
                and parts[1].endswith((".pdb", ".json")):
            grouped.setdefault(parts[0], []).append(name)
            continue
        if len(parts) <= 2 or not parts[1].endswith(f"_{jp.LIGAND_SUFFIX}"):
            continue
        # A member with no extension is a directory entry, not an AF3 output.
        # score_designs never writes those, but a tarball made by hand does.
        if not Path(parts[-1]).suffix:
            continue
        # The ORIGINAL name is kept: it is the key getmember() needs.
        grouped.setdefault(parts[0], []).append(name)
    return grouped


def run_transfer(stage: Path, experiment: Optional[str] = None) -> TransferReport:
    return _run_transfer(stage, jp.stage05_archive_path, LABEL, experiment, job_members)


def main(argv: Optional[List[str]] = None) -> int:
    return cli(jp.stage05_archive_path, LABEL, STAGE, __doc__, argv, job_members)


if __name__ == "__main__":
    raise SystemExit(main())
