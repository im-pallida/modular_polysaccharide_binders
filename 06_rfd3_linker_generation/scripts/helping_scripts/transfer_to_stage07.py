#!/usr/bin/env python3
"""
Hand stage 06's generated backbones to stage 07.

    outputs_clean/<experiment>/<group>.tar.gz
    ../07_proteinmpnn_linker/inputs/<experiment>/<group>.tar.gz

The mechanics are shared with every other stage. Only the destination is local.

A design moves only with BOTH halves -- the cif and its metadata json -- which
is the shared rule and happens to be exactly right here: the json carries
diffused_index_map, and without it stage 07 cannot tell the linker from the two
copies it must not touch. A cif that arrived alone would be redesigned end to
end, which is the opposite of what stage 07 is for, and it would look like it
worked.

Nothing is filtered on the way. Stage 06 generates; judging a linker is stage
08's job, once it has been given a sequence and folded.
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
LABEL = "stage 07"


def run_transfer(stage: Path, experiment: Optional[str] = None) -> TransferReport:
    return _run_transfer(stage, jp.stage07_archive_path, LABEL, experiment)


def main(argv: Optional[List[str]] = None) -> int:
    return cli(jp.stage07_archive_path, LABEL, STAGE, __doc__, argv)


if __name__ == "__main__":
    raise SystemExit(main())
