#!/usr/bin/env python3
"""
Runs stage 05 end to end:

1. Rebuilds each validated fold's ligand complex -- the AF3 monomer superposed
   onto both reference chains on the scaffolded residues, with the reference's
   polysaccharide kept where it is;
2. Redesigns the residues within 8 A of that polysaccharide with LigandMPNN,
   the two chains tied and the scaffolded residues held;
3. Keeps the three highest-confidence designs per complex;
4. Hands those to stage 06.

Needs LIGANDMPNN_ROOT pointing at the LigandMPNN checkout. The weights are
looked for at $LIGANDMPNN_ROOT/model_params/ligandmpnn_v_32_010_25.pt unless
LIGANDMPNN_CHECKPOINT says otherwise.

Usage:
    ./stage_05_launcher.py                      # everything outstanding
    ./stage_05_launcher.py --experiment NAME
    ./stage_05_launcher.py --no-design          # rebuild complexes only
    ./stage_05_launcher.py --cutoff 10.0        # a wider shell
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_ROOT = SCRIPT_DIR.parent
HELPING_SCRIPTS_DIR = SCRIPT_DIR / "helping_scripts"
COMMON_DIR = STAGE_ROOT.parent / "common"

sys.path.insert(0, str(COMMON_DIR))
sys.path.insert(0, str(HELPING_SCRIPTS_DIR))

import job_paths as jp  # noqa: E402
from ligand_prepare import run_prepare  # noqa: E402
from run_ligandmpnn import LigandMpnnError, run_design  # noqa: E402
from select_ligand_sequences import SelectionError, run_selection  # noqa: E402
from transfer_to_stage06 import TransferError, run_transfer  # noqa: E402


def _log(*parts: object) -> None:
    print(*parts, flush=True)


def build_complexes(stage: Path, experiment: Optional[str], cutoff: float,
                    reference_root: Optional[Path] = None,
                    chain_clash: float = 4.0, ligand_clash: float = 3.0) -> bool:
    report = run_prepare(stage, experiment, cutoff, reference_root,
                         chain_clash, ligand_clash)
    _log(f"[prepare] {report.summary()}")
    for problem in report.problems[:10]:
        _log(f"[prepare] {problem}")
    return report.ok


def redesign(stage: Path, experiment: Optional[str], num_seq: int) -> bool:
    try:
        report = run_design(stage, experiment, num_seq)
    except LigandMpnnError as exc:
        _log(f"[ligandmpnn] FAILED: {exc}")
        return False
    _log(f"[ligandmpnn] {report.summary()}")
    for problem in report.problems[:10]:
        _log(f"[ligandmpnn] {problem}")
    return report.ok


def select_best(stage: Path, experiment: Optional[str], keep: int) -> bool:
    try:
        report = run_selection(stage, experiment, keep)
    except (OSError, SelectionError) as exc:
        _log(f"[select] FAILED: {exc}")
        return False
    _log(f"[select] {report.summary()}")
    for table in report.tables:
        _log(f"[select] results -> {table}")
    return report.ok


def hand_over(stage: Path, experiment: Optional[str]) -> bool:
    _log("[transfer] handing the best designs to stage 06...")
    try:
        report = run_transfer(stage, experiment)
    except (OSError, TransferError) as exc:
        _log(f"[transfer] FAILED: {exc}")
        return False
    _log(f"[transfer] {report.summary()}")
    return report.ok


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=Path, default=STAGE_ROOT)
    parser.add_argument("--experiment", default=None,
                        help="only this experiment (default: every outstanding one)")
    parser.add_argument("--cutoff", type=float, default=8.0,
                        help="shell radius in angstroms (default: 8.0)")
    parser.add_argument("--keep", type=int, default=3,
                        help="designs to keep per complex (default: 3)")
    parser.add_argument("--num-seq", type=int, default=16,
                        help="designs to generate per complex (default: 16)")
    parser.add_argument("--chain-clash", type=float, default=4.0,
                        help="CA-CA between chains below this is a clash (default: 4.0)")
    parser.add_argument("--ligand-clash", type=float, default=3.0,
                        help="protein CA to ligand atom below this is a clash (default: 3.0)")
    parser.add_argument("--reference-archives", type=Path, default=None,
                        help="a directory holding passed/<group>.tar.gz from an "
                             "archived stage-02 run, for structures the current "
                             "stage 02 never processed")
    parser.add_argument("--no-design", action="store_true",
                        help="build the complexes only; skip LigandMPNN")
    parser.add_argument("--no-transfer", action="store_true",
                        help="do not hand anything to stage 06")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    stage = args.stage.resolve()

    if not jp.inputs_root(stage).is_dir():
        _log(f"[stage 05] nothing handed over yet: {jp.inputs_root(stage)} does not exist")
        return 0

    prepared_ok = build_complexes(
        stage, args.experiment, args.cutoff,
        args.reference_archives.expanduser().resolve() if args.reference_archives else None,
        args.chain_clash, args.ligand_clash,
    )

    designed_ok = True
    selected_ok = True
    if not args.no_design:
        designed_ok = redesign(stage, args.experiment, args.num_seq)
        selected_ok = select_best(stage, args.experiment, args.keep)

    transferred_ok = True
    if not args.no_transfer and not args.no_design:
        transferred_ok = hand_over(stage, args.experiment)

    return 0 if prepared_ok and designed_ok and selected_ok and transferred_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

