#!/usr/bin/env python3
"""
Runs stage 05 end to end:

1. Superposes each validated fold onto BOTH protein chains of its reference
   complex, on the scaffolded residues, then drops the cellulose -- so one
   sequence ends up occupying both seed sides and the fibre, having done its
   job, does not survive into the output;
2. Records the fit and how close the copies came, to each other and to the
   fibril. NOTHING IS REJECTED on those numbers: see align_pairs.py for why;
3. Writes two RFD3 jsons per pair, A-linker-B and B-linker-A, with the linker
   length left as the literal LINKER for you, straight into stage 06;
4. Hands the pdb to stage 06 in a tar -- one file per sequence, nothing else;
5. Clears the scratch and the loose pdbs, which by then are in the archive.
   --keep-pairs leaves them if you would rather not extract to look.

Nothing here needs a GPU: superposition and distance arithmetic, so a whole
experiment runs in seconds and re-running costs nothing.

Usage:
    ./stage_05_launcher.py                      # everything outstanding
    ./stage_05_launcher.py --experiment NAME
    ./stage_05_launcher.py --no-transfer
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
from align_pairs import (  # noqa: E402
    LIGAND_CLASH,
    PROTEIN_CLASH,
    AlignReport,
    run_align,
)
from cleanup import run_cleanup  # noqa: E402
from linker_jsons import LinkerReport, run_build  # noqa: E402
from transfer_to_stage06 import TransferError, run_transfer  # noqa: E402


def _log(*parts: object) -> None:
    print(*parts, flush=True)


def align(stage: Path, experiment: Optional[str], protein_clash: float,
          ligand_clash: float) -> AlignReport:
    _log("[align] placing each fold on both seed sides...")
    report = run_align(stage, experiment, protein_clash, ligand_clash)
    _log(f"[align] {report.summary()}")
    for problem in report.problems[:10]:
        _log(f"[align]   {problem}")
    return report


def write_jsons(stage: Path, experiment: Optional[str]) -> LinkerReport:
    _log("[linker] writing the RFD3 jsons...")
    report = run_build(stage, experiment)
    _log(f"[linker] {report.summary()}")
    for problem in report.problems[:10]:
        _log(f"[linker]   {problem}")
    return report


def transfer_pairs(stage: Path, experiment: Optional[str]) -> bool:
    _log("[transfer] handing the prepared pairs to stage 06...")
    try:
        report = run_transfer(stage, experiment)
    except (OSError, TransferError) as exc:
        _log(f"[transfer] FAILED: {exc}")
        return False
    _log(f"[transfer] {report.summary()}")
    return report.ok


def closing_message(stage: Path, experiment: Optional[str],
                    report: AlignReport, linker: LinkerReport) -> None:
    """What to do next, which is the one part of this pipeline a person decides.

    An empty input and a stage that failed on every pair both end with nothing
    ready, and they are opposite problems -- one means stage 04 has not handed
    over, the other means the pairs could not be built. Reporting the alarming
    one for the mundane one sends you looking in the wrong place, so they are
    told apart by what came IN rather than by what came out.
    """
    names = [experiment] if experiment else jp.inputs_experiments(stage)
    directories = [jp.pair_dir(stage, name, path.name[: -len(".tar.gz")])
                   for name in names
                   for path in sorted((jp.inputs_root(stage) / name).glob("*.tar.gz"))
                   if (jp.inputs_root(stage) / name).is_dir()]
    # Counted from the archives, because the loose pdbs are cleared by the time
    # this runs. Counting files on disk here reported zero and told you stage 04
    # had not handed anything over, which was the opposite of true.
    archived = sorted({jp.sorted_archive_path(stage, name, "passed",
                                              path.name[: -len(".tar.gz")])
                       for name in names
                       for path in sorted((jp.inputs_root(stage) / name).glob("*.tar.gz"))
                       if (jp.inputs_root(stage) / name).is_dir()})
    archived = [path for path in archived if path.is_file()]
    built = archived

    _log("")
    _log("=" * 72)
    if not directories:
        _log("  Nothing was handed over, so nothing was aligned.")
        _log("")
        _log(f"  Stage 05 reads {jp.inputs_root(stage)}")
        _log("  which stage 04 fills by running its transfer:")
        _log("      python3 04_alphafold/scripts/helping_scripts/transfer_to_stage05.py \\")
        _log("             --stage <project>/04_alphafold")
        _log("")
        _log("  If that reports nothing to move while stage 04's passed archives")
        _log("  clearly exist, they predate this pipeline: stage 05 needs the")
        _log("  '_ligand' folds, the reference complex and the redesign spec that")
        _log("  stages 03 and 04 now send along, and an older run carries none of")
        _log("  them. Such an experiment re-enters at stage 03.")
        _log("=" * 72)
        return

    if not built:
        _log("  Pairs were handed over but none could be built.")
        _log("")
        _log("  This stage rejects nothing, so every one of these is a failure")
        _log("  rather than a verdict. The reasons are above; the usual ones are a")
        _log("  reference with other than two protein chains, a residue count that")
        _log("  does not match the fold, or a missing redesign spec.")
        _log("=" * 72)
        return

    _log(f"  {len(archived)} aligned pair(s) are ready, {linker.jsons} json(s) written.")
    if report.touching:
        _log("")
        _log(f"  {report.touching} pair(s) have heavy atoms in contact. NONE were")
        _log("  rejected -- AlphaFold folds slightly differently from the design, so")
        _log("  that number is partly its deviation. The columns are in")
        _log(f"    {jp.alignment_table_path(stage, names[0]) if names else '<tables>'}")
        _log("  sort on closest_chain and closest_ligand if you want to look at the")
        _log("  worst before spending GPU time on them.")
    _log("")
    _log("  TO USE THE JSON FILES FROM STAGE 06, INSPECT THE STRUCTURES AND ENTER")
    _log("  THE LENGTHS OF THE LINKERS.")
    _log("")
    _log("  The pairs are in the archives, not loose -- two protein copies each,")
    _log("  no cellulose. To look at one:")
    for path in archived[:2]:
        _log(f"    tar xzf {path} -C /tmp && ls /tmp/*.pdb")
        break
    _log("")
    _log("  How far each one has to reach:")
    _log(f"    {jp.linker_span_table_path(stage, names[0]) if names else '<tables>'}")
    _log("  min_residues is a floor at 3.4 A per residue fully extended -- a linker")
    _log("  that length is a taut string, so allow slack above it.")
    _log("")
    _log("  Then set \"length\" in each json under")
    _log(f"    {jp.stage06_root(stage) / 'json'}")
    _log("  Two jsons per pair: one linking A to B, one linking B to A. They are")
    _log("  different questions -- the termini are not the same distance apart in")
    _log("  both directions -- so fill in both unless you have ruled one out.")
    _log("=" * 72)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=Path, default=STAGE_ROOT)
    parser.add_argument("--experiment", default=None,
                        help="only this experiment (default: every outstanding one)")
    parser.add_argument("--protein-clash", type=float, default=PROTEIN_CLASH,
                        help=f"heavy-atom distance between the two copies below "
                             f"which a contact is recorded -- recorded only, "
                             f"nothing is rejected (default: {PROTEIN_CLASH})")
    parser.add_argument("--ligand-clash", type=float, default=LIGAND_CLASH,
                        help=f"the same, to the fibril (default: {LIGAND_CLASH})")
    parser.add_argument("--no-jsons", action="store_true",
                        help="align only; do not write the RFD3 jsons")
    parser.add_argument("--no-transfer", action="store_true",
                        help="do not hand anything to stage 06")
    parser.add_argument("--no-clean", action="store_true",
                        help="keep the scratch directories")
    parser.add_argument("--keep-pairs", action="store_true",
                        help="leave the loose pdbs in inputs_prepared/ instead of "
                             "removing the ones already in the archive")
    parser.add_argument("--clean-dry-run", action="store_true",
                        help="report what cleanup would remove, remove nothing")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    stage = args.stage.resolve()

    report = align(stage, args.experiment, args.protein_clash, args.ligand_clash)
    aligned_ok = report.ok

    linker = LinkerReport()
    if not args.no_jsons:
        linker = write_jsons(stage, args.experiment)

    transferred_ok = True
    if not args.no_transfer and not args.no_jsons:
        transferred_ok = transfer_pairs(stage, args.experiment)

    if not args.no_clean:
        cleaned = run_cleanup(stage, args.experiment, args.clean_dry_run,
                              clean_pairs=not args.keep_pairs)
        _log(f"[clean] {cleaned.summary()}")

    closing_message(stage, args.experiment, report, linker)
    return 0 if aligned_ok and linker.ok and transferred_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
