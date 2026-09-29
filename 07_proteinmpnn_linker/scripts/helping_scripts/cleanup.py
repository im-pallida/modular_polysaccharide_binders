#!/usr/bin/env python3
"""
Stage 07, last step: delete the working copies the archives already hold.

    inputs_prepared/<experiment>/<group>/<design>.pdb  <- removed once inside
                     ../08_alphafold_construct/inputs/<experiment>/<group>.tar.gz
    job_runs/mpnn/<experiment>/<group>/out/            <- ProteinMPNN's raw
    job_runs/mpnn/<experiment>/<group>/_preflight/        fastas and the
    job_runs/mpnn/<experiment>/<group>/_scratch/          verification run
    job_runs/mpnn/<experiment>/<group>/parsed.jsonl    <- rebuilt every run
    job_runs/mpnn/<experiment>/<group>/assigned.jsonl

LINKER.JSONL IS KEPT, and this is the whole reason this file is careful.

Stage 08 reads job_runs/mpnn/<experiment>/<group>/linker.jsonl to learn which
residues are the linker -- it excludes them from the RMSD and splits the unit
sequence on them. Delete it and every fold at stage 08 fails with "stage 07
recorded no linker", after the folding has been paid for. It is small, it is
the record of what was designed and why, and it stays.

fixed_positions.jsonl and tied_positions.jsonl stay with it: nothing downstream
reads them, but they are a few kilobytes and they are the evidence for which
residues were opened up and which were tied to their twins.

The prepared backbones go because they have already travelled: the handover to
stage 08 carries each <design>.pdb into that stage's inputs archive, so the
copy here is the second one.

Usage:
    ./cleanup.py --dry-run
    ./cleanup.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import job_paths as jp  # noqa: E402
from raw_cleanup import CleanReport, clean_tree, drop_tree, remove_empty_dirs  # noqa: E402

STAGE = Path(__file__).resolve().parents[2]

# Read by stage 08, or too small and too explanatory to be worth reclaiming.
KEEP = ("linker.jsonl", "fixed_positions.jsonl", "tied_positions.jsonl")
SCRATCH = ("out", "_preflight", "_scratch")
REBUILT = ("parsed.jsonl", "assigned.jsonl")


def _log(*parts: object) -> None:
    print(*parts, flush=True)


def handover_archive_for(stage: Path, path: Path) -> Optional[Path]:
    """inputs_prepared/<experiment>/<group>/<design>.pdb -> stage 08's inputs.

    Verified against the NEXT stage's archive rather than this one's, because
    that is where this particular file went: stage 07's own outputs archive
    holds sequences, not backbones.
    """
    root = stage / "inputs_prepared"
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    parts = relative.parts
    if len(parts) != 3:
        return None
    return jp.stage08_archive_path(stage, parts[0], parts[1])


def group_dirs(stage: Path, experiment: str) -> List[Path]:
    root = stage / "job_runs" / "mpnn" / experiment
    return sorted(path for path in root.glob("*") if path.is_dir()) \
        if root.is_dir() else []


def run_cleanup(stage: Path, experiment: Optional[str] = None,
                dry_run: bool = False) -> CleanReport:
    report = CleanReport()
    prepared_root = stage / "inputs_prepared"
    names = [experiment] if experiment else jp.inputs_experiments(stage)
    if not names:
        _log(f"[clean] nothing handed over under {jp.inputs_root(stage)}")

    for experiment_name in names:
        clean_tree(prepared_root / experiment_name,
                   lambda path: handover_archive_for(stage, path),
                   report, dry_run, log=_log)
        for work_dir in group_dirs(stage, experiment_name):
            for name in SCRATCH:
                drop_tree(work_dir / name, report, dry_run)
            for name in REBUILT:
                path = work_dir / name
                if path.is_file():
                    size = path.stat().st_size
                    if not dry_run:
                        path.unlink()
                    report.removed += 1
                    report.bytes_freed += size
                    report.paths.append(path)
        _log(f"[clean] {experiment_name}: {report.removed} cleared so far")

    remove_empty_dirs([prepared_root, stage / "job_runs"], report, dry_run)
    kept = [name for name in KEEP]
    _log(f"[clean] kept for stage 08 and the record: {', '.join(kept)}")
    return report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would go; delete nothing")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    report = run_cleanup(args.stage.resolve(), args.experiment, args.dry_run)
    _log(f"[done] {report.summary()}")
    for path in report.unarchived[:10]:
        _log(f"[done]   kept (not in any archive): {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
