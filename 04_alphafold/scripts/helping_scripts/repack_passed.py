#!/usr/bin/env python3
"""
One-off: pack already-scored AF3 output into the archives stage 05 reads.

    tables/stage_04_results*.csv     <- which sequences passed, and their group
    outputs/<sequence_id>_monomer/   <- the models themselves
    sorted_clean/<experiment>/passed/<group>.tar.gz

For a run that was scored before stage 04 was rewritten. Normally
score_designs.py writes these archives as it scores; this does the packing half
alone, from a table that already exists, so nothing is refolded or rescored.

It reads whichever results table is present and works out the pass rule from the
columns:

    status == PASSED        the rewritten stage 04
    monomer_pass == True    the 16k production table, monomer half only

The nesting is <protein_id>/<sequence_id>_monomer/..., which is what the
transfer to stage 05 groups on. Archives are extended, never replaced, and a
sequence already in one is skipped -- so this is safe to run twice and safe to
run after score_designs has written some of them.

Two sources are understood:

    outputs/<sequence_id>_monomer/          loose AF3 output (the default)
    <dir>/<group>.tar.gz                    --source-archives: group tarballs
                                            holding <sequence_id>_monomer/...,
                                            which is what save_monomer_passed
                                            wrote for the 16k production run

The second is re-nested under <protein_id>/ on the way in, because that is what
the transfer to stage 05 groups on. --experiment-as renames the experiment
while doing it: the old table records the long
..._20260818_pp_1p8_pc_2p2_cont_4p0_8p0 form, and the rest of the pipeline now
uses the short one, so without the rename the structures would land in a second
experiment directory that stage 05 could not match against stage 02.

The af3_runs/ tree is NOT a source: the AF3 job scripts move each prediction out
of it into outputs/ and leave only the input json behind.

Usage:
    ./repack_passed.py --stage ~/1cbh_clear/04_alphafold --dry-run
    ./repack_passed.py --stage ~/1cbh_clear/04_alphafold
    ./repack_passed.py --stage ~/1cbh_clear/04_alphafold \
        --table ~/1cbh_clear/results/<run>/stage_04/stage_04_monomer_passed.csv \
        --source-archives ~/1cbh_clear/results/<run>/stage_04/<long_experiment> \
        --experiment-as 15082026_16k_production_4_seeds_4_sizes
"""
from __future__ import annotations

import argparse
import csv
import sys
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import job_paths as jp  # noqa: E402
from archives import merge_members_into_archive  # noqa: E402

STAGE = Path(__file__).resolve().parents[2]


class RepackError(Exception):
    """The passed structures could not be packed."""


def _log(*parts: object) -> None:
    print(*parts, flush=True)


def find_tables(stage: Path, experiment: Optional[str],
                explicit: Optional[Path] = None) -> List[Path]:
    """Every results table that might say what passed, newest name first."""
    if explicit is not None:
        if not explicit.is_file():
            raise RepackError(f"no results table at {explicit}")
        return [explicit]
    tables_dir = stage / "tables"
    if not tables_dir.is_dir():
        raise RepackError(f"no tables directory at {tables_dir}")
    candidates: List[Path] = []
    if experiment:
        candidates.append(jp.results_table_path(stage, experiment))
    else:
        candidates.extend(sorted(tables_dir.glob("stage_04_results_*.csv")))
    candidates.append(tables_dir / "stage_04_monomer_passed.csv")
    candidates.append(tables_dir / "stage_04_results.csv")
    return [path for path in candidates if path.is_file()]


def passing_rows(table_path: Path) -> Tuple[List[dict], str]:
    """(rows that passed, which column decided it)."""
    with table_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return [], "empty"

    columns = set(rows[0])
    if "status" in columns:
        return [row for row in rows
                if str(row.get("status", "")).strip().upper() == "PASSED"], "status"
    if "monomer_pass" in columns:
        return [row for row in rows
                if str(row.get("monomer_pass", "")).strip().lower() == "true"], "monomer_pass"
    raise RepackError(
        f"{table_path.name}: no 'status' or 'monomer_pass' column; found {sorted(columns)}"
    )


def experiment_of(row: dict) -> str:
    """The rewritten table calls it experiment_name, the old one experiment."""
    return row.get("experiment_name") or row.get("experiment") or ""


def collect_from_archive(archive_path: Path,
                         wanted: Dict[str, str]) -> List[Tuple[tarfile.TarInfo, bytes]]:
    """Members of a <group>.tar.gz written as <sequence_id>_monomer/..., re-nested.

    wanted maps sequence_id -> protein_id. A member for a sequence that did not
    pass, or that the table does not know, is left behind rather than guessed at.
    """
    members: List[Tuple[tarfile.TarInfo, bytes]] = []
    with tarfile.open(archive_path, "r:gz") as archive:
        for info in archive.getmembers():
            if not info.isfile():
                continue
            name = info.name[2:] if info.name.startswith("./") else info.name
            job_name = name.split("/")[0]
            if not job_name.endswith("_monomer"):
                continue
            sequence_id = job_name[: -len("_monomer")]
            protein_id = wanted.get(sequence_id)
            if protein_id is None:
                continue
            handle = archive.extractfile(info)
            if handle is None:
                continue
            payload = handle.read()
            nested = tarfile.TarInfo(f"{protein_id}/{name}")
            nested.size = len(payload)
            members.append((nested, payload))
    return members


def collect_job(job_dir: Path, protein_id: str) -> List[Tuple[tarfile.TarInfo, bytes]]:
    """Everything AF3 wrote for one job, nested <protein_id>/<job_name>/..."""
    members: List[Tuple[tarfile.TarInfo, bytes]] = []
    for file_path in sorted(job_dir.rglob("*")):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(job_dir.parent).as_posix()
        payload = file_path.read_bytes()
        info = tarfile.TarInfo(f"{protein_id}/{relative}")
        info.size = len(payload)
        members.append((info, payload))
    return members


def repack(stage: Path, source_root: Path, experiment: Optional[str],
           dry_run: bool, table: Optional[Path] = None,
           source_archives: Optional[Path] = None,
           experiment_as: Optional[str] = None) -> int:
    tables = find_tables(stage, experiment, table)
    if not tables:
        raise RepackError(
            f"no results table under {stage / 'tables'} -- nothing says what passed"
        )
    if source_archives is not None:
        if not source_archives.is_dir():
            raise RepackError(f"no source archive directory at {source_archives}")
    elif not source_root.is_dir():
        raise RepackError(f"no AF3 output directory at {source_root}")

    seen: set = set()
    by_archive: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for table_path in tables:
        rows, rule = passing_rows(table_path)
        fresh = [row for row in rows if row.get("sequence_id") not in seen]
        seen.update(row.get("sequence_id") for row in fresh)
        _log(f"[table] {table_path.name}: {len(rows)} passed by {rule}, "
             f"{len(fresh)} not already counted")
        for row in fresh:
            key = (experiment_of(row), row.get("group", ""))
            by_archive[key].append(row)

    if not by_archive:
        _log("[done] nothing passed in any table")
        return 0

    total_new = 0
    missing: List[str] = []
    for (experiment_name, group_key), rows in sorted(by_archive.items()):
        if not experiment_name or not group_key:
            _log(f"[skip] {len(rows)} row(s) with no experiment or group recorded")
            continue
        destination_experiment = experiment_as or experiment_name
        archive_path = jp.sorted_archive_path(
            stage, destination_experiment, "passed", group_key
        )

        existing: set = set()
        if archive_path.is_file():
            with tarfile.open(archive_path, "r:gz") as archive:
                existing = set(archive.getnames())

        members: List[Tuple[tarfile.TarInfo, bytes]] = []
        packed = 0

        if source_archives is not None:
            source_path = source_archives / f"{group_key}.tar.gz"
            if not source_path.is_file():
                missing.extend(row["sequence_id"] for row in rows)
                _log(f"[skip] no source archive at {source_path}")
                continue
            wanted = {row["sequence_id"]: (row.get("protein_id") or "")
                      for row in rows}
            found = collect_from_archive(source_path, wanted)
            fresh = [(info, payload) for info, payload in found
                     if info.name not in existing]
            packed = len({info.name.split("/")[1] for info, _ in fresh})
            members.extend(fresh)
            missing.extend(
                sequence_id for sequence_id in wanted
                if not any(f"/{sequence_id}_monomer/" in info.name
                           for info, _ in found)
            )
            if members:
                if dry_run:
                    _log(f"[dry-run] {packed} sequence(s), {len(members)} file(s) "
                         f"-> {archive_path}")
                    total_new += packed
                else:
                    added, total = merge_members_into_archive(archive_path, members)
                    total_new += packed
                    _log(f"[pack] {packed} sequence(s), +{added} file(s), {total} total "
                         f"-> {archive_path}")
            continue

        for row in sorted(rows, key=lambda item: item.get("sequence_id", "")):
            sequence_id = row["sequence_id"]
            protein_id = row.get("protein_id") or ""
            job_dir = source_root / jp.af3_job_name(sequence_id)
            if not job_dir.is_dir():
                missing.append(sequence_id)
                continue
            fresh = [(info, payload) for info, payload in collect_job(job_dir, protein_id)
                     if info.name not in existing]
            if not fresh:
                continue
            members.extend(fresh)
            packed += 1

        if not members:
            continue
        if dry_run:
            _log(f"[dry-run] {packed} sequence(s), {len(members)} file(s) "
                 f"-> {archive_path}")
            total_new += packed
            continue
        added, total = merge_members_into_archive(archive_path, members)
        total_new += packed
        _log(f"[pack] {packed} sequence(s), +{added} file(s), {total} total "
             f"-> {archive_path}")

    if missing:
        where = source_archives if source_archives is not None else source_root
        _log(f"[warn] {len(missing)} passing sequence(s) not found under "
             f"{where}, e.g. {missing[:3]}")
    _log(f"[done] {total_new} sequence(s) {'would be' if dry_run else ''} packed")
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--source", type=Path, default=None,
                        help="where the <sequence_id>_monomer directories are "
                             "(default: <stage>/outputs)")
    parser.add_argument("--table", type=Path, default=None,
                        help="use this results table instead of looking in tables/")
    parser.add_argument("--source-archives", type=Path, default=None,
                        help="a directory of <group>.tar.gz holding "
                             "<sequence_id>_monomer/..., instead of loose outputs")
    parser.add_argument("--experiment-as", default=None,
                        help="write under this experiment name instead of the "
                             "one the table records")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    stage = args.stage.expanduser().resolve()
    source = (args.source.expanduser().resolve() if args.source else stage / "outputs")
    try:
        return repack(
            stage, source, args.experiment, args.dry_run,
            args.table.expanduser().resolve() if args.table else None,
            args.source_archives.expanduser().resolve() if args.source_archives else None,
            args.experiment_as,
        )
    except (RepackError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
