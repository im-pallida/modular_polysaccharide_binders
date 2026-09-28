"""
Tar-archive and results-table handling shared by every stage.

Each stage does the same two things with its outputs: fold loose per-structure
files into one <group>.tar.gz, and keep a per-experiment results table that
records what has already been processed so a re-run is cheap. The mechanics are
identical stage to stage; only the columns and the pass/fail rule differ, and
those stay in the stage's own module.

Everything that rewrites an archive does it the same way: read what is already
there, write the combination to a temp file, verify every expected member is
present, then os.replace. A crash at any point leaves the previous archive
intact rather than a truncated one, and loose originals are deleted only after
the swap succeeds.
"""
from __future__ import annotations

import csv
import io
import os
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import job_paths as jp  # noqa: E402

# tarfile gained an extraction `filter` in 3.12 and makes it mandatory in 3.14.
# Passing it unconditionally would break older interpreters on a cluster.
_EXTRACT_KWARGS = {"filter": "data"} if sys.version_info >= (3, 12) else {}

STRUCTURE_SUFFIXES = (".cif", ".pdb")


class ArchiveError(Exception):
    """An archive could not be read or rewritten."""


def _log(*parts: object) -> None:
    print(*parts, flush=True)


@dataclass
class SortResult:
    """One <group>.tar.gz that a run added members to."""
    experiment: str
    outcome: str
    group_key: str
    archive: Path
    added: int = 0
    total: int = 0


def group_members_by_protein(member_names: Sequence[str]) -> Dict[str, Dict[str, str]]:
    """Group tar members by protein_id -> {'json': name, 'structure': name}.

    Any member whose suffix is not .json/.cif/.pdb is ignored, which is what
    keeps the directory entry ('./') that `tar czf ... -C dir .` writes from
    becoming a bogus empty protein_id. Either half may be missing for a real
    protein_id; deciding what that means belongs to the caller, so this never
    raises.
    """
    groups: Dict[str, Dict[str, str]] = {}
    for member_name in member_names:
        suffix = Path(member_name).suffix.lower()
        if suffix != ".json" and suffix not in STRUCTURE_SUFFIXES:
            continue
        entry = groups.setdefault(Path(member_name).stem, {})
        entry["json" if suffix == ".json" else "structure"] = member_name
    return groups


def extract_member(archive: tarfile.TarFile, member_name: str, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    archive.extract(member_name, path=extract_dir, **_EXTRACT_KWARGS)
    return extract_dir / member_name


def read_archive_members(archive_path: Path) -> List[Tuple[tarfile.TarInfo, bytes]]:
    """Every regular file in an archive, as (info, bytes)."""
    members: List[Tuple[tarfile.TarInfo, bytes]] = []
    with tarfile.open(archive_path, "r:gz") as archive:
        for member_info in archive.getmembers():
            if not member_info.isfile():
                continue
            extracted = archive.extractfile(member_info)
            if extracted is None:
                raise ArchiveError(
                    f"could not read member {member_info.name!r} from {archive_path}"
                )
            members.append((member_info, extracted.read()))
    return members


def _write_and_swap(
    archive_path: Path,
    existing: Sequence[Tuple[tarfile.TarInfo, bytes]],
    add: Callable[[tarfile.TarFile], None],
    expected_names: set,
) -> None:
    """Write existing members plus whatever `add` contributes, verify, swap in."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = archive_path.with_name(f".{archive_path.name}.tmp{os.getpid()}")
    try:
        with tarfile.open(temp_path, "w:gz") as out:
            for member_info, payload in existing:
                out.addfile(member_info, io.BytesIO(payload))
            add(out)

        with tarfile.open(temp_path, "r:gz") as check:
            # Full member names, not basenames: stage 03 nests its fastas as
            # <protein_id>/<sequence_id>.fa, and comparing basenames there
            # would report every member missing. A leading "./" is normalised
            # away because that is how `tar czf ... -C dir .` writes them.
            present = {name[2:] if name.startswith("./") else name
                       for name in check.getnames()}
        missing = expected_names - present
        if missing:
            raise ArchiveError(
                f"{archive_path}: verification failed, missing {sorted(missing)}"
            )
        os.replace(temp_path, archive_path)
    finally:
        # No-op after a successful replace; cleans up after any failure above.
        temp_path.unlink(missing_ok=True)


def merge_files_into_archive(archive_path: Path, files: Sequence[Path]) -> Tuple[int, int]:
    """Add loose files on disk to an archive, keeping what is already in it.

    Returns (added, total) counted in members. The loose originals are deleted
    only once the swap has succeeded.
    """
    existing = read_archive_members(archive_path) if archive_path.exists() else []
    existing_names = {info.name for info, _ in existing}
    new_names = {path.name for path in files}

    collision = existing_names & new_names
    if collision:
        raise ArchiveError(
            f"{archive_path}: already contains {sorted(collision)} but they were "
            f"about to be added again -- a protein_id was routed twice"
        )

    def add(out: tarfile.TarFile) -> None:
        for path in files:
            out.add(path, arcname=path.name)

    _write_and_swap(archive_path, existing, add, existing_names | new_names)

    for path in files:
        path.unlink()
    return len(new_names), len(existing_names | new_names)


def merge_members_into_archive(
    archive_path: Path, new_members: Sequence[Tuple[tarfile.TarInfo, bytes]]
) -> Tuple[int, int]:
    """Add members held in memory (read from another archive) to an archive."""
    existing = read_archive_members(archive_path) if archive_path.exists() else []
    existing_names = {info.name for info, _ in existing}
    new_names = {info.name for info, _ in new_members}

    collision = existing_names & new_names
    if collision:
        raise ArchiveError(
            f"{archive_path}: already contains {sorted(collision)} but they were "
            f"about to be added again"
        )

    def add(out: tarfile.TarFile) -> None:
        for member_info, payload in new_members:
            out.addfile(member_info, io.BytesIO(payload))

    _write_and_swap(archive_path, existing, add, existing_names | new_names)
    return len(new_names), len(existing_names | new_names)


def regroup_and_archive(stage: Path, experiment: str) -> List[SortResult]:
    """Fold a run's loose routed files into per-group tarballs.

    Reads sorted_raw/<experiment>/<outcome>/ and writes
    sorted_clean/<experiment>/<outcome>/<group>.tar.gz, one group at a time so
    an unparseable name in one group does not stop the others from archiving.
    """
    results: List[SortResult] = []
    for outcome in jp.OUTCOMES:
        raw_dir = jp.sorted_raw_dir(stage, experiment, outcome)
        if not raw_dir.is_dir():
            continue
        loose_files = sorted(path for path in raw_dir.rglob("*") if path.is_file())
        if not loose_files:
            continue

        files_by_group: Dict[str, List[Path]] = {}
        for file_path in loose_files:
            # A file under <outcome>/<group>/ already states its group. Only a
            # file sitting directly in <outcome>/ has to have it derived, and a
            # name that cannot be parsed is reported rather than raised: one
            # unfilable file must not strand every other group's.
            if file_path.parent != raw_dir:
                group_key = file_path.parent.name
            else:
                try:
                    group_key = jp.group_key_from_job_name(file_path.stem)
                except ValueError as exc:
                    _log(f"[archive] cannot file {file_path.name}: {exc}")
                    continue
            files_by_group.setdefault(group_key, []).append(file_path)

        for group_key, group_files in sorted(files_by_group.items()):
            archive_path = jp.sorted_archive_path(stage, experiment, outcome, group_key)
            added, total = merge_files_into_archive(archive_path, group_files)
            results.append(
                SortResult(experiment, outcome, group_key, archive_path, added, total)
            )
    return results


def load_table(
    table_path: Path, legacy_path: Optional[Path] = None, experiment: Optional[str] = None,
    key: str = "protein_id",
) -> Dict[str, Dict[str, str]]:
    """Existing rows keyed by `key` -- the set of things already done.

    The key is a parameter because stages count different things: stages 01 and
    02 record one row per structure, stage 03 one row per designed sequence. A
    table keyed on the wrong column silently re-does all its work and leaves
    stale rows behind, which is exactly what happened before this was a
    parameter.

    If the per-experiment table does not exist yet but a legacy shared table
    does, that file's rows for this experiment are read instead, so structures
    processed before results were split per experiment are still recognised.
    The legacy file is only ever read.
    """
    source = table_path
    if not table_path.exists() and legacy_path is not None and legacy_path.exists():
        source = legacy_path

    by_id: Dict[str, Dict[str, str]] = {}
    if not (source.exists() and source.stat().st_size > 0):
        return by_id

    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if source is legacy_path and experiment and row.get("experiment_name") != experiment:
                continue
            if key not in row:
                raise ArchiveError(
                    f"{source}: no {key!r} column; found {sorted(row)}"
                )
            by_id[row[key]] = row
    return by_id


def save_table(
    by_id: Dict[str, Dict[str, str]],
    table_path: Path,
    fields: Sequence[str],
    sort_key: Optional[Callable[[Dict[str, str]], tuple]] = None,
) -> None:
    """Write every row, worst first by the stage's own ordering."""
    table_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(by_id.values())
    if sort_key is not None:
        rows.sort(key=sort_key)
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
