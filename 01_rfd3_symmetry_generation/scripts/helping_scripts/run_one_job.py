#!/usr/bin/env python3
"""
Runs a single RFD3 structure-generation job, and archive finished batches.

1. Resolves the RFD3 environment by sourcing scripts/env/rfd3.env;
2. Runs RFD3 into a private scratch directory;
3. Locates the one .cif.gz it produced (plus its .json metadata);
4. Unpacks them, flat and named, into outputs_raw/<experiment>/<group_key>/.

Archiving is not part of a job. archive_experiment() rolls the finished flat 
files into outputs_clean/<experiment>/<group_key>.tar.gz once, after a dispatch 
completes, under an exclusive lock. That keeps concurrent jobs off a shared tarball
-- appending per job meant every job of a group rewrote the same archive at once, 
which lost structures and crashed jobs -- and it makes archiving idempotent: 
members already in the archive are skipped, so re-running an experiment picks up 
anything an earlier run missed and adds newly generated structures to the archive 
that already exists.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import gzip
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# job_paths.py lives one level up, in scripts/ -- TRANSFER IT
# job_paths.py is shared by every stage and lives at the repo root, so a
# fix to the naming rules reaches all of them at once.
# helping_scripts/ -> scripts/ -> <stage>/ -> <repo root>/common
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import job_paths as jp  # noqa: E402
# DispatchReport describes a dispatch, not a job, so it lives with the
# dispatcher. Re-exported here because this is where it used to be.
from rfd3_dispatch import DispatchReport  # noqa: E402,F401
from archives import archived_stems  # noqa: E402
from designs import (  # noqa: E402
    DesignError,
    fixed_residue_map,
    record_line,
)

STAGE = Path(__file__).resolve().parents[2]  # scripts/helping_scripts/../.. -> STAGE

ENV_FILE_REL = Path("scripts") / "env" / "rfd3.env"
OVERLAY_REL = Path("overlay") / "rfd3_t3_overlay"

# This stage's overrides. They are the DEFAULT rather than the only option:
# stage 06 drives the same runner to generate linkers, which are not symmetric,
# and passing kind=symmetry on a json that declares no symmetry makes RFD3 stop
# with "Symmetry transform not found". Whatever a caller passes replaces this
# list wholesale; nothing merges, so a caller cannot inherit a symmetry setting
# it did not ask for.
RFD3_SAMPLER_OVERRIDES: Tuple[str, ...] = (
    "diffusion_batch_size=1",
    "n_batches=1",
    "skip_existing=False",
    "inference_sampler.kind=symmetry",
    "inference_sampler.num_timesteps=200",
    "+inference_sampler.sym_step_frac=0.9",
    "inference_sampler.use_classifier_free_guidance=False",
    "inference_sampler.allow_realignment=False",
)

# Archives are rewritten once per dispatch rather than once per job, so a
# middling compression level keeps that step quick on large groups.
ARCHIVE_COMPRESS_LEVEL = 6

# The truthy spellings checks.py accepts for its ASU-motif flags.
_TRUE_VALUES = {"1", "true", "yes", "y", "on"}


class JobStatus(Enum):
    SKIPPED = "skipped"
    DONE = "done"
    FAILED = "failed"


@dataclass
class JobResult:
    status: JobStatus
    name: str
    raw_cif: Optional[Path] = None
    clean_archive: Optional[Path] = None
    message: str = ""


@dataclass
class ArchiveResult:
    group_key: str
    archive: Path
    added: int = 0
    already_present: int = 0
    missing: List[str] = field(default_factory=list)


class JobError(Exception):
    """Internal control flow only: raised by the helpers below, caught once
    in run_one_job() and turned into a JobResult(FAILED, ...)."""


def _log(*parts: object) -> None:
    print(*parts, flush=True)


# Layer 1. Sourcing RFD3


def _source_env_file(env_file: Path) -> Dict[str, str]:
    """Source a bash env file and return the *complete* resulting environment.
    Anything the env file prints is redirected to stderr so it cannot corrupt
    the dump, and `env -0` is used so values containing newlines survive.
    """
    if not env_file.is_file():
        raise JobError(f"env file not found: {env_file}")

    marker = "___RFD3_ENV_DUMP___"
    script = (
        f"set -euo pipefail; "
        f"source {shlex.quote(str(env_file))} 1>&2; "
        f"printf '%s' {shlex.quote(marker)}; "
        f"env -0"
    )
    proc = subprocess.run(["bash", "-c", script], capture_output=True)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", "replace")
        raise JobError(f"failed to source env file {env_file}:\n{stderr}")

    blob = proc.stdout
    index = blob.rfind(marker.encode())
    if index < 0:
        raise JobError(f"could not read environment back from {env_file}")

    environment: Dict[str, str] = {}
    for chunk in blob[index + len(marker):].split(b"\0"):
        if not chunk:
            continue
        key, sep, value = chunk.partition(b"=")
        if sep:
            environment[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    return environment


@dataclass
class Rfd3Env:
    rfd3_exe: str
    ckpt: str
    environment: Dict[str, str]


def _resolve_rfd3_env(env_file: Path) -> Rfd3Env:
    environment = _source_env_file(env_file)
    rfd3_exe = environment.get("RFD3_EXE", "")
    ckpt = environment.get("CKPT", "")

    if not rfd3_exe or not ckpt:
        raise JobError(f"RFD3_EXE and/or CKPT not set - check {env_file} defines both")
    exe_path = Path(rfd3_exe)
    if not (exe_path.is_file() and exe_path.stat().st_mode & 0o111):
        raise JobError(f"RFD3_EXE does not exist or is not executable: {rfd3_exe}")
    if not Path(ckpt).is_file():
        raise JobError(f"CKPT does not exist: {ckpt}")

    return Rfd3Env(rfd3_exe, ckpt, environment)


def _symmetry_translations_env_name(symmetry_id: str) -> str:
    """The overlay builds this name at runtime from the json's symmetry id --
    'T2EXACT' -> 'RFD3_T2EXACT_TRANSLATIONS' -- so it is derived here the same
    way rather than hardcoded."""
    return f"RFD3_{symmetry_id.upper()}_TRANSLATIONS"


def _exact_order(symmetry_id: str) -> Optional[int]:
    """The N in T<N>EXACT, which is how many translation vectors the overlay
    requires. None for symmetry ids that are not of that form."""
    match = re.fullmatch(r"T(\d+)EXACT", symmetry_id.upper())
    return int(match.group(1)) if match else None


def _prepare_rfd3_inputs(
    stage: Path, json_path: Path, work_dir: Path, run_env: Dict[str, str]
) -> Path:
    """Give RFD3 an input it can actually apply the translational frames to."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        entries = jp.design_entries(data)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise JobError(f"could not read {json_path}: {exc}")

    try:
        from symmetry_check import (
            StructureError,
            copy_translations,
            load_structure,
            protein_chains,
            referenced_residues,
            write_asu,
        )
    except ImportError as exc:
        raise JobError(
            f"cannot prepare the RFD3 input: {exc}. "
            f"symmetry_check needs gemmi and numpy in the python running this job."
        )

    rewritten: Dict[str, dict] = {}
    changed = False

    for design, config in entries.items():
        label = f"design {design!r}" if design else "json"
        config = dict(config)
        rewritten[design] = config

        # The input path is resolved for EVERY design, symmetric or not, and the
        # resolved one written back.
        #
        # It used to be resolved only after the symmetry check below, so a json
        # declaring no symmetry -- which is every stage 06 linker json -- went to
        # RFD3 with its "input" exactly as stored. Stored by whichever machine
        # wrote it: a path under /home/<user>/1cbh_clear is not there on a
        # cluster, and RFD3's own failure for that is several hundred lines into
        # a log naming a path that means nothing without knowing who wrote it.
        #
        # resolve_seed_path re-anchors such a path onto this checkout, so a json
        # written on one machine runs on the other WITHOUT being regenerated --
        # which matters because stage 06's linker lengths are set by hand.
        raw_input = str(config.get("input", ""))
        seed = jp.resolve_seed_path(stage, raw_input, json_path)
        if seed is None:
            raise JobError(f"{label}: input structure not found: {raw_input!r}")
        if str(seed) != raw_input:
            config["input"] = str(seed)
            changed = True
            _log(f"[input] {label}: {raw_input}\n        -> {seed}")

        symmetry = config.get("symmetry")
        if not isinstance(symmetry, dict) or not symmetry.get("id"):
            _log(f"[symmetry] {label} declares no symmetry.id")
            continue
        symmetry_id = str(symmetry["id"])
        env_name = _symmetry_translations_env_name(symmetry_id)
        order = _exact_order(symmetry_id)

        try:
            structure = load_structure(seed)
            chains = [c.name for c in protein_chains(structure)]
        except (StructureError, OSError, RuntimeError, ValueError) as exc:
            raise JobError(f"{label}: could not read {seed}: {exc}")

        if len(chains) < 2:
            _check_translations_env(label, symmetry_id, env_name, order, run_env)
            continue

        # Several copies present: derive the frames and hand RFD3 one protomer.
        if order is not None and len(chains) != order:
            raise JobError(
                f"{label}: {seed.name} holds {len(chains)} protein chains "
                f"({', '.join(chains)}) but {symmetry_id} declares {order} frames"
            )

        named = {chain for chain, _ in referenced_residues(config)}
        if len(named) != 1:
            raise JobError(
                f"{label}: the contig must name exactly one chain as the ASU, "
                f"but names {sorted(named) or 'none'}. Seed chains: {', '.join(chains)}"
            )
        asu_chain = named.pop()
        if asu_chain not in chains:
            raise JobError(
                f"{label}: contig names chain {asu_chain!r}, which is not a protein "
                f"chain of {seed.name} ({', '.join(chains)})"
            )

        try:
            offsets = copy_translations(structure, asu_chain)
        except StructureError as exc:
            raise JobError(f"{label}: cannot derive frames from {seed.name}: {exc}")

        spec = ";".join(
            "0,0,0" if not any(v) else ",".join(f"{c:.4f}" for c in v) for v in offsets
        )
        asu_path = work_dir / f"asu_{design or 'design'}_{asu_chain}.pdb"
        write_asu(structure, asu_chain, asu_path)

        config["input"] = str(asu_path)
        changed = True

        existing = run_env.get(env_name, "").strip()
        if existing and existing != spec:
            raise JobError(
                f"{env_name} is set to {existing} but {seed.name} measures {spec}.\n"
                f"  Drop it from the env file and let the seed define the geometry, "
                f"or fix the seed."
            )
        run_env[env_name] = spec

        _log(f"[symmetry] {label}: {seed.name} holds {len(chains)} copies "
             f"({', '.join(chains)}); ASU = chain {asu_chain}")
        _log(f"[symmetry]   {env_name}={spec}")
        _log(f"[symmetry]   RFD3 input -> {asu_path.name} "
             f"(1 protomer + ligand, so the overlay keeps these frames)")

    if not changed:
        return json_path

    prepared = work_dir / f"prepared_{json_path.name}"
    payload = rewritten.get("") if list(rewritten) == [""] else rewritten
    prepared.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return prepared


def _check_translations_env(
    label: str, symmetry_id: str, env_name: str, order: Optional[int],
    run_env: Dict[str, str],
) -> None:
    """A single-ASU seed carries no geometry to measure, so the frames must
    come from the environment -- validated the same way frames.py validates
    them, but before the checkpoint loads."""
    value = run_env.get(env_name, "").strip()
    if not value:
        raise JobError(
            f"{env_name} is required by the {symmetry_id} sampler but is not set.\n"
            f'  Define it in the env file, e.g. export {env_name}="0,0,0;27.2860,11.6759,9.4259"\n'
            f"  (a seed holding several copies would let it be measured instead)"
        )
    vectors = [part.strip() for part in value.split(";") if part.strip()]
    for vector in vectors:
        components = vector.split(",")
        if len(components) != 3:
            raise JobError(f"{env_name}: expected x,y,z per operator, got {vector!r}")
        try:
            [float(c) for c in components]
        except ValueError:
            raise JobError(f"{env_name}: non-numeric operator {vector!r}")
    if order is not None and len(vectors) != order:
        raise JobError(
            f"{env_name} declares {order} frames ({symmetry_id}) but has "
            f"{len(vectors)} translation(s): {value}"
        )
    _log(f"[symmetry] {label}: {symmetry_id} -> {env_name}={value}")


def _invoke_rfd3(stage: Path, env: Rfd3Env, overlay_root: Path, json_path: Path,
                 work_dir: Path,
                 sampler_overrides: Sequence[str] = RFD3_SAMPLER_OVERRIDES) -> None:
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Start from the sourced environment, so everything the env file set up
    # (modules, conda, RFD3_T2EXACT_TRANSLATIONS) reaches RFD3.
    run_env = dict(env.environment)
    existing_pythonpath = run_env.get("PYTHONPATH", "")
    run_env["PYTHONPATH"] = (
        f"{overlay_root}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(overlay_root)
    )

    rfd3_json = _prepare_rfd3_inputs(stage, json_path, work_dir, run_env)

    cmd = [
        env.rfd3_exe,
        f"inputs={rfd3_json}",
        f"out_dir={work_dir}",
        f"ckpt_path={env.ckpt}",
        *sampler_overrides,
    ]
    _log("$ " + " ".join(shlex.quote(part) for part in cmd))
    # cwd is pinned to the stage root so a relative 'input' path in a json
    # resolves the same way here as it does in jp.resolve_seed_path.
    proc = subprocess.run(cmd, env=run_env, cwd=stage)
    if proc.returncode != 0:
        raise JobError(f"RFD3 exited with code {proc.returncode}")


# Step 2. Locating outputs


def _metadata_json_for(produced_cifgz: Path) -> Optional[Path]:
    """RFD3's metrics json for one produced structure: same directory, same
    stem, .json instead of .cif.gz --

        prepared_xlarge_xlarge_0_model_0.cif.gz
        prepared_xlarge_xlarge_0_model_0.json

    Path.stem strips only one suffix, so the '.cif.gz' double extension is
    removed explicitly. Returns None if RFD3 wrote no matching file.
    """
    name = produced_cifgz.name
    for suffix in (".cif.gz", ".cif"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    candidate = produced_cifgz.with_name(f"{name}.json")
    return candidate if candidate.is_file() else None


def _locate_outputs(work_dir: Path) -> Tuple[Path, Optional[Path]]:
    """The one .cif.gz RFD3 produced, and the metadata json belonging to it.

    The json is matched to the structure by name, not by being the only json
    in the directory: _prepare_rfd3_inputs writes the rewritten input json
    into this same folder (out_dir and our scratch dir are one and the same),
    so a glob finds two and the metadata would be dropped with only a warning
    -- leaving the filter with a structure it cannot evaluate.
    """
    cifgz_files = sorted(work_dir.rglob("*.cif.gz"))
    if not cifgz_files:
        listing = "\n".join(f"  {p}" for p in sorted(work_dir.rglob("*")) if p.is_file())
        raise JobError(
            f"no .cif.gz file found in {work_dir} after RFD3 ran\n{listing or '  (nothing)'}"
        )
    if len(cifgz_files) > 1:
        listing = "\n".join(f"  {p}" for p in cifgz_files)
        raise JobError(f"expected exactly 1 .cif.gz output, found {len(cifgz_files)}:\n{listing}")

    produced_cifgz = cifgz_files[0]
    _log(f"Found structure: {produced_cifgz}")

    produced_json = _metadata_json_for(produced_cifgz)
    if produced_json is None:
        present = sorted(path.name for path in produced_cifgz.parent.glob("*.json"))
        _log(f"WARNING: no metadata json named after {produced_cifgz.name} in "
             f"{produced_cifgz.parent}; jsons present: {present or '(none)'}")
    else:
        _log(f"Found metadata: {produced_json}")
    return produced_cifgz, produced_json


def _place_raw_files(
    produced_cifgz: Path,
    produced_json: Optional[Path],
    raw_dir: Path,
    raw_cif: Path,
    raw_json: Path,
) -> List[Path]:
    """Unpack the structure (and copy its metadata) into outputs_raw, writing
    each through a temporary file so an interrupted job cannot leave a
    half-written .cif that later runs would treat as complete."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    tmp_cif = raw_cif.with_name(f".tmp_{raw_cif.name}.{os.getpid()}")
    try:
        with gzip.open(produced_cifgz, "rb") as gz_in, open(tmp_cif, "wb") as cif_out:
            shutil.copyfileobj(gz_in, cif_out)
        tmp_cif.replace(raw_cif)
    finally:
        tmp_cif.unlink(missing_ok=True)
    _log(str(raw_cif))
    written.append(raw_cif)

    if produced_json is not None:
        tmp_json = raw_json.with_name(f".tmp_{raw_json.name}.{os.getpid()}")
        try:
            shutil.copy(produced_json, tmp_json)
            tmp_json.replace(raw_json)
        finally:
            tmp_json.unlink(missing_ok=True)
        _log(str(raw_json))
        written.append(raw_json)
    return written


# Step 3.Archiving.


@contextlib.contextmanager
def exclusive_lock(lock_path: Path) -> Iterator[None]:
    """Serialise access to a shared file across processes and nodes.

    Used for archive rewrites (two dispatches of the same experiment must not
    interleave their read-modify-write of one tarball) and by the launcher for
    the global sequence counter.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "w")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _rewrite_archive(archive: Path, group_key: str, candidates: Sequence[Path]) -> Tuple[int, int]:
    """Add whatever of `candidates` is not already in `archive`.

    gzip cannot be appended to in place, so the archive is expanded to a plain
    tar, extended, recompressed and moved back atomically. Temporary names
    carry the pid, and the replace is atomic, so a crash mid-rewrite leaves the
    previous archive intact rather than a truncated one.
    """
    clean_dir = archive.parent
    clean_dir.mkdir(parents=True, exist_ok=True)
    tmp_tar = clean_dir / f".tmp_{group_key}_{os.getpid()}.tar"
    tmp_gz = clean_dir / f".tmp_{group_key}_{os.getpid()}.tar.gz"

    try:
        if archive.exists():
            with gzip.open(archive, "rb") as gz_in, open(tmp_tar, "wb") as tar_out:
                shutil.copyfileobj(gz_in, tar_out)
            with tarfile.open(tmp_tar, "r") as tar:
                existing = set(tar.getnames())
            mode = "a"
        else:
            existing = set()
            mode = "w"

        new_files = [path for path in candidates if path.name not in existing]
        if not new_files:
            return 0, len(existing)

        with tarfile.open(tmp_tar, mode) as tar:
            for path in new_files:
                tar.add(path, arcname=path.name)

        with open(tmp_tar, "rb") as tar_in, gzip.open(
            tmp_gz, "wb", compresslevel=ARCHIVE_COMPRESS_LEVEL
        ) as gz_out:
            shutil.copyfileobj(tar_in, gz_out)
        tmp_gz.replace(archive)
        return len(new_files), len(existing)
    finally:
        tmp_tar.unlink(missing_ok=True)
        tmp_gz.unlink(missing_ok=True)


def archive_experiment(
    stage: Path, experiment: str, rows: Iterable[jp.RunRow]
) -> List[ArchiveResult]:
    """Safe to run repeatedly: members already present are left alone, so this
    both extends an archive from an earlier run and repairs one that is
    missing structures because a previous dispatch was interrupted.
    """
    results: List[ArchiveResult] = []
    for group_key, group_rows in sorted(jp.group_rows(rows).items()):
        raw_dir = jp.raw_dir(stage, experiment, group_key)
        archive = jp.clean_archive_path(stage, experiment, group_key)
        result = ArchiveResult(group_key, archive)

        candidates: List[Path] = []
        for json_rel, global_seq in group_rows:
            name = jp.job_name(group_key, global_seq)
            cif = raw_dir / f"{name}.cif"
            if not cif.is_file():
                result.missing.append(name)
                continue
            candidates.append(cif)
            metadata = raw_dir / f"{name}.json"
            if metadata.is_file():
                candidates.append(metadata)

        if candidates:
            with exclusive_lock(Path(f"{archive}.lock")):
                result.added, result.already_present = _rewrite_archive(
                    archive, group_key, candidates
                )
        results.append(result)
    return results


def _record_fixed_residues(
    stage: Path, experiment: str, json_path: Path, group_key: str, name: str,
    produced_json: Optional[Path],
) -> None:
    """Append what was held fixed for this structure, in both numberings.

    Written here because this is the only point where both halves exist at
    once: the design config says which residues had their side chains fixed
    (in the seed's numbering), and RFD3's own metadata says where each of them
    ended up in this structure. Later stages need the pairing -- stage 02 to
    fit the design back onto its seed, stage 03 to tell ProteinMPNN what not to
    touch -- and deriving it again downstream would mean re-reading a json that
    may have been edited since.

    Appended under the same lock the sequence counter uses, so parallel cluster
    jobs cannot interleave a line. Never fatal: a design that fixes no side
    chains is still a valid thing to generate, it just cannot go past stage 02.
    """
    if produced_json is None:
        _log("[fixed] no metadata json, nothing recorded")
        return
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        entries = jp.design_entries(data)
        config = entries.get("") or next(iter(entries.values()))
        payload = json.loads(produced_json.read_text(encoding="utf-8"))
        diffused_index_map = payload.get("diffused_index_map") or {}
        mapped, unmapped = fixed_residue_map(config, diffused_index_map)
    except (OSError, ValueError, json.JSONDecodeError, DesignError) as exc:
        _log(f"[fixed] not recorded: {str(exc).splitlines()[0]}")
        _log("[fixed] this structure cannot be aligned or redesigned past stage 02")
        return

    record_path = jp.fixed_residues_path(stage, experiment)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(Path(f"{record_path}.lock")):
        with record_path.open("a", encoding="utf-8") as handle:
            handle.write(record_line(name, group_key, mapped))
    _log(f"[fixed] {len(mapped)} residue(s) recorded: {mapped}")
    if unmapped:
        _log(f"[fixed] {len(unmapped)} not in diffused_index_map: {unmapped}")


# Step 4. Run one job.


def run_one_job(
    experiment: str, json_rel: str, global_seq: int, stage: Path = STAGE,
    sampler_overrides: Sequence[str] = RFD3_SAMPLER_OVERRIDES,
) -> JobResult:
    json_path = jp.json_path(stage, experiment, json_rel)
    group_key = jp.group_key_from_json_rel(json_rel)
    name = jp.job_name(group_key, global_seq)

    if not json_path.is_file():
        return JobResult(JobStatus.FAILED, name, message=f"missing json config: {json_path}")

    raw_dir = jp.raw_dir(stage, experiment, group_key)
    raw_cif = jp.raw_cif_path(stage, experiment, group_key, name)
    raw_json = jp.raw_json_path(stage, experiment, group_key, name)
    clean_archive = jp.clean_archive_path(stage, experiment, group_key)
    work_dir = jp.tmp_work_dir(stage, experiment, group_key, name)
    env_file = stage / ENV_FILE_REL
    overlay_root = stage / OVERLAY_REL

    _log("===== RFD3 JOB =====")
    _log(datetime.now().isoformat())
    _log(f"experiment={experiment}  json={json_rel}  group_key={group_key}"
         f"  global_seq={global_seq}  name={name}")
    _log(f"json_path={json_path}")
    _log(f"raw_cif={raw_cif}")
    _log(f"clean_archive={clean_archive}  (written after dispatch, not here)")
    _log()

    if raw_cif.exists():
        _log(f"[skip] {name} -> {raw_cif} already exists")
        return JobResult(JobStatus.SKIPPED, name, raw_cif, clean_archive)

    # The raw file is not the only evidence this was already generated. Once
    # cleanup has cleared outputs_raw, the structure lives only in the group's
    # archive -- and a resume that looked at the raw file alone would diffuse
    # the whole experiment again, which is precisely what stage 06 did before
    # its skip check was pointed at the archive instead.
    if name in archived_stems(clean_archive, ".cif"):
        _log(f"[skip] {name} -> already inside {clean_archive.name}")
        return JobResult(JobStatus.SKIPPED, name, raw_cif, clean_archive)

    try:
        env = _resolve_rfd3_env(env_file)

        _log("===== RUN RFD3 =====")
        _invoke_rfd3(stage, env, overlay_root, json_path, work_dir,
                     sampler_overrides)

        _log()
        _log("===== LOCATE OUTPUT FILES =====")
        produced_cifgz, produced_json = _locate_outputs(work_dir)

        _log()
        _log("===== PLACE FLAT RAW FILES =====")
        _place_raw_files(produced_cifgz, produced_json, raw_dir, raw_cif, raw_json)

        _log()
        _log("===== RECORD FIXED RESIDUES =====")
        _record_fixed_residues(stage, experiment, json_path, group_key, name, produced_json)
    except JobError as exc:
        return JobResult(JobStatus.FAILED, name, message=str(exc))
    except Exception:  # noqa: BLE001 - one bad job must not take down a batch
        return JobResult(
            JobStatus.FAILED, name,
            message=f"unexpected error while running {name}:\n{traceback.format_exc()}",
        )

    _log()
    _log("===== CLEAN TEMP WORK DIR =====")
    shutil.rmtree(work_dir, ignore_errors=True)
    _log(f"[done] {name} -> raw: {raw_cif}")
    _log(datetime.now().isoformat())

    return JobResult(JobStatus.DONE, name, raw_cif, clean_archive)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("experiment", help="folder name under json/")
    parser.add_argument("json_relative_path", help="json filename within json/<experiment>/")
    parser.add_argument("global_seq", type=int, help="global sequence number for this structure")
    parser.add_argument(
        "--stage", type=Path, default=STAGE,
        help=f"stage root directory (default: {STAGE})",
    )
    # The sampler on the COMMAND LINE, not only in the function signature.
    #
    # run_one_job() has always taken sampler_overrides, so a stage that imports
    # it could pass its own. This script could not, and the cluster route goes
    # through this script -- so every submitted job ran stage 01's SYMMETRIC
    # sampler whatever stage it belonged to. That is the whole reason stage 06
    # had no cluster path: not a different architecture, one missing argument.
    #
    # Repeated --override REPLACES the defaults rather than adding to them. The
    # caller that needs this is a stage whose defaults are wrong for it, and
    # "symmetry settings plus your settings" is exactly what breaks a linker.
    parser.add_argument(
        "--override", action="append", default=None, metavar="KEY=VALUE",
        help=f"an RFD3 sampler override, repeatable. Given at all, these "
             f"REPLACE the stage 01 defaults ({', '.join(RFD3_SAMPLER_OVERRIDES)})",
    )
    args = parser.parse_args(argv)

    overrides = tuple(args.override) if args.override else RFD3_SAMPLER_OVERRIDES
    result = run_one_job(
        args.experiment, args.json_relative_path, args.global_seq,
        stage=args.stage, sampler_overrides=overrides,
    )
    if result.status is JobStatus.FAILED:
        print(f"ERROR: {result.message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
