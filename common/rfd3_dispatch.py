#!/usr/bin/env python3
"""
Dispatching RFD3, shared by every stage that diffuses.

    job_runs/<experiment>.runlist            <- read
    outputs_raw/<experiment>/<group>/*.cif   <- written

Stage 01 diffuses symmetric dimers; stage 06 diffuses a linker between the two
copies of one. Only three things differ, and all three are arguments here:

    the stage root          where the outputs land
    the run list            which (json, seed) pairs to generate
    the sampler overrides   stage 01 pins the symmetry sampler, stage 06 must
                            NOT -- a linker runs from one copy's C-terminus to
                            the other's N-terminus and no symmetry operation
                            maps that onto itself, so RFD3 stops with
                            "Symmetry transform not found" if it is handed
                            stage 01's settings

That third one is why this file exists. run_one_job() the FUNCTION always took
sampler_overrides, so stage 06 could call it in-process and pass its own. But
run_one_job.py the COMMAND LINE had no way to say them, and the cluster route
goes through the command line -- so every submitted job ran stage 01's sampler
whatever stage it belonged to. Stage 06 therefore had no cluster path at all,
not because it needed a different one, but because it could not use this one.

WHAT EACH MODE DOES

    workstation   every row in turn, in-process, serial
    cluster       one sbatch per row, CANARY FIRST: the first is submitted
                  blocking and its output verified before the rest are queued,
                  so a broken environment costs one job instead of the run list

Both return the same DispatchReport, and both check for missing outputs the
same way -- against the raw file AND the group's archive, because cleanup
removes the raw once it is archived and a run that only looked on disk would
diffuse the whole experiment again.

Callers: 01_.../run_cluster.py and run_workstation.py are thin shims over this,
so stage 01's launcher is unchanged; stage 06 calls dispatch() directly.
"""
from __future__ import annotations

import getpass
import importlib.util
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import job_paths as jp  # noqa: E402
import partitions  # noqa: E402
import site_config as site  # noqa: E402
from archives import archived_stems  # noqa: E402

SBATCH_SCRIPT = Path(__file__).resolve().parent / "rfd3" / "run_rfd3_one.sbatch"
POLL_INTERVAL_S = 30
LOG_TAIL_LINES = 25

# squeue can fail transiently while the controller is busy. Tolerate a few
# failures in a row, then give up loudly rather than silently concluding that
# every job has finished.
MAX_CONSECUTIVE_SQUEUE_ERRORS = 5


def _log(*parts: object) -> None:
    print(*parts, flush=True)


@dataclass
class DispatchReport:
    """What a dispatcher did, so the launcher can archive and set an exit code
    without each dispatcher reimplementing the reporting.

    Defined here rather than in run_one_job.py, where it used to live: it
    describes a DISPATCH, the job runner never constructs one, and taking it
    from the runner module meant this file could only work with a runner that
    happened to define it.
    """

    rows: List[jp.RunRow] = field(default_factory=list)
    failed: List[jp.RunRow] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed

    def summary(self) -> str:
        if self.ok:
            return f"all {len(self.rows)} job(s) completed"
        return f"{len(self.failed)} of {len(self.rows)} job(s) failed"


def load_runner(runner: Path):
    """Import run_one_job.py from wherever the calling stage keeps it.

    By path rather than by name: there is one RFD3 job runner in the pipeline
    and it lives in stage 01, but stage 06 is not a subpackage of stage 01 and
    should not have to pretend to be. Imported once and cached by the loader.
    """
    runner = Path(runner).resolve()
    if not runner.is_file():
        raise SystemExit(
            f"ERROR: RFD3 job runner not found at {runner}. Every stage that "
            f"diffuses drives RFD3 through this one file rather than carrying "
            f"its own copy."
        )
    # One module object per file, shared with anything that imported it the
    # ordinary way. Loading a second copy would give two RFD3_SAMPLER_OVERRIDES
    # and two JobStatus enums, and `result.status is JobStatus.FAILED` would
    # then be False for a job that failed.
    existing = sys.modules.get(runner.stem)
    if existing is not None:
        where = getattr(existing, "__file__", None)
        if where and Path(where).resolve() == runner:
            return existing
    sys.path.insert(0, str(runner.parent))
    spec = importlib.util.spec_from_file_location(runner.stem, runner)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _submit(runner: Path, experiment: str, json_rel: str, global_seq: int,
            wait: bool, stage: Path, overrides: Sequence[str]) -> Optional[str]:
    """Submit one job. wait=True blocks until it finishes; wait=False returns
    the job ID."""
    log_dir = stage / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        "sbatch",
        *partitions.sbatch_args(),
        f"--chdir={stage}",
        f"--output={log_dir / 'rfd3_%j.out'}",
        f"--error={log_dir / 'rfd3_%j.err'}",
    ]
    if wait:
        cmd.append("--wait")
    # The runner and the stage both go on the command line: a submitted batch
    # script runs from SLURM's spool and can find neither from its own path.
    # The overrides go last, as a variable-length tail, so adding one never
    # shifts the position of anything the script reads by number.
    cmd += [str(SBATCH_SCRIPT), str(runner), str(stage),
            experiment, json_rel, str(global_seq), *overrides]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(_failure(proc, log_dir, json_rel, global_seq))
    if wait:
        return None

    job_id = _job_id(proc.stdout)
    if job_id is None:
        raise SystemExit(f"ERROR: could not parse job id from sbatch output: {proc.stdout!r}")
    return job_id


def _job_id(text: str) -> Optional[str]:
    match = re.search(r"Submitted batch job (\d+)", text or "")
    return match.group(1) if match else None


def _log_tail(log_dir: Path, job_id: str) -> str:
    """The end of what the job itself wrote, from SLURM's own files."""
    pieces: List[str] = []
    for suffix in ("err", "out"):
        path = log_dir / f"rfd3_{job_id}.{suffix}"
        if not path.is_file():
            pieces.append(f"  {path}\n    (not written)")
            continue
        lines = path.read_text(errors="replace").splitlines()
        shown = lines[-LOG_TAIL_LINES:]
        body = "\n".join(f"    {line}" for line in shown) or "    (empty)"
        pieces.append(f"  {path}  [last {len(shown)} of {len(lines)} line(s)]\n{body}")
    return "\n".join(pieces)


def _failure(proc: "subprocess.CompletedProcess", log_dir: Path,
             json_rel: str, global_seq: int) -> str:
    """Why it failed -- telling the SUBMISSION apart from the JOB.

    With --wait, sbatch's exit code is the job's, but its stdout is still only
    "Submitted batch job N". Printing that stdout as the error, which is what
    this used to do, reports a successful submission as the cause of a failed
    run. The job id is sitting in it, and SLURM has been writing the real
    reason into logs/ the whole time.

    The two cases are not the same problem:

        no job id   sbatch REFUSED it -- a partition that does not exist, a
                    walltime over the limit, no allocation. Nothing ran, and
                    logs/ is empty by design, so pointing at it wastes an
                    afternoon.
        a job id    it RAN and exited non-zero. sbatch has nothing to say about
                    that; the job's own log has everything.
    """
    job_id = _job_id(proc.stdout)
    if job_id is None:
        said = (proc.stderr or proc.stdout).strip() or "(sbatch said nothing)"
        return (f"ERROR: sbatch refused {json_rel} seq={global_seq} "
                f"(exit {proc.returncode}). Nothing ran, so there is no job log:\n"
                f"  {said}")
    return (f"ERROR: job {job_id} ran and failed for {json_rel} seq={global_seq} "
            f"(exit {proc.returncode}).\n"
            f"sbatch submitted it without complaint, so the reason is in the "
            f"job's own output:\n{_log_tail(log_dir, job_id)}")


def _poll_until_done(job_ids: Sequence[str]) -> None:
    """Wait for every submitted job to leave the queue."""
    remaining = set(job_ids)
    user = getpass.getuser()
    consecutive_errors = 0

    while remaining:
        proc = subprocess.run(
            ["squeue", "--noheader", "--user", user, "--format=%i"],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            consecutive_errors += 1
            if consecutive_errors >= MAX_CONSECUTIVE_SQUEUE_ERRORS:
                raise SystemExit(
                    f"ERROR: squeue failed {consecutive_errors} times in a row; "
                    f"cannot tell which of {len(remaining)} job(s) are still running.\n"
                    f"{proc.stderr.strip()}"
                )
            _log(f"[cluster] WARNING: squeue failed ({proc.returncode}), retrying: "
                 f"{proc.stderr.strip()}")
            time.sleep(POLL_INTERVAL_S)
            continue

        consecutive_errors = 0
        still_queued = {line.strip() for line in proc.stdout.split() if line.strip()}
        for job_id in sorted(remaining - still_queued):
            _log(f"[cluster] job {job_id} finished")
        remaining &= still_queued
        if remaining:
            time.sleep(POLL_INTERVAL_S)


def missing_outputs(experiment: str, rows: Sequence[jp.RunRow], stage: Path,
                    archived_names: Optional[set] = None) -> List[jp.RunRow]:
    """Rows with no structure yet, on disk OR already archived.

    Checking the raw file alone was right only while nothing ever deleted it.
    Once cleanup clears outputs_raw, every already-generated structure would
    look missing and the whole experiment would be diffused again -- hours on a
    workstation, days of queue on the cluster.

    Where the archive IS depends on the stage, so a caller that already knows
    passes archived_names and this does not guess. Stage 01 keeps its finished
    designs in outputs_clean/<experiment>/<group>.tar.gz, which is the default
    read here; stage 06 keeps its in sorted_clean/<experiment>/passed/, and
    reading the wrong one would report every archived design as missing.

    The default is read once per group rather than once per row: a 16k run
    would otherwise open the same tarball sixteen thousand times.
    """
    archived: Dict[str, set] = {}
    outstanding: List[jp.RunRow] = []
    for json_rel, global_seq in rows:
        if jp.expected_raw_cif(stage, experiment, json_rel, global_seq).exists():
            continue
        group_key = jp.group_key_from_json_rel(json_rel)
        if archived_names is None:
            if group_key not in archived:
                archived[group_key] = archived_stems(
                    jp.clean_archive_path(stage, experiment, group_key), ".cif"
                )
            known = archived[group_key]
        else:
            known = archived_names
        if jp.job_name(group_key, global_seq) in known:
            continue
        outstanding.append((json_rel, global_seq))
    return outstanding


def dispatch_workstation(experiment: str, run_list: Path, stage: Path,
                         runner: Path, overrides: Sequence[str] = ()):
    """Every row in turn, in-process, serial."""
    module = load_runner(runner)
    rows = jp.read_run_list(run_list)
    report = DispatchReport(rows=list(rows))
    _log(f"[dispatch] mode=workstation -> {len(rows)} job(s), serial, in-process")
    if overrides:
        _log(f"[dispatch] sampler: {' '.join(overrides)}")

    for index, (json_rel, global_seq) in enumerate(rows, start=1):
        _log(f"[dispatch] ({index}/{len(rows)}) {json_rel} seq={global_seq}")
        # run_one_job() converts every failure into a FAILED result, so one bad
        # job cannot abandon the rest of the queue.
        kwargs = {"stage": stage}
        if overrides:
            kwargs["sampler_overrides"] = tuple(overrides)
        result = module.run_one_job(experiment, json_rel, global_seq, **kwargs)
        # By NAME, not identity: a runner imported from a path and one imported
        # by name are two module objects with two JobStatus enums, and
        # `is JobStatus.FAILED` is then False for a job that failed.
        if getattr(result.status, "name", str(result.status)) == "FAILED":
            _log(f"[fail] {json_rel} seq={global_seq}: {result.message}")
            report.failed.append((json_rel, global_seq))
        _log()

    _log(f"[dispatch] {report.summary()}")
    for json_rel, global_seq in report.failed:
        _log(f"  {json_rel} seq={global_seq}")
    return report


def dispatch_cluster(experiment: str, run_list: Path, stage: Path,
                     runner: Path, overrides: Sequence[str] = (),
                     archived_names: Optional[set] = None):
    """One sbatch per row, canary first."""
    if not SBATCH_SCRIPT.is_file():
        raise SystemExit(f"ERROR: sbatch script not found: {SBATCH_SCRIPT}")

    module = load_runner(runner)
    rows = jp.read_run_list(run_list)
    if not rows:
        raise SystemExit(f"ERROR: run list {run_list} is empty, nothing to dispatch")
    report = DispatchReport(rows=list(rows))
    _log(f"[dispatch] mode=cluster -> {len(rows)} job(s) via sbatch, canary first")
    if overrides:
        _log(f"[dispatch] sampler: {' '.join(overrides)}")

    # Canary: one blocking job, verified, before committing the rest to the queue.
    canary_json_rel, canary_seq = rows[0]
    _log(f"[canary] submitting {canary_json_rel} seq={canary_seq} (sbatch --wait)")
    _submit(runner, experiment, canary_json_rel, canary_seq, True, stage, overrides)
    canary_cif = jp.expected_raw_cif(stage, experiment, canary_json_rel, canary_seq)
    if not canary_cif.exists():
        raise SystemExit(
            f"ERROR: canary finished but its output is missing: {canary_cif}\n"
            f"  check the sbatch logs in {stage / 'logs'} before resubmitting."
        )
    _log(f"[canary] ok -> {canary_cif}")

    remaining = rows[1:]
    if remaining:
        job_ids: List[str] = []
        for json_rel, global_seq in remaining:
            job_id = _submit(runner, experiment, json_rel, global_seq, False,
                             stage, overrides)
            job_ids.append(str(job_id))
            _log(f"[dispatch] submitted {json_rel} seq={global_seq} -> job {job_id}")

        _log(f"[dispatch] waiting for {len(job_ids)} job(s)...")
        _poll_until_done(job_ids)
    else:
        _log("[dispatch] canary was the whole run list")

    _log("[dispatch] verifying outputs...")
    report.failed = missing_outputs(experiment, rows, stage, archived_names)
    _log(f"[dispatch] {report.summary()}")
    for json_rel, global_seq in report.failed:
        missing = jp.expected_raw_cif(stage, experiment, json_rel, global_seq)
        _log(f"  {json_rel} seq={global_seq}: missing {missing}")
    return report


def dispatch(experiment: str, run_list: Path, stage: Path, runner: Path,
             overrides: Sequence[str] = (),
             archived_names: Optional[set] = None):
    """Whichever of the two this machine is. One line, so no stage decides it
    for itself and they cannot drift apart."""
    if site.mode() == "cluster":
        return dispatch_cluster(experiment, run_list, stage, runner, overrides,
                                archived_names)
    return dispatch_workstation(experiment, run_list, stage, runner, overrides)
