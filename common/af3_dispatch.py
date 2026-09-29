#!/usr/bin/env python3
"""
Dispatching AlphaFold3, shared by every stage that folds.

    job_runs/af3/<experiment>/<group>/<sequence_id>_<suffix>.json  <- read
    outputs/<sequence_id>_<suffix>/                                <- written

Stage 04 folds the monomer; stage 06 folds it again with the ligand. Only the
suffix and the stage root differ, so the canary, the concurrency cap and the
resume behaviour live here rather than in two copies that drift apart.

One job per sequence, because that is how AF3 runs. On the cluster each job is
its own `sbatch --wait`, and MAX_CONCURRENT of them are in flight at a time --
the account's cap is 4, and a 5th submission is rejected outright. On a
workstation they run one after another in the foreground.

The first job is a canary: it runs alone, and the rest are only queued once it
has produced output. Every AF3 job shares the same module, image and database
paths, so a misconfiguration fails all of them identically -- better to find
that out after one job than after four hundred.

Blocking by design. A few hundred sequences is days of queue, so run this
under tmux or screen; it is safe to interrupt and re-run, because a job whose
output already exists is skipped.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import job_paths as jp  # noqa: E402
import partitions  # noqa: E402
import site_config as site  # noqa: E402
from archives import load_table  # noqa: E402

SBATCH_SCRIPT = jp.af3_shared_script("run_af3_one.sbatch")
WS_SCRIPT = jp.af3_shared_script("run_af3_one.sh")

MAX_CONCURRENT = 4


class Af3Error(Exception):
    """AF3 could not be dispatched."""


def _log(*parts: object) -> None:
    print(*parts, flush=True)


@dataclass
class RunReport:
    submitted: int = 0
    succeeded: int = 0
    skipped: int = 0
    failed: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed

    def summary(self) -> str:
        return (f"af3: {self.submitted} job(s) run, {self.succeeded} produced output, "
                f"{self.skipped} already folded, {len(self.failed)} failed")


def pending_jobs(stage: Path, experiment: Optional[str],
                 report: RunReport,
                 suffix: str = jp.LIGAND_SUFFIX) -> List[Tuple[str, Path]]:
    """(job_name, json path) for everything prepared but not yet folded."""
    names = [experiment] if experiment else jp.inputs_experiments(stage)
    jobs: List[Tuple[str, Path]] = []
    for experiment_name in names:
        root = jp.af3_json_experiment_dir(stage, experiment_name)
        if not root.is_dir():
            continue
        # A scored sequence has already folded, whether or not cleanup has since
        # removed its outputs/ directory.
        scored = set(load_table(jp.results_table_path(stage, experiment_name),
                                key="sequence_id"))
        for json_path in sorted(root.rglob(f"*_{suffix}.json")):
            job_name = json_path.stem
            ending = f"_{suffix}"
            sequence_id = (job_name[: -len(ending)]
                           if job_name.endswith(ending) else job_name)
            if sequence_id in scored or jp.af3_output_dir(stage, job_name).is_dir():
                report.skipped += 1
                continue
            jobs.append((job_name, json_path))
    return jobs


def run_one(stage: Path, job_name: str, json_path: Path,
            cluster: bool) -> Tuple[bool, str]:
    """Fold one sequence. Returns (produced output, why not).

    The reason is returned rather than left on the terminal because the two
    ways this fails look identical from outside and are not the same problem:

        the job RAN and wrote nothing   -- AF3's own failure, and SLURM's log
                                           holds the traceback
        the job was never SUBMITTED     -- sbatch rejected it, printed why, and
                                           SLURM created no log at all

    An empty logs/ is the signature of the second, and telling someone to read
    an empty directory is how an afternoon goes.
    """
    logs = stage / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    if cluster:
        command = [
            "sbatch", "--wait",
            *partitions.sbatch_args(),
            "--job-name", job_name,
            "--chdir", str(stage),
            "--output", str(logs / f"{job_name}-%j.out"),
            "--error", str(logs / f"{job_name}-%j.err"),
            str(SBATCH_SCRIPT), str(json_path), str(stage),
        ]
    else:
        command = ["bash", str(WS_SCRIPT), str(json_path), str(stage)]

    completed = subprocess.run(command, capture_output=True, text=True)
    # Echoed as it happens: these runs are long and watching them is the point.
    if completed.stdout:
        _log(completed.stdout.rstrip())
    if completed.stderr:
        _log(completed.stderr.rstrip())

    produced = jp.af3_output_dir(stage, job_name).is_dir()
    if produced and completed.returncode == 0:
        return True, ""

    # The exit code alone is not enough on the cluster: sbatch --wait reports
    # the job's status, but a job can exit 0 after AF3 wrote nothing usable.
    written = sorted(logs.glob(f"{job_name}-*"))
    detail = (completed.stderr or completed.stdout or "").strip()
    tail = detail.splitlines()[-12:] if detail else []

    if not cluster:
        # No SLURM here, so logs/ is empty by design and says nothing. The
        # runner's own output is the whole diagnosis.
        reason = (f"the workstation runner exited {completed.returncode}. "
                  f"There is no SLURM log on this machine -- what it printed "
                  f"is above")
        if tail:
            reason += ":\n      " + "\n      ".join(tail)
    elif not written:
        reason = ("nothing was submitted -- SLURM wrote no log for it, which "
                  "means the job was rejected rather than run")
        reason += f". sbatch said: {detail}" if detail else (
            ", and sbatch printed nothing. Try submitting one by hand to see "
            "the refusal")
    elif not produced:
        reason = (f"it ran but wrote no output directory. The traceback is in "
                  f"{written[-1]}")
    else:
        reason = f"it exited {completed.returncode}. See {written[-1]}"
    return False, reason


def run_af3(stage: Path, experiment: Optional[str] = None,
            max_concurrent: int = MAX_CONCURRENT,
            suffix: str = jp.LIGAND_SUFFIX) -> RunReport:
    report = RunReport()
    cluster = site.mode() == "cluster"
    script = SBATCH_SCRIPT if cluster else WS_SCRIPT
    if not script.is_file():
        raise Af3Error(f"missing runner script: {script}")

    jobs = pending_jobs(stage, experiment, report, suffix)
    if not jobs:
        _log(f"[af3] nothing to fold ({report.skipped} already folded)")
        return report

    mode = "cluster" if cluster else "workstation"
    concurrent = max_concurrent if cluster else 1
    _log(f"[af3] {len(jobs)} job(s) to fold, mode={mode}, "
         f"{concurrent} at a time, {report.skipped} already folded")
    _log(f"[af3] this blocks until every job finishes -- run it under tmux or screen")

    canary_name, canary_json = jobs[0]
    _log(f"[af3] canary: {canary_name} runs alone first; the rest are queued only "
         f"if it produces output")
    ok, reason = run_one(stage, canary_name, canary_json, cluster)
    if ok:
        report.succeeded += 1
    else:
        report.failed.append(canary_name)
    report.submitted += 1

    if report.failed:
        raise Af3Error(
            f"the canary job {canary_name} did not fold: {reason}\n"
            f"  Every AF3 job uses the same module, image and database paths, so "
            f"the remaining {len(jobs) - 1} would fail the same way -- nothing "
            f"else was submitted."
        )

    remaining = jobs[1:]
    if not remaining:
        return report

    _log(f"[af3] canary succeeded; queueing the remaining {len(remaining)} job(s)")
    with ThreadPoolExecutor(max_workers=concurrent) as pool:
        futures = {
            pool.submit(run_one, stage, job_name, json_path, cluster): job_name
            for job_name, json_path in remaining
        }
        done = 0
        # Collected in submission order so the log reads sensibly; the pool
        # still keeps `concurrent` of them running at once.
        for future, job_name in futures.items():
            report.submitted += 1
            try:
                # A tuple now, not a bool. Unpacked rather than tested: a
                # non-empty tuple is truthy, so `if produced:` would have
                # counted every failure as a success.
                produced, why = future.result()
            except Exception as exc:  # noqa: BLE001 - one job must not stop the batch
                report.failed.append(f"{job_name}: {exc}")
                continue
            if produced:
                report.succeeded += 1
            else:
                report.failed.append(f"{job_name}: {why}")
            done += 1
            if done % 10 == 0 or done == len(remaining):
                _log(f"[af3] {done}/{len(remaining)} done, "
                     f"{report.succeeded} produced output, {len(report.failed)} failed")
    return report
