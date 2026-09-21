#!/usr/bin/env python3
"""
Stage 04, step 2: fold every prepared json with AlphaFold3.

    job_runs/af3/<experiment>/<group>/<sequence_id>_monomer.json  <- read
    outputs/<sequence_id>_monomer/                                <- written

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
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import job_paths as jp  # noqa: E402
from archives import load_table  # noqa: E402

HELPING_SCRIPTS = Path(__file__).resolve().parent
STAGE = HELPING_SCRIPTS.parent.parent

SBATCH_SCRIPT = HELPING_SCRIPTS / "run_af3_one.sbatch"
WS_SCRIPT = HELPING_SCRIPTS / "run_af3_one.sh"

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
                 report: RunReport) -> List[Tuple[str, Path]]:
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
        for json_path in sorted(root.rglob("*_monomer.json")):
            job_name = json_path.stem
            sequence_id = (job_name[: -len("_monomer")]
                           if job_name.endswith("_monomer") else job_name)
            if sequence_id in scored or jp.af3_output_dir(stage, job_name).is_dir():
                report.skipped += 1
                continue
            jobs.append((job_name, json_path))
    return jobs


def run_one(stage: Path, job_name: str, json_path: Path, cluster: bool) -> bool:
    """Fold one sequence. Returns whether AF3 actually produced output."""
    logs = stage / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    if cluster:
        command = [
            "sbatch", "--wait",
            "--job-name", job_name,
            "--chdir", str(stage),
            "--output", str(logs / f"{job_name}-%j.out"),
            "--error", str(logs / f"{job_name}-%j.err"),
            str(SBATCH_SCRIPT), str(json_path), str(stage),
        ]
    else:
        command = ["bash", str(WS_SCRIPT), str(json_path), str(stage)]

    completed = subprocess.run(command)
    produced = jp.af3_output_dir(stage, job_name).is_dir()
    # The exit code alone is not enough on the cluster: sbatch --wait reports
    # the job's status, but a job can exit 0 after AF3 wrote nothing usable.
    if completed.returncode != 0 or not produced:
        return False
    return True


def run_af3(stage: Path, experiment: Optional[str] = None,
            max_concurrent: int = MAX_CONCURRENT) -> RunReport:
    report = RunReport()
    cluster = shutil.which("sbatch") is not None
    script = SBATCH_SCRIPT if cluster else WS_SCRIPT
    if not script.is_file():
        raise Af3Error(f"missing runner script: {script}")

    jobs = pending_jobs(stage, experiment, report)
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
    if run_one(stage, canary_name, canary_json, cluster):
        report.succeeded += 1
    else:
        report.failed.append(canary_name)
    report.submitted += 1

    if report.failed:
        raise Af3Error(
            f"the canary job {canary_name} produced no output. Every AF3 job uses the "
            f"same module, image and database paths, so the remaining {len(jobs) - 1} "
            f"would fail the same way -- nothing else was submitted. "
            f"Check {stage / 'logs'}."
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
                produced = future.result()
            except Exception as exc:  # noqa: BLE001 - one job must not stop the batch
                report.failed.append(f"{job_name}: {exc}")
                continue
            if produced:
                report.succeeded += 1
            else:
                report.failed.append(job_name)
            done += 1
            if done % 10 == 0 or done == len(remaining):
                _log(f"[af3] {done}/{len(remaining)} done, "
                     f"{report.succeeded} produced output, {len(report.failed)} failed")
    return report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT,
                        help=f"cluster jobs in flight (default: {MAX_CONCURRENT}, "
                             f"the account cap)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = run_af3(args.stage.resolve(), args.experiment, args.max_concurrent)
    except Af3Error as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    _log(f"[done] {report.summary()}")
    for failure in report.failed[:10]:
        _log(f"[done]   failed: {failure}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

