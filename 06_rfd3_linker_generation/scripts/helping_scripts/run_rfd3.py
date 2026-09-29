#!/usr/bin/env python3
"""
Stage 06, step 2: run RFD3 on the jsons whose linker length you have set.

    json/<experiment>/<sid>_AB.json      <- read; an unfilled one FAILS
    outputs_raw/<experiment>/<key>/<name>.cif    <- RFD3 writes
    outputs_raw/<experiment>/<key>/<name>.json   <- with diffused_index_map

An unfilled json is an ERROR, not a skip. It used to be skipped, on the argument
that you would rather fill a few and come back -- but a skip is silent, and a
json nobody noticed was unfilled looks exactly like one that generated. So it is
reported and the run exits non-zero. What it does NOT do is stop the ready jsons
from running: the incremental loop still works, and the unfilled ones are simply
listed at the end where they cannot be missed.

"Filled" means the contig's diffused slot is a number or a range. The literal
LINKER that linker_jsons.py writes is not, so an untouched json cannot reach
RFD3 by accident -- and neither can one where only `length` was edited and the
contig was forgotten, because both are checked.

DESIGNS_PER_JSON backbones are generated for each, on consecutive seeds. One
linker length does not have one answer: RFD3 places a different path on each
seed, and having several to choose between is the point of generating at all.

The RFD3 invocation itself is stage 01's, imported rather than copied. It knows
how to source the env file, apply the sampler overrides and place the outputs,
and there is no version of this worth maintaining twice. What is local here is
which jsons are ready and how many designs each gets.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import job_paths as jp  # noqa: E402
import partitions  # noqa: E402
import rfd3_dispatch  # noqa: E402
import site_config as site  # noqa: E402

STAGE = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent

LINKER_TOKEN = "LINKER"
DIFFUSED = re.compile(r"^\d+(?:-\d+)?$")
DESIGNS_PER_JSON = 5

# The sampler settings for a LINKER run, which is not a symmetric one.
#
# Stage 01's defaults carry "inference_sampler.kind=symmetry" and a
# sym_step_frac, because everything it generates is a symmetric dimer. A linker
# runs from one copy's C-terminus to the other's N-terminus and no symmetry
# operation maps that onto itself, so the jsons written for it declare no
# symmetry block at all -- and handing RFD3 the symmetry sampler anyway makes it
# stop with "Symmetry transform not found".
#
# Deliberately short. Every setting stage 01 needs for exact symmetry is left
# off rather than translated: kind, sym_step_frac, allow_realignment and
# use_classifier_free_guidance all exist to hold a symmetric motif in place, and
# guessing which of them the default sampler still accepts is how you get a
# hydra error instead of a structure. Add any you want with --override.
RFD3_LINKER_OVERRIDES = (
    "diffusion_batch_size=1",
    "n_batches=1",
    "skip_existing=False",
    "inference_sampler.num_timesteps=200",
)
ENV_FILE_REL = Path("scripts") / "env" / "rfd3.env"


class Rfd3Error(Exception):
    """RFD3 could not be run."""


def _log(*parts: object) -> None:
    print(*parts, flush=True)


@dataclass
class RunReport:
    ran: int = 0
    unfilled: List[str] = field(default_factory=list)
    already: int = 0
    failed: List[str] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems and not self.failed and not self.unfilled

    def summary(self) -> str:
        parts = [f"rfd3: {self.ran} design(s) generated"]
        if self.already:
            parts.append(f"{self.already} already there")
        if self.unfilled:
            parts.append(f"{len(self.unfilled)} json(s) STILL HOLD {LINKER_TOKEN}")
        if self.failed:
            parts.append(f"{len(self.failed)} FAILED")
        return ", ".join(parts)


def linker_slot(contig: str) -> Optional[str]:
    """The diffused segment of a linker contig: the one with no chain letter.

    Returns None when the contig has no such segment, which means the file was
    not written by this stage.
    """
    for segment in contig.split(","):
        segment = segment.strip()
        if segment == LINKER_TOKEN or DIFFUSED.fullmatch(segment):
            return segment
    return None


def is_filled(path: Path) -> Tuple[bool, str]:
    """(ready to run, why not). A json is ready only when BOTH fields are set.

    `length` is checked as well as the contig because set_linker.py writes them
    together and nothing else should: a file where they disagree was edited by
    hand, and running it would generate a construct of a length nobody chose.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"unreadable ({exc})"
    if not isinstance(payload, dict) or len(payload) != 1:
        return False, "not a single-design json"

    entry = next(iter(payload.values()))
    if not isinstance(entry, dict):
        return False, "design entry is not an object"

    contig = str(entry.get("contig", ""))
    slot = linker_slot(contig)
    if slot is None:
        return False, "no diffused segment in the contig"
    if slot == LINKER_TOKEN:
        return False, "linker length not set"

    length = str(entry.get("length", ""))
    if LINKER_TOKEN in length:
        return False, "contig was filled but length was not"
    if not DIFFUSED.fullmatch(length.strip()):
        return False, f"length {length!r} is not a number or a range"
    return True, ""


def input_problem(stage: Path, path: Path) -> Optional[str]:
    """Why this json's input pdb cannot be read, or None if it can.

    Checked here rather than left to RFD3. run_one_job rewrites "input" only for
    a design that declares symmetry, and a linker json declares none, so the
    path goes to RFD3 exactly as written -- and what comes back when it is wrong
    is RFD3's own failure, several hundred lines into a log, naming a path that
    means nothing without knowing which stage wrote it.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        entry = next(iter(payload.values()))
        raw = str(entry.get("input", ""))
    except (OSError, json.JSONDecodeError, StopIteration, AttributeError) as exc:
        return f"cannot read its input path ({exc})"
    if not raw:
        return "has no input path"
    if jp.resolve_seed_path(stage, raw, path) is not None:
        return None
    # A path from another machine is NOT this error any more -- resolve_seed_path
    # re-anchors one onto this checkout. So by the time this fires, the pdb is
    # genuinely absent from this project, and the remedy is to lay it out rather
    # than to regenerate the json. Saying "delete the jsons and re-run stage 05",
    # which this used to, would throw away linker lengths set by hand.
    expected = (jp.pair_dir(stage, "<experiment>", "<group>") / Path(raw).name)
    return (f"its input pdb is not there: {raw}\n"
            f"        Nothing in this checkout matches it either, so the pdb has "
            f"not been laid out yet. Stage 06 unpacks it from the handover "
            f"archive:\n"
            f"            {SCRIPT_DIR / 'unpack_inputs.py'}\n"
            f"        or just run the launcher, which unpacks before it "
            f"generates. It belongs at\n"
            f"            {expected}\n"
            f"        Leave the json alone -- its linker length is yours, and "
            f"a path written on another machine is re-anchored automatically.")


def ready_jsons(stage: Path, experiment: str, report: RunReport,
                sequence_id: Optional[str] = None,
                orientation: Optional[str] = None) -> List[Tuple[str, Path]]:
    """(json_rel, path) for every json with a linker length.

    An unfilled one is recorded as a failure of that json and left out of the
    list. The others still run -- one unset length must not cost you the
    generation of every pair you did fill in.
    """
    directory = jp.experiment_json_dir(stage, experiment)
    if not directory.is_dir():
        return []
    ready: List[Tuple[str, Path]] = []
    for path in sorted(directory.glob("*.json")):
        # Filtered before anything is judged, so a json you did not ask for is
        # not reported as unfilled either. Same spellings as set_linker.py, so
        # the flags that chose a length choose what to generate from it.
        if sequence_id and not path.stem.startswith(sequence_id):
            continue
        if orientation and not path.stem.endswith(f"_{orientation}"):
            continue
        filled, reason = is_filled(path)
        if not filled:
            report.unfilled.append(f"{experiment}/{path.name}: {reason}")
            _log(f"[rfd3] {path.name}: ERROR -- {reason}")
            continue
        missing = input_problem(stage, path)
        if missing:
            report.problems.append(f"{experiment}/{path.name}: {missing}")
            _log(f"[rfd3] {path.name}: ERROR -- {missing}")
            continue
        ready.append((path.name, path))
    return ready


def archived_designs(stage: Path, experiment: str) -> set:
    """The design names already inside this experiment's passed archives.

    The runner decides a job is done by looking for its raw cif, and cleanup
    deletes those as soon as they are archived -- so on every run after the
    first, the raw file is gone and the runner would generate the whole batch
    again. Twenty designs of stub output costs nothing; twenty of RFD3 costs a
    GPU hour. The archive is the record of what exists, so that is what is
    asked.
    """
    found: set = set()
    directory = jp.sorted_clean_dir(stage, experiment, "passed")
    if not directory.is_dir():
        return found
    for archive_path in sorted(directory.glob("*.tar.gz")):
        try:
            with tarfile.open(archive_path, "r:gz") as archive:
                for name in archive.getnames():
                    if name.endswith(".cif"):
                        found.add(Path(name).stem)
        except (OSError, tarfile.TarError):
            continue
    return found


def runner_path() -> Path:
    """Stage 01's RFD3 job runner: one file, used by both stages.

    Returned as a path rather than an imported module because the cluster route
    does not import it -- it runs it as a script on a compute node, and the
    dispatcher needs somewhere to point the sbatch script at. On a workstation
    the dispatcher imports the same path in-process.
    """
    root = Path(__file__).resolve().parents[3] / jp.STAGE01_DIRNAME
    runner = root / "scripts" / "helping_scripts" / "run_one_job.py"
    if not runner.is_file():
        raise Rfd3Error(
            f"stage 01's RFD3 runner is not at {runner}; stage 06 drives RFD3 "
            f"through it rather than carrying its own copy"
        )
    return runner


def check_env(stage: Path) -> None:
    env_file = stage / ENV_FILE_REL
    if env_file.is_file():
        return
    stage01 = stage.parent / jp.STAGE01_DIRNAME / ENV_FILE_REL
    raise Rfd3Error(
        f"no RFD3 env file at {env_file}.\n"
        f"  It defines RFD3_EXE and CKPT, and stage 01 already has one:\n"
        f"      mkdir -p {env_file.parent}\n"
        f"      ln -s {stage01} {env_file}\n"
        f"  A symlink rather than a copy, so both stages stay on one install."
    )


def build_rows(stage: Path, experiment: str, jobs: Sequence[Tuple[str, Path]],
               designs: int, done: set, report: RunReport,
               limit: Optional[int] = None) -> List[Tuple[str, int]]:
    """(json_rel, seed) for everything still to generate.

    Consecutive seeds, 1..designs. The runner names the output from the seed,
    so a re-run with a larger --designs adds the new ones and leaves the
    existing ones alone rather than regenerating them.

    The run list holds ONLY outstanding work, which is why the dispatcher is
    told which designs are already archived: it verifies every row afterwards,
    and a row it cannot find on disk and cannot find in an archive is a
    failure. Stage 06 archives into sorted_clean/<experiment>/passed/, not the
    outputs_clean/ the dispatcher reads by default.
    """
    rows: List[Tuple[str, int]] = []
    for json_rel, _ in jobs:
        group_key = jp.group_key_from_json_rel(json_rel)
        for seed in range(1, designs + 1):
            if limit is not None and len(rows) >= limit:
                _log(f"[rfd3] stopping at --limit {limit}")
                return rows
            if jp.job_name(group_key, seed) in done:
                report.already += 1
                continue
            rows.append((json_rel, seed))
    return rows


def run_experiment(stage: Path, experiment: str, report: RunReport,
                   designs: int = DESIGNS_PER_JSON,
                   overrides: Sequence[str] = RFD3_LINKER_OVERRIDES,
                   sequence_id: Optional[str] = None,
                   orientation: Optional[str] = None,
                   limit: Optional[int] = None) -> None:
    check_env(stage)

    jobs = ready_jsons(stage, experiment, report, sequence_id, orientation)
    if not jobs:
        _log(f"[rfd3] {experiment}: nothing is ready to run")
        return
    _log(f"[rfd3] {experiment}: {len(jobs)} json(s) ready x {designs} design(s)")
    _log(f"[rfd3] sampler: {' '.join(overrides)}")

    done = archived_designs(stage, experiment)
    rows = build_rows(stage, experiment, jobs, designs, done, report, limit)
    if not rows:
        _log(f"[rfd3] {experiment}: every design already exists ({report.already})")
        return

    # Same run list, same dispatcher, same canary as stage 01 -- the only thing
    # stage 06 supplies that stage 01 does not is its own sampler, which is the
    # one thing that could not previously travel to a compute node.
    run_list = jp.run_list_path(stage, experiment)
    jp.write_run_list(run_list, rows)
    _log(f"[rfd3] {len(rows)} job(s) queued in {run_list}")

    dispatched = rfd3_dispatch.dispatch(experiment, run_list, stage,
                                        runner_path(), overrides, done)
    report.ran += len(rows) - len(dispatched.failed)
    for json_rel, seed in dispatched.failed:
        missing = jp.expected_raw_cif(stage, experiment, json_rel, seed)
        report.failed.append(f"{json_rel}#{seed}: no structure at {missing}")


def run_rfd3(stage: Path, experiment: Optional[str] = None,
             designs: int = DESIGNS_PER_JSON,
             overrides: Sequence[str] = RFD3_LINKER_OVERRIDES,
             sequence_id: Optional[str] = None,
             orientation: Optional[str] = None,
             limit: Optional[int] = None) -> RunReport:
    report = RunReport()
    root = jp.json_root(stage)
    if not root.is_dir():
        _log(f"[rfd3] no jsons under {root}")
        return report
    names = [experiment] if experiment else sorted(
        path.name for path in root.iterdir() if path.is_dir()
    )
    for experiment_name in names:
        run_experiment(stage, experiment_name, report, designs, overrides,
                       sequence_id, orientation, limit)
    return report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--designs", type=int, default=DESIGNS_PER_JSON,
                        help=f"backbones per json, on consecutive seeds "
                             f"(default: {DESIGNS_PER_JSON})")
    parser.add_argument("--sequence-id", default=None,
                        help="only the jsons for this design")
    parser.add_argument("--orientation", choices=("AB", "BA"), default=None,
                        help="only one direction")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="stop after N RFD3 invocations IN TOTAL. --designs is "
                             "per json, so --designs 1 over 28 jsons is 28 jobs; "
                             "--limit 1 is one job")
    parser.add_argument("--override", action="append", default=None, metavar="KEY=VALUE",
                        help="an extra RFD3 sampler override, repeatable. Added to "
                             "the linker defaults, which carry no symmetry settings")
    parser.add_argument("--list", action="store_true",
                        help="say which jsons are ready and which are not, run nothing")
    parser.add_argument("--partition", default=None,
                        help="submit to this Slurm partition instead of asking "
                             "(cluster only; PIPELINE_PARTITION does the same)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    stage = args.stage.resolve()
    if args.list:
        report = RunReport()
        names = [args.experiment] if args.experiment else sorted(
            path.name for path in jp.json_root(stage).iterdir() if path.is_dir()
        ) if jp.json_root(stage).is_dir() else []
        for name in names:
            ready = ready_jsons(stage, name, report,
                                args.sequence_id, args.orientation)
            for json_rel, _ in ready:
                _log(f"[rfd3] {json_rel}: ready")
        _log(f"[done] {len(names)} experiment(s), "
             f"{len(report.unfilled)} not ready")
        return 0
    # This file can be run on its own, so it settles the machine itself rather
    # than relying on the launcher having done it.
    try:
        site.apply()
        if site.mode() == "cluster":
            partitions.choose(args.partition)
    except (site.SiteError, partitions.PartitionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    try:
        overrides = tuple(RFD3_LINKER_OVERRIDES) + tuple(args.override or ())
        report = run_rfd3(stage, args.experiment, args.designs, overrides,
                          args.sequence_id, args.orientation, args.limit)
    except Rfd3Error as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    _log(f"[done] {report.summary()}")
    for failure in report.failed[:10]:
        _log(f"[done]   {failure}")
    for unfilled in report.unfilled[:10]:
        _log(f"[done]   {unfilled}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
