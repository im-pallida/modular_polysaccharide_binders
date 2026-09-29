#!/usr/bin/env python3
"""
Which Slurm partition this run should submit to, chosen once and asked once.

WHY IT IS ASKED AT ALL

The partition was written into four .sbatch files -- gpu_h200 in three of them,
gpu_l40s in the fourth. That is one cluster's naming, so the pipeline could not
be submitted anywhere else without editing the job scripts, and on a busy day
it queued behind everything else on a partition that happened to be full while
another sat idle.

So the partition is picked per run, from what sinfo actually reports, with the
emptiest GPU partition recommended. A command-line option overrides it and
skips the question entirely.

PRECEDENCE, highest first

    1. --partition on the launcher
    2. PIPELINE_PARTITION in the environment or site.local.env
    3. the answer to the prompt
    4. the recommendation, when nothing can be asked

ASKED ONCE

choose() memoises, so a stage that submits four hundred jobs asks once and the
rest read the answer. It is called from the launcher's main(), before any
thread pool starts -- a prompt racing four worker threads is a prompt nobody
can answer.

NEVER BLOCKS A BATCH RUN

If stdin is not a terminal -- a scheduled run, output piped to a file, a job
script -- there is nobody to answer, so the recommendation is taken and said
out loud rather than waited on. A pipeline that hangs on a question in the
middle of the night has failed more expensively than one that guessed.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

SINFO_FORMAT = "%R|%a|%l|%D|%T|%G"
CHOSEN: Optional[str] = None          # memoised answer for this process
ASKED = False


class PartitionError(Exception):
    """The requested partition does not exist on this cluster."""


def _log(*parts: object) -> None:
    print(*parts, flush=True)


@dataclass
class Partition:
    name: str
    up: bool = True
    time_limit: str = ""
    idle: int = 0
    total: int = 0
    pending: int = 0
    gpu: bool = False

    @property
    def free_fraction(self) -> float:
        return self.idle / self.total if self.total else 0.0

    def row(self, marker: str = " ") -> str:
        kind = "gpu" if self.gpu else "cpu"
        state = "" if self.up else "  (down)"
        return (f" {marker} {self.name:<16} {self.idle:>3} / {self.total:<4} idle"
                f"  {self.pending:>4} queued  {self.time_limit:<12} {kind}{state}")


def _run(command: Sequence[str]) -> str:
    try:
        done = subprocess.run(list(command), capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout if done.returncode == 0 else ""


def discover() -> List[Partition]:
    """Every partition sinfo reports, with idle nodes and queue depth.

    sinfo prints one line per partition AND state, so a partition with mixed,
    idle and allocated nodes arrives as three lines that have to be summed --
    reading only the first would report a busy partition as empty.
    """
    if shutil.which("sinfo") is None:
        return []
    found: Dict[str, Partition] = {}
    for line in _run(["sinfo", "-h", "-o", SINFO_FORMAT]).splitlines():
        fields = line.strip().split("|")
        if len(fields) < 5:
            continue
        name = fields[0].strip().rstrip("*")      # sinfo marks the default with *
        if not name:
            continue
        entry = found.setdefault(name, Partition(name))
        entry.up = entry.up and fields[1].strip().lower() in ("up", "")
        entry.time_limit = entry.time_limit or fields[2].strip()
        try:
            count = int(fields[3].strip())
        except ValueError:
            count = 0
        entry.total += count
        if fields[4].strip().lower().startswith("idle"):
            entry.idle += count
        gres = fields[5].strip().lower() if len(fields) > 5 else ""
        entry.gpu = entry.gpu or "gpu" in gres or "gpu" in name.lower()

    for line in _run(["squeue", "-h", "-t", "PD", "-o", "%P"]).splitlines():
        name = line.strip().rstrip("*")
        if name in found:
            found[name].pending += 1
    return sorted(found.values(), key=lambda item: item.name)


def offered(quiet: bool = True) -> List[Partition]:
    """The partitions this account may submit to: discovery, then the allow-list.

    One implementation, used by both the prompt and `python3 partitions.py`.
    They disagreed before -- the report listed all thirteen while the prompt
    offered three -- which makes the report useless for checking whether the
    setting took effect, the one thing you would run it for.
    """
    available = discover()
    allowed = allowed_names()
    if not allowed:
        return available
    named = {item.name for item in available}
    kept = [item for item in available if item.name in allowed]
    if not kept:
        if not quiet:
            _log(f"[partition] none of PIPELINE_PARTITIONS ({', '.join(allowed)}) "
                 f"exist here; offering all of them instead")
        return available
    missing = [name for name in allowed if name not in named]
    if missing and not quiet:
        _log(f"[partition] not on this cluster, ignored: {', '.join(missing)}")
    return kept


def recommend(available: Sequence[Partition]) -> Optional[Partition]:
    """The emptiest usable GPU partition.

    GPU first, because every stage that submits needs one. Then most idle
    nodes, then shortest queue -- idle capacity is what decides when a job
    starts, and the pending count breaks the tie when nothing is idle anywhere.
    """
    usable = [item for item in available if item.up]
    gpu = [item for item in usable if item.gpu]
    pool = gpu or usable
    if not pool:
        return None
    return sorted(pool, key=lambda item: (-item.idle, item.pending, item.name))[0]


def _setting(name: str) -> str:
    """One setting, from the environment then site.local.env."""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import site_config
        return site_config.read_local().get(name, "").strip()
    except Exception:  # noqa: BLE001 - a missing or odd local file must not stop a run
        return ""


def allowed_names() -> List[str]:
    """PIPELINE_PARTITIONS: the ones this account may actually submit to.

    sinfo lists everything the cluster has, which on a shared system is mostly
    partitions you have no allocation on -- thirteen here, of which three are
    usable. Offering the other ten is not neutral: the recommendation lands on
    whichever is emptiest, and the emptiest partition is usually the one nobody
    can use.

    Machine-specific, so it belongs in site.local.env and not in the repo:

        PIPELINE_PARTITIONS=gpu_l40s,gpu_h200,gpu_h100

    Unset, every GPU partition is offered, which is the old behaviour.
    """
    raw = _setting("PIPELINE_PARTITIONS")
    return [name.strip() for name in raw.replace(" ", ",").split(",") if name.strip()]


def ask(available: Sequence[Partition], suggested: Optional[Partition]) -> Optional[str]:
    """Show the table and read a name. Enter takes the recommendation."""
    _log("")
    _log("[partition] this run submits to the cluster. Available partitions:")
    _log("")
    for index, item in enumerate(available, start=1):
        marker = "*" if suggested is not None and item.name == suggested.name else " "
        _log(f"{index:>3}) {item.row(marker)}")
    _log("")
    if suggested is not None:
        _log(f"[partition] * recommended: {suggested.name} "
             f"({suggested.idle} idle node(s), {suggested.pending} job(s) queued)")

    default = suggested.name if suggested else ""
    names = {item.name for item in available}
    ordered = list(available)
    for _ in range(3):
        try:
            answer = input(f"partition [{default}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            _log("")
            return default or None
        if not answer:
            return default or None
        if answer in names:
            return answer
        # A number as well as a name: a pasted or doubled name is easy to get
        # wrong in a terminal, and "3" is not.
        if answer.isdecimal() and 1 <= int(answer) <= len(ordered):
            return ordered[int(answer) - 1].name
        _log(f"  '{answer}' is not a partition you can submit to. "
             f"One of: {', '.join(sorted(names))}")
    return default or None


def choose(requested: Optional[str] = None, interactive: Optional[bool] = None) -> Optional[str]:
    """The partition for this run. Asks at most once per process.

    Returns None when there is nothing to choose -- no sinfo, or no partitions
    -- and the callers then submit without --partition, leaving whatever the
    .sbatch files say. That is the old behaviour, which is the right thing to
    fall back to.
    """
    global CHOSEN, ASKED
    if requested:
        CHOSEN, ASKED = requested, True
    if ASKED:
        return CHOSEN

    settings = _setting("PIPELINE_PARTITION")
    available = offered(quiet=False)
    if settings:
        names = {item.name for item in available}
        if available and settings not in names:
            raise PartitionError(
                f"PIPELINE_PARTITION={settings!r} is not a partition on this "
                f"cluster. One of: {', '.join(sorted(names))}"
            )
        CHOSEN, ASKED = settings, True
        _log(f"[partition] {settings} (from PIPELINE_PARTITION)")
        return CHOSEN

    ASKED = True
    if not available:
        _log("[partition] sinfo reported nothing; submitting with whatever the "
             "sbatch scripts specify")
        CHOSEN = None
        return CHOSEN

    suggested = recommend(available)
    can_ask = sys.stdin.isatty() if interactive is None else interactive
    if not can_ask:
        CHOSEN = suggested.name if suggested else None
        _log(f"[partition] {CHOSEN} (nothing attached to ask, took the "
             f"recommendation)")
        return CHOSEN

    CHOSEN = ask(available, suggested)
    _log(f"[partition] submitting to {CHOSEN}")
    return CHOSEN


def sbatch_args() -> List[str]:
    """['--partition=<name>'], or nothing if no choice was made.

    Passed on the command line rather than written into the .sbatch files,
    because a command-line option overrides an in-file #SBATCH directive and
    the files then stop being one cluster's property.
    """
    return [f"--partition={CHOSEN}"] if CHOSEN else []


def reset() -> None:
    """Forget the answer. For tests."""
    global CHOSEN, ASKED
    CHOSEN, ASKED = None, False


def main(argv: Optional[Sequence[str]] = None) -> int:
    """`python3 common/partitions.py` reports what this cluster offers."""
    available = offered(quiet=False)
    if not available:
        _log("no sinfo on this machine, or it reported no partitions")
        return 1
    allowed = allowed_names()
    setting = ", ".join(allowed) if allowed else "(unset -- every GPU partition is offered)"
    _log(f"PIPELINE_PARTITIONS = {setting}")
    suggested = recommend(available)
    for index, item in enumerate(available, start=1):
        _log(f"{index:>3}) {item.row('*' if suggested and item.name == suggested.name else ' ')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
