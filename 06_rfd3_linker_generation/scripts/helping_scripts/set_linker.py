#!/usr/bin/env python3
"""
Stage 06: fill in the linker length the jsons were left blank for.

    ./set_linker.py --length 14                     # every json in the stage
    ./set_linker.py --length 12-18                  # a range for RFD3 to sample
    ./set_linker.py --length 14 --sequence-id hsx015039_00003901
    ./set_linker.py --length 14 --orientation AB

The contig carries the literal LINKER where its length belongs. This replaces it
and recomputes `length` from what the contig then says, so the two cannot
disagree -- which is the whole reason for a helper rather than an editor: the
total is the motif residues plus the linker, and getting it wrong by hand is
easy and silent.

A range like 12-18 leaves RFD3 to choose within it and `length` spans the same
way. Already-filled jsons are skipped unless --force is given, so re-running
after adding more pairs only touches the new ones.

Nothing here decides the length for you. stage_06_pairs_<experiment>.csv records
the gap each orientation spans and the fewest residues that could reach it fully
extended; a linker at exactly that floor is a taut string, so a real one wants
slack on top.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))

STAGE = Path(__file__).resolve().parents[2]

LINKER_TOKEN = "LINKER"
SEGMENT = re.compile(r"^([A-Za-z])(\d+)(?:-(\d+))?$")
DIFFUSED = re.compile(r"^\d+(?:-\d+)?$")


class SetError(Exception):
    """A length could not be applied."""


def _log(*parts: object) -> None:
    print(*parts, flush=True)


@dataclass
class SetReport:
    filled: int = 0
    already: int = 0
    skipped: int = 0
    problems: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        parts = [f"set: {self.filled} json(s) filled"]
        if self.already:
            parts.append(f"{self.already} already had a length")
        if self.skipped:
            parts.append(f"{self.skipped} did not match the filter")
        return ", ".join(parts)


def parse_length(text: str) -> Tuple[int, int]:
    """'14' -> (14, 14); '12-18' -> (12, 18)."""
    match = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", text)
    if not match:
        raise SetError(f"cannot read a linker length from {text!r}; use 14 or 12-18")
    low = int(match.group(1))
    high = int(match.group(2)) if match.group(2) else low
    if low < 1 or high < low:
        raise SetError(f"linker length {text!r} is not a sensible range")
    return low, high


def motif_residues(contig: str) -> int:
    """How many residues the contig keeps from the input.

    Only chain-prefixed segments count. A bare number or range is a diffused
    region's length, not residues taken from the input -- which is what the
    linker becomes the moment it is filled in, so counting it here would double
    it in `length`.
    """
    total = 0
    for segment in contig.split(","):
        segment = segment.strip()
        if not segment or segment == LINKER_TOKEN or DIFFUSED.fullmatch(segment):
            continue
        match = SEGMENT.fullmatch(segment)
        if not match:
            raise SetError(f"cannot read contig segment {segment!r}")
        first = int(match.group(2))
        last = int(match.group(3)) if match.group(3) else first
        total += last - first + 1
    return total


def apply(path: Path, low: int, high: int, force: bool, report: SetReport) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if len(payload) != 1:
        raise SetError(f"{path.name} holds {len(payload)} designs, expected one")
    name, entry = next(iter(payload.items()))
    contig = str(entry.get("contig", ""))
    if LINKER_TOKEN not in contig:
        if not force:
            report.already += 1
            return
        raise SetError(
            f"{path.name} has no {LINKER_TOKEN} left in its contig, so there is "
            f"nothing to replace -- it was filled in already"
        )

    replacement = str(low) if low == high else f"{low}-{high}"
    entry["contig"] = contig.replace(LINKER_TOKEN, replacement)
    kept = motif_residues(entry["contig"])
    entry["length"] = str(kept + low) if low == high else f"{kept + low}-{kept + high}"

    path.write_text(json.dumps({name: entry}, indent=2) + "\n", encoding="utf-8")
    report.filled += 1
    _log(f"[set] {name}: linker {replacement}, total length {entry['length']}")


def run_set(stage: Path, length: str, experiment: Optional[str] = None,
            sequence_id: Optional[str] = None, orientation: Optional[str] = None,
            force: bool = False) -> SetReport:
    report = SetReport()
    low, high = parse_length(length)

    root = stage / "json"
    if not root.is_dir():
        _log(f"[set] no jsons under {root}")
        return report

    for path in sorted(root.rglob("*.json")):
        if experiment and path.parent.parent.name != experiment:
            report.skipped += 1
            continue
        if sequence_id and not path.stem.startswith(sequence_id):
            report.skipped += 1
            continue
        if orientation and not path.stem.endswith(f"_{orientation}"):
            report.skipped += 1
            continue
        try:
            apply(path, low, high, force, report)
        except (SetError, json.JSONDecodeError, OSError) as exc:
            report.problems.append(f"{path.name}: {exc}")
            _log(f"[set] {path.name}: FAILED ({exc})")
    return report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--length", required=True,
                        help="linker residues: 14, or a range like 12-18")
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--sequence-id", default=None,
                        help="only the jsons for this design")
    parser.add_argument("--orientation", choices=("AB", "BA"), default=None,
                        help="only one direction")
    parser.add_argument("--force", action="store_true",
                        help="fail loudly on a json that was already filled, "
                             "instead of leaving it alone")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = run_set(args.stage.resolve(), args.length, args.experiment,
                         args.sequence_id, args.orientation, args.force)
    except SetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    _log(f"[done] {report.summary()}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
