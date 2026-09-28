"""
Reading a stage-01 design config, for the stages downstream of it.

Every stage after generation needs to ask the same question: what did stage 01
actually ask RFD3 to build here? Stage 02 needs the seed to align against;
stage 03 needs the scaffolded residues, so ProteinMPNN leaves them alone. Both
answers live in the json the group was generated from, and the group key is
that json's filename stem -- so the config is recoverable from an archive path
with nothing recorded along the way.

Keeping this in one place matters more than it looks: referenced_residues()
defines what counts as a scaffolded residue. If stage 01 validated one set and
stage 03 froze a different one, nothing would raise and the designs would
quietly differ from what was asked for.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import job_paths as jp  # noqa: E402


class DesignError(Exception):
    """A design config could not be read or does not say what is needed."""


def split_map_key(key: str) -> Tuple[str, int]:
    """'A24' -> ('A', 24).

    diffused_index_map's keys name seed residues, its values name the
    generated residues they became -- both carry the chain letter, which is
    why neither stage has to be told what its chains are called.
    """
    match = re.fullmatch(r"(.+?)(-?\d+)", key)
    if not match:
        raise DesignError(f"cannot parse residue key {key!r}")
    return match.group(1), int(match.group(2))


RESIDUE_KEY = re.compile(r"^\s*([A-Za-z])(\d+)(?:-(\d+))?\s*$")


def expand_residue_key(key: str) -> List[str]:
    """'B5' -> ['B5']; 'B28-32' -> ['B28','B29','B30','B31','B32'].

    select_fixed_atoms accepts a range for a run of residues, and every caller
    downstream works one residue at a time -- diffused_index_map is keyed per
    residue, and so is everything derived from it. Expanding here is what lets
    both spellings mean the same thing.

    Without this the range form is not rejected, it is misread: split_map_key
    backtracks 'B28-32' into chain 'B28', residue -32, which matches nothing in
    diffused_index_map, so those five residues are quietly never held and no
    error is raised anywhere.

    An unparseable key returns empty rather than raising: the callers decide
    what an unusable config means, and they already do.
    """
    match = RESIDUE_KEY.match(str(key))
    if not match:
        return []
    chain, first = match.group(1), int(match.group(2))
    last = int(match.group(3)) if match.group(3) else first
    if last < first:
        return []
    return [f"{chain}{number}" for number in range(first, last + 1)]


def referenced_residues(config: dict) -> Set[Tuple[str, int]]:
    """(chain, residue number) pairs a design config points at.

    Covers contig motif segments ('A24-36', 'A5', 'B28-32'), select_fixed_atoms
    keys and select_exposed entries. Bare numbers in a contig are designed-region
    lengths rather than residues, so they are skipped.
    """
    wanted: Set[Tuple[str, int]] = set()

    for segment in str(config.get("contig", "")).split(","):
        match = re.fullmatch(r"\s*([A-Za-z])(\d+)(?:-(\d+))?\s*", segment)
        if match:
            chain, first = match.group(1), int(match.group(2))
            last = int(match.group(3)) if match.group(3) else first
            wanted |= {(chain, n) for n in range(first, last + 1)}

    for key in config.get("select_fixed_atoms", {}) or {}:
        for expanded in expand_residue_key(key):
            match = re.fullmatch(r"([A-Za-z])(\d+)", expanded)
            if match:
                wanted.add((match.group(1), int(match.group(2))))

    for key in str(config.get("select_exposed", "")).split(","):
        match = re.fullmatch(r"\s*([A-Za-z])(\d+)\s*", key)
        if match:
            wanted.add((match.group(1), int(match.group(2))))

    return wanted


def design_config_for_group(stage: Path, experiment: str, group_key: str) -> Tuple[Path, dict]:
    """(json path, the one design config) a group was generated from.

    A group comes from exactly one json; a json holding several designs that
    disagree about their input is rejected rather than guessed at, because
    which one a given structure came from would be unknowable.
    """
    json_path = jp.stage01_json_path(stage, experiment, group_key)
    if not json_path.is_file():
        raise DesignError(
            f"no stage-01 json for group {group_key!r}: {json_path}\n"
            f"  the archive name must match the json that generated it"
        )
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        entries = jp.design_entries(data)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise DesignError(f"could not read {json_path}: {exc}")

    if len(entries) == 1:
        return json_path, next(iter(entries.values()))

    inputs = {str(config.get("input", "")) for config in entries.values()}
    if len(inputs) != 1:
        raise DesignError(
            f"{json_path} holds {len(entries)} designs naming {len(inputs)} different "
            f"seeds ({sorted(inputs)}); a group must come from exactly one"
        )
    # Several designs, same seed: contigs may still differ, so refuse rather
    # than pick one and silently freeze the wrong residues.
    contigs = {str(config.get("contig", "")) for config in entries.values()}
    if len(contigs) != 1:
        raise DesignError(
            f"{json_path} holds {len(entries)} designs with different contigs; "
            f"split them into one json per design so each group has one config"
        )
    return json_path, next(iter(entries.values()))


FIXED_ALL = "ALL"


def fixed_residues(config: dict) -> List[str]:
    """The seed residues whose SIDE CHAINS stage 01 fixed: select_fixed_atoms
    entries valued "ALL".

    "BKBN" entries are deliberately excluded -- only their backbone was
    constrained, so their side-chain identity was never preserved and there is
    nothing there worth freezing or fitting on.

    Returns seed-numbered keys ('B5', 'B29'), sorted, one per residue: a range
    key like 'B28-32' is expanded, because everything downstream is keyed per
    residue. Empty when the config has no select_fixed_atoms, which the caller
    must treat as "this design cannot be aligned or redesigned" rather than as
    zero work to do.
    """
    selected = config.get("select_fixed_atoms", {}) or {}
    keys = [expanded for key, value in selected.items()
            if str(value).strip().upper() == FIXED_ALL
            for expanded in expand_residue_key(key)]
    return sorted(set(keys),
                  key=lambda key: (split_map_key(key)[0], split_map_key(key)[1]))


def fixed_residue_map(config: dict, diffused_index_map: dict) -> Tuple[Dict[str, str], List[str]]:
    """{seed key: generated key} for the side-chain-fixed residues of ONE
    structure, plus the ones diffused_index_map had no entry for.

    Both numberings are kept because both are needed: stage 02 pairs them to
    fit the design back onto the seed, stage 03 takes the generated side to
    tell ProteinMPNN what not to touch.
    """
    wanted = fixed_residues(config)
    if not wanted:
        raise DesignError(
            "the design config fixes no side chains: select_fixed_atoms is absent "
            "or has no entries valued \"ALL\", so there is nothing to align on "
            "or to hold fixed"
        )
    mapped: Dict[str, str] = {}
    unmapped: List[str] = []
    for key in wanted:
        if key in diffused_index_map:
            mapped[key] = str(diffused_index_map[key])
        else:
            unmapped.append(key)
    if not mapped:
        raise DesignError(
            f"none of the {len(wanted)} side-chain-fixed residue(s) {wanted} appear "
            f"in diffused_index_map"
        )
    return mapped, unmapped


def generated_positions(mapped: Dict[str, str], chains: Sequence[str]) -> Dict[str, List[int]]:
    """The fixed residues as ProteinMPNN wants them: {chain: [position, ...]},
    mirrored across every generated chain because the design is symmetric and
    the chains are tied."""
    positions = sorted({split_map_key(value)[1] for value in mapped.values()})
    return {chain: list(positions) for chain in chains}


# ---------------------------------------------------------------------------
# The record stage 01 writes, and the stages downstream read
# ---------------------------------------------------------------------------


def record_line(protein_id: str, group_key: str, mapped: Dict[str, str]) -> str:
    """One JSONL line: which residues were held fixed, in both numberings."""
    return json.dumps(
        {"protein_id": protein_id, "group": group_key, "fixed": mapped},
        sort_keys=True,
    ) + "\n"


def read_records(path: Path) -> Dict[str, Dict[str, str]]:
    """{protein_id: {seed key: generated key}} from the record stage 01 wrote.

    A missing file is not an error: structures generated before the record
    existed simply are not in it, and the caller derives their set instead.
    Later lines win, so re-generating a structure supersedes its old entry.
    """
    records: Dict[str, Dict[str, str]] = {}
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                records[payload["protein_id"]] = dict(payload["fixed"])
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise DesignError(f"{path}:{lineno}: malformed record: {exc}")
    return records
