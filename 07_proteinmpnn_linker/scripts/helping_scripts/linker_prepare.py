#!/usr/bin/env python3
"""
Stage 07, step 1: work out which residues of the fused construct may be designed.

    inputs/<experiment>/<group>.tar.gz                 <- stage 06: <name>.cif.gz
                                                          and <name>.json
    inputs_prepared/<experiment>/<group>/<name>.pdb    <- the construct
    job_runs/mpnn/<experiment>/<group>/fixed_positions.jsonl
    job_runs/mpnn/<experiment>/<group>/tied_positions.jsonl
    job_runs/mpnn/<experiment>/<group>/linker.jsonl    <- what was found, and why

RFD3 built the linker; nothing has given it a sequence yet. Only the linker and
what it touches are designed here -- the two copies were designed with the
ligand in context, folded and validated, and redesigning them now would throw
that away to solve a problem they are not part of.

Which residues are the linker is READ, not counted. RFD3's metadata carries
diffused_index_map, whose values name the residues that came from the input, so
the linker is every residue of the generated structure with no entry there. The
alternative -- taking the contig's numbers and assuming positions 126 to 143 are
new -- is right only as long as RFD3 never renumbers or reorders, and wrong in
silence the day it does.

Around the linker, everything within SHELL_CUTOFF of it is opened up too: a new
loop lands against a surface that was designed for a different neighbour, and
leaving that surface untouched leaves the contact unoptimised. Measured on heavy
atoms, any atom to any atom, because a side chain reaches where its CA does not.

THE SHELL IS MIRRORED, and this is the point of the file.

That shell is asymmetric by nature. The linker leaves copy A's C-terminus and
arrives at copy B's N-terminus, so it lies against a different face of each one,
and redesigning each face independently gave two copies with DIFFERENT
sequences -- visible at stage 08 as one chain whose two halves no longer match,
and a construct that is no longer two copies of one binder.

So the shell is converted to offsets within a copy, the two copies' offsets are
unioned, and that union is designed in BOTH. It costs a handful of extra
positions per copy and it makes the two units identical again. The pairing is
then made binding rather than hoped for: --tied_positions_jsonl ties offset i of
copy A to offset i of copy B, so ProteinMPNN averages their logits and decodes
one amino acid for both. Untied, the same positions would be free to diverge and
usually would.

The linker itself is not tied to anything -- it is one segment, not two.

Everything else is fixed. ProteinMPNN's --fixed_positions_jsonl names what must
NOT change, so that is what is written.

Nothing carries a fibril. There used to be one set aside per pair, and this
stage copied it along for a later holo fold; stage 05 stopped producing one when
both fibres were dropped after the alignment. A fibre wanted again is regenerable
by re-running stage 05, which still has the reference complexes in its inputs.
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import gemmi
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import job_paths as jp  # noqa: E402
from archives import extract_member  # noqa: E402
from structures import classify_chains, drop_chains  # noqa: E402

STAGE = Path(__file__).resolve().parents[2]

SHELL_CUTOFF = 8.0     # angstroms, any heavy atom of a residue to any of the linker
MIN_LINKER = 1         # a construct with no diffused residue is not a fusion


class PrepareError(Exception):
    """One construct could not be prepared."""


def _log(*parts: object) -> None:
    print(*parts, flush=True)


def why_failed(failed: Dict[str, str], limit: int = 5) -> str:
    """The reasons, for an exception message that would otherwise be a count.

    This stage logs each failure as it happens, so the reason is already on
    screen -- but the launcher reports only the raised message, and that one
    line should stand on its own in a job log somebody reads a day later.
    """
    items = sorted(failed.items())
    shown = "; ".join(f"{name}: {reason}" for name, reason in items[:limit])
    if len(items) > limit:
        shown += f" (+{len(items) - limit} more)"
    return shown or "no reason recorded"


@dataclass
class PrepareResult:
    experiment: str
    group_key: str
    prepared_dir: Path
    fixed_positions: Path
    tied_positions: Path
    spec_path: Path
    prepared: List[str] = field(default_factory=list)
    already: List[str] = field(default_factory=list)
    failed: Dict[str, str] = field(default_factory=dict)
    linker_sizes: List[int] = field(default_factory=list)
    shell_sizes: List[int] = field(default_factory=list)
    tied_sizes: List[int] = field(default_factory=list)

    @property
    def nothing_to_do(self) -> bool:
        return not self.prepared and not self.failed and bool(self.already)

    @property
    def ok(self) -> bool:
        return not self.failed and (bool(self.prepared) or bool(self.already))


def is_amino_acid(residue: gemmi.Residue) -> bool:
    info = gemmi.find_tabulated_residue(residue.name)
    return info is not None and info.is_amino_acid()


def heavy_coords(residue: gemmi.Residue) -> np.ndarray:
    points = [[atom.pos.x, atom.pos.y, atom.pos.z] for atom in residue
              if atom.element != gemmi.Element("H")]
    return np.array(points) if points else np.zeros((0, 3))


@dataclass(frozen=True)
class CopyLayout:
    """Where the two copies sit in the fused chain, in 1-based positions.

    length is the residue count of ONE copy; a_start and b_start are where each
    begins. The linker is everything between them.
    """
    chain: str
    length: int
    a_start: int
    b_start: int
    linker_start: int
    linker_end: int

    def offset_of(self, position: int) -> int:
        """Which residue of a copy this is, counting from 1 within either."""
        if self.a_start <= position < self.a_start + self.length:
            return position - self.a_start + 1
        if self.b_start <= position < self.b_start + self.length:
            return position - self.b_start + 1
        raise PrepareError(
            f"residue {position} of chain {self.chain} is in neither copy "
            f"({self.a_start}-{self.a_start + self.length - 1} and "
            f"{self.b_start}-{self.b_start + self.length - 1})"
        )

    def twins(self, offset: int) -> Tuple[int, int]:
        """The two positions -- one per copy -- that hold the same offset."""
        return self.a_start + offset - 1, self.b_start + offset - 1

    def as_record(self) -> Dict[str, int]:
        return {"length": self.length, "a_start": self.a_start,
                "b_start": self.b_start}


def chain_numbers(structure: gemmi.Structure, chain_name: str) -> List[int]:
    """The chain's amino-acid residue numbers, checked against their positions.

    ProteinMPNN indexes BOTH jsonls by position in the parsed sequence, not by
    the number written in the file. Everything here is recorded as a residue
    number because RFD3 numbers a construct 1..N, so the two namespaces coincide
    -- but that is an assumption, and if RFD3 ever renumbers, the held and tied
    positions would quietly name the wrong residues. So it is checked.
    """
    numbers = [residue.seqid.num for residue in structure[0][chain_name]
               if is_amino_acid(residue)]
    off = [(position, number) for position, number in enumerate(numbers, start=1)
           if position != number]
    if off:
        position, number = off[0]
        raise PrepareError(
            f"chain {chain_name} is numbered {number} at position {position} "
            f"({len(off)} residue(s) disagree); ProteinMPNN indexes by position, "
            f"so the fixed and tied positions would name the wrong residues"
        )
    return numbers


def copy_layout(chain_name: str, total: int,
                linker_numbers: Sequence[int]) -> CopyLayout:
    """Split the fused chain into copy A, the linker, copy B.

    The linker is one contiguous run in the middle by construction: stage 05
    laid the two copies down end to end and stage 06 diffused the span between
    them. Anything else -- two runs, a run reaching a terminus, copies of
    unequal length -- is not the A-linker-B shape this stage designs, and
    mirroring the shell across it would pair up unrelated residues.
    """
    ordered = sorted(linker_numbers)
    start, end = ordered[0], ordered[-1]
    if ordered != list(range(start, end + 1)):
        present = set(ordered)
        inside = [number for number in range(start, end + 1) if number not in present]
        raise PrepareError(
            f"the diffused residues are not one run: {len(ordered)} of them "
            f"between {start} and {end}, with {len(inside)} carried-over "
            f"residue(s) in between (first at {inside[0]}). A construct with "
            f"more than one new segment has no single linker to design"
        )
    if start < 2 or end > total - 1:
        raise PrepareError(
            f"the diffused run {start}-{end} reaches a terminus of a "
            f"{total}-residue chain, so there is no copy on both sides of it"
        )
    before, after = start - 1, total - end
    if before != after:
        raise PrepareError(
            f"the copies flanking the linker are {before} and {after} residues; "
            f"they must be the same length for the shell to be mirrored between "
            f"them"
        )
    return CopyLayout(chain_name, before, 1, end + 1, start, end)


def mismatched_copies(structure: gemmi.Structure, chain_name: str,
                      layout: CopyLayout) -> List[Tuple[int, str, str]]:
    """(offset, name in copy A, name in copy B) wherever the copies differ.

    They should not differ at all: stage 05 laid the same designed protein down
    twice, and RFD3 carried both across as motif. A difference here means the
    two units were already not the same protein, and no redesign of the shell
    would make them so -- worse, tying positions whose natives differ lets
    ProteinMPNN write one copy's residue over the other's.
    """
    residues = [residue for residue in structure[0][chain_name]
                if is_amino_acid(residue)]
    found: List[Tuple[int, str, str]] = []
    for offset in range(1, layout.length + 1):
        first, second = layout.twins(offset)
        if residues[first - 1].name != residues[second - 1].name:
            found.append((offset, residues[first - 1].name, residues[second - 1].name))
    return found


def symmetric_offsets(layout: CopyLayout,
                      shell: Sequence[Tuple[str, int]]) -> List[int]:
    """The shell as offsets into a copy, unioned across both copies.

    An offset opened up in either copy is opened up in both, because the two
    have to come out identical and a position designed in one but held in the
    other cannot.
    """
    return sorted({layout.offset_of(number) for _, number in shell})


def mirrored_shell(layout: CopyLayout,
                   offsets: Sequence[int]) -> List[Tuple[str, int]]:
    """Every offset in the union, in both copies."""
    positions: List[Tuple[str, int]] = []
    for offset in offsets:
        first, second = layout.twins(offset)
        positions.append((layout.chain, first))
        positions.append((layout.chain, second))
    return sorted(positions, key=lambda item: item[1])


def tied_positions_for(layout: CopyLayout,
                       offsets: Sequence[int]) -> List[Dict[str, List[int]]]:
    """ProteinMPNN's tied_positions entry: one group per tie.

    Each group is {chain: [positions]}, and tied_featurize flattens the list
    under every chain key into one group -- so a single chain key holding two
    positions ties those two to each other. That is what is wanted here: the
    two copies are one chain, not two, since RFD3 fused them.
    """
    return [{layout.chain: list(layout.twins(offset))} for offset in offsets]


def mapped_keys(metadata: dict) -> Set[str]:
    """Generated residue keys that came from the input, per diffused_index_map.

    The map is {seed key: generated key}; the values are the ones that exist in
    the structure RFD3 produced. Anything else in that structure is new.
    """
    index_map = metadata.get("diffused_index_map") or {}
    if not isinstance(index_map, dict):
        raise PrepareError("diffused_index_map is not an object")
    return {str(value) for value in index_map.values()}


def linker_residues(structure: gemmi.Structure, protein_chains: Sequence[str],
                    kept: Set[str]) -> List[Tuple[str, int]]:
    """(chain, number) for every residue RFD3 generated rather than carried over."""
    found: List[Tuple[str, int]] = []
    for name in protein_chains:
        for residue in structure[0][name]:
            if not is_amino_acid(residue):
                continue
            if f"{name}{residue.seqid.num}" not in kept:
                found.append((name, residue.seqid.num))
    return found


def shell_around(structure: gemmi.Structure, protein_chains: Sequence[str],
                 linker: Sequence[Tuple[str, int]],
                 cutoff: float = SHELL_CUTOFF) -> List[Tuple[str, int]]:
    """Residues with a heavy atom within cutoff of any linker heavy atom."""
    linker_set = set(linker)
    points: List[np.ndarray] = []
    for name in protein_chains:
        for residue in structure[0][name]:
            if (name, residue.seqid.num) in linker_set:
                coords = heavy_coords(residue)
                if coords.size:
                    points.append(coords)
    if not points:
        return []
    linker_points = np.vstack(points)

    near: List[Tuple[str, int]] = []
    for name in protein_chains:
        for residue in structure[0][name]:
            key = (name, residue.seqid.num)
            if key in linker_set:
                continue
            if not is_amino_acid(residue):
                continue
            coords = heavy_coords(residue)
            if coords.size == 0:
                continue
            distances = np.linalg.norm(
                coords[:, None, :] - linker_points[None, :, :], axis=2
            )
            if distances.min() <= cutoff:
                near.append(key)
    return near


def fixed_positions_for(structure: gemmi.Structure, protein_chains: Sequence[str],
                        designable: Set[Tuple[str, int]]) -> Dict[str, List[int]]:
    """{chain: [residue numbers ProteinMPNN must not change]}.

    The complement of the designable set, because --fixed_positions_jsonl names
    what is held rather than what is free.
    """
    fixed: Dict[str, List[int]] = {}
    for name in protein_chains:
        held = [residue.seqid.num for residue in structure[0][name]
                if is_amino_acid(residue)
                and (name, residue.seqid.num) not in designable]
        fixed[name] = sorted(held)
    return fixed


def prepare_group(stage: Path, experiment: str, group_key: str,
                  cutoff: float = SHELL_CUTOFF) -> PrepareResult:
    archive_path = jp.input_archive_path(stage, experiment, group_key)
    if not archive_path.is_file():
        raise PrepareError(f"no input archive: {archive_path}")

    prepared_dir = jp.prepared_dir(stage, experiment, group_key)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    work_dir = jp.mpnn_work_dir(stage, experiment, group_key)
    work_dir.mkdir(parents=True, exist_ok=True)
    scratch = work_dir / "_scratch"

    already: Set[str] = set()
    designed = jp.mpnn_outputs_path(stage, experiment, group_key)
    if designed.is_file():
        with tarfile.open(designed, "r:gz") as archive:
            already = {name.split("/")[0] for name in archive.getnames() if "/" in name}

    result = PrepareResult(
        experiment, group_key, prepared_dir,
        jp.fixed_positions_path(stage, experiment, group_key),
        jp.tied_positions_path(stage, experiment, group_key),
        jp.linker_spec_path(stage, experiment, group_key),
    )
    fixed: Dict[str, Dict[str, List[int]]] = {}
    tied: Dict[str, List[Dict[str, List[int]]]] = {}
    records: List[dict] = []

    with tarfile.open(archive_path, "r:gz") as archive:
        names = archive.getnames()
        structures: Dict[str, str] = {}
        metadata: Dict[str, str] = {}
        for name in names:
            cleaned = name[2:] if name.startswith("./") else name
            stem = Path(cleaned).name
            for suffix in (".cif.gz", ".cif"):
                if stem.endswith(suffix):
                    structures[stem[: -len(suffix)]] = name
            if stem.endswith(".json"):
                metadata[stem[: -len(".json")]] = name

        for design_name, member in sorted(structures.items()):
            if design_name in already:
                result.already.append(design_name)
                continue
            try:
                if design_name not in metadata:
                    raise PrepareError(
                        f"no metadata json beside {design_name}; the linker cannot "
                        f"be identified without diffused_index_map"
                    )
                handle = archive.extractfile(metadata[design_name])
                if handle is None:
                    raise PrepareError("metadata json is unreadable")
                payload = json.loads(handle.read().decode("utf-8"))

                path = extract_member(archive, member, scratch)
                structure = gemmi.read_structure(str(path))
                structure.setup_entities()
                path.unlink(missing_ok=True)

                protein_chains, ligand_chains = classify_chains(structure)
                if ligand_chains:
                    drop_chains(structure, ligand_chains)
                if not protein_chains:
                    raise PrepareError("no protein chain in the generated structure")
                if len(protein_chains) != 1:
                    raise PrepareError(
                        f"{len(protein_chains)} protein chain(s) {protein_chains}; "
                        f"a fused construct is one -- if RFD3 left the copies as "
                        f"separate chains, the linker did not join them"
                    )
                chain_name = protein_chains[0]
                numbers = chain_numbers(structure, chain_name)

                kept = mapped_keys(payload)
                linker = linker_residues(structure, protein_chains, kept)
                if len(linker) < MIN_LINKER:
                    raise PrepareError(
                        f"every residue is accounted for by diffused_index_map, so "
                        f"nothing was generated -- this is not a fused construct"
                    )
                layout = copy_layout(chain_name, len(numbers),
                                     [number for _, number in linker])
                differing = mismatched_copies(structure, chain_name, layout)
                if differing:
                    offset, first, second = differing[0]
                    raise PrepareError(
                        f"the two copies already differ at {len(differing)} "
                        f"position(s) -- offset {offset} is {first} in one and "
                        f"{second} in the other -- so redesigning the shell "
                        f"cannot make them identical"
                    )

                found = shell_around(structure, protein_chains, linker, cutoff)
                offsets = symmetric_offsets(layout, found)
                shell = mirrored_shell(layout, offsets)
                designable = set(linker) | set(shell)

                fixed[design_name] = fixed_positions_for(
                    structure, protein_chains, designable
                )
                tied[design_name] = tied_positions_for(layout, offsets)
                out_path = prepared_dir / f"{design_name}.pdb"
                structure.write_pdb(str(out_path))

                records.append({
                    "id": design_name,
                    "experiment_name": experiment,
                    "group": group_key,
                    "pdb": out_path.name,
                    "chains": list(protein_chains),
                    "linker": [f"{chain}{number}" for chain, number in linker],
                    "linker_span": [layout.linker_start, layout.linker_end],
                    "copies": layout.as_record(),
                    "shell": [f"{chain}{number}" for chain, number in shell],
                    "shell_offsets": offsets,
                    "tied": len(tied[design_name]),
                    "held": sum(len(numbers) for numbers in fixed[design_name].values()),
                })
                result.prepared.append(design_name)
                result.linker_sizes.append(len(linker))
                result.shell_sizes.append(len(shell))
                result.tied_sizes.append(len(offsets))
                _log(f"[prepare] {design_name}: {len(linker)} linker residue(s), "
                     f"{len(found)} within {cutoff:g} A -> {len(offsets)} offset(s) "
                     f"mirrored into both copies ({len(shell)} position(s), "
                     f"{len(tied[design_name])} tied), {records[-1]['held']} held")
            except (PrepareError, RuntimeError, ValueError, KeyError, OSError,
                    json.JSONDecodeError) as exc:
                result.failed[design_name] = str(exc).splitlines()[0]
                _log(f"[prepare] {design_name}: FAILED ({exc})")

    if not fixed and result.already:
        _log(f"[prepare] {experiment}/{group_key}: all {len(result.already)} "
             f"construct(s) already designed, nothing to prepare")
        return result
    if not fixed:
        raise PrepareError(
            f"{experiment}/{group_key}: nothing could be prepared "
            f"({len(result.failed)} construct(s) failed) -- {why_failed(result.failed)}"
        )

    result.fixed_positions.write_text(json.dumps(fixed) + "\n", encoding="utf-8")
    result.tied_positions.write_text(json.dumps(tied) + "\n", encoding="utf-8")
    with result.spec_path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    if scratch.exists() and not any(scratch.iterdir()):
        scratch.rmdir()
    return result


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("experiment")
    parser.add_argument("group_key")
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--shell-cutoff", type=float, default=SHELL_CUTOFF,
                        help=f"heavy-atom distance from the linker that opens a "
                             f"residue up for design (default: {SHELL_CUTOFF})")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = prepare_group(args.stage.resolve(), args.experiment,
                               args.group_key, args.shell_cutoff)
    except PrepareError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    _log(f"[prepare] {result.experiment}/{result.group_key}: "
         f"{len(result.prepared)} construct(s) -> {result.prepared_dir}")
    if result.linker_sizes:
        _log(f"[prepare] linker {min(result.linker_sizes)}-{max(result.linker_sizes)} "
             f"residue(s), shell {min(result.shell_sizes)}-{max(result.shell_sizes)} "
             f"position(s) over {min(result.tied_sizes)}-{max(result.tied_sizes)} "
             f"tied offset(s)")
    _log(f"[prepare] fixed positions -> {result.fixed_positions}")
    _log(f"[prepare] tied positions  -> {result.tied_positions}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
