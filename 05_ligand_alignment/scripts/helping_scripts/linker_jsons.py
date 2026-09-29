#!/usr/bin/env python3
"""
Stage 05, step 2: write the RFD3 jsons that fuse each aligned pair, two per pair.

    inputs_prepared/<experiment>/<group>/<sid>.pdb  <- align_pairs wrote it:
                                                       two aligned copies, no cellulose
    sorted_clean/<experiment>/passed/<group>.tar.gz <- that pdb, flat, and nothing else
    tables/stage_05_pairs_<experiment>.csv          <- how far each linker must reach

    ../06_rfd3_linker_generation/json/<experiment>/<sid>_AB.json   <- A, linker, B
    ../06_rfd3_linker_generation/json/<experiment>/<sid>_BA.json   <- B, linker, A

Two kinds of output and no third: jsons as loose files in stage 06, one pdb per
sequence in a tar. Nothing strips anything here any more -- align_pairs.py drops
both fibres before it writes, so what this reads is already what RFD3 wants.

The jsons are written into stage 06 rather than here, because they are the one
thing in this pipeline a person edits by hand. A copy written here and then
transferred would mean two files with the same name, one of them carrying the
length you typed and the other not, and only one of them being the file RFD3
actually reads. For the same reason they are not in the handover archive: an
archived json is a pristine LINKER copy sitting beside the edited one, which is
the same confusion by another route.

A json that already exists is never rewritten. Re-running this stage after
filling some lengths in is therefore safe, and is the normal way to pick up
pairs that arrived since.

The generation is not symmetric. The two copies are identical in sequence, but
the linker runs from one copy's C-terminus to the other's N-terminus and no
symmetry maps that onto itself, so no `symmetry` block is written and what comes
back is one continuous chain.

Everything that already exists is held: select_fixed_atoms names EVERY residue
of both copies with "ALL". They have been designed, folded and validated by this
point, and only the linker is still an open question.

Two jsons per pair, because A-linker-B and B-linker-A are different questions.
The molecule is the same two bodies in the same places; what differs is which
C-terminus has to reach which N-terminus, and those are not the same distance
apart. Both are written and both are worth running unless you have ruled one out.

The linker length is yours. Every json carries the literal LINKER where its
length belongs, so an unedited file fails loudly instead of quietly diffusing
some default. Stage 06's set_linker.py fills it in and recomputes `length` to
match; run_rfd3.py then generates whichever have a length and skips the rest.

The table records the gap each orientation has to span and the fewest residues
that could do it, at 3.4 A per residue fully extended. Treat that as a floor and
nothing else: a linker at its own contour length is a taut string with no
freedom, and RFD3 will struggle to place one.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import gemmi
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import job_paths as jp  # noqa: E402
from archives import load_table, merge_members_into_archive  # noqa: E402
from structures import classify_chains  # noqa: E402

STAGE = Path(__file__).resolve().parents[2]

COPY_CHAINS = ("A", "B")
LINKER_TOKEN = "LINKER"
# A peptide bond contributes about 3.4 A along the chain when fully extended.
RISE_PER_RESIDUE = 3.4

PAIR_FIELDS = [
    "sequence_id", "protein_id", "experiment_name", "group", "orientation",
    "from_chain", "from_residue", "to_chain", "to_residue",
    "gap_angstrom", "min_residues", "linker_length",
]


class LinkerError(Exception):
    """One pair could not be turned into a pair of jsons."""


def _log(*parts: object) -> None:
    print(*parts, flush=True)


@dataclass
class LinkerReport:
    pairs: int = 0
    jsons: int = 0
    already: int = 0
    already_filled: int = 0   # ... and had a length typed into them
    rows: List[dict] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        # The filled count is said on both paths. This stage writes into stage
        # 06's json directory, where the lengths are typed by hand, so "did that
        # just overwrite my work" is the question the summary has to answer --
        # including on the run where nothing else happened.
        held = (f", {self.already_filled} of them with a length set and left "
                f"alone") if self.already_filled else ""
        if not self.jsons:
            return f"linker: nothing new ({self.already} already written{held})"
        return (f"linker: {self.pairs} pair(s), {self.jsons} json(s) written, "
                f"{self.already} already there{held}")


# ---------------------------------------------------------------------------
# Reading the pair
# ---------------------------------------------------------------------------

def polymer_residues(chain: gemmi.Chain) -> List[gemmi.Residue]:
    out: List[gemmi.Residue] = []
    for residue in chain:
        info = gemmi.find_tabulated_residue(residue.name)
        if info is not None and info.is_amino_acid():
            out.append(residue)
    return out


def gaps_in(numbers: Sequence[int]) -> List[Tuple[int, int]]:
    """[(before, after), ...] for each break in the residue numbering."""
    ordered = sorted(numbers)
    return [(first, second) for first, second in zip(ordered, ordered[1:])
            if second != first + 1]


def chain_contig(chain_name: str, residues: Sequence[gemmi.Residue]) -> str:
    """'A1-125' -- one segment covering the whole chain.

    A backbone RFD3 generated and AlphaFold folded runs 1..n with nothing
    missing, so the whole copy is one motif segment. A gapped chain is refused
    rather than written as several segments: 'A1-125' for a chain that skips
    residues would tell RFD3 to keep ones that are not in the file, and the
    alternative -- quietly emitting 'A1-40,A44-125' -- hides a structure that
    should not have reached this stage.
    """
    numbers = [residue.seqid.num for residue in residues]
    breaks = gaps_in(numbers)
    if breaks:
        shown = ", ".join(f"{first}->{second}" for first, second in breaks[:4])
        raise LinkerError(
            f"chain {chain_name} is not contiguous ({len(breaks)} gap(s): {shown}); "
            f"a fused contig cannot be written for a chain with residues missing"
        )
    return f"{chain_name}{min(numbers)}-{max(numbers)}"


def terminus(residues: Sequence[gemmi.Residue], which: str) -> Tuple[int, np.ndarray]:
    """(residue number, CA coordinates) of the N- or C-terminal residue."""
    residue = residues[0] if which == "N" else residues[-1]
    for atom in residue:
        if atom.name == "CA":
            return residue.seqid.num, np.array([atom.pos.x, atom.pos.y, atom.pos.z])
    raise LinkerError(f"the {which}-terminal residue {residue.seqid.num} has no CA")


def span(first: np.ndarray, second: np.ndarray) -> Tuple[float, int]:
    """(distance, fewest residues that could bridge it fully extended)."""
    distance = float(np.linalg.norm(first - second))
    return distance, int(np.ceil(distance / RISE_PER_RESIDUE))


# ---------------------------------------------------------------------------
# Writing the jsons
# ---------------------------------------------------------------------------

def fixed_atoms(residues: Dict[str, Sequence[gemmi.Residue]]) -> Dict[str, str]:
    """{'A1-125': 'ALL', 'B1-125': 'ALL'} -- every residue of both copies.

    EVERYTHING is held, not just the scaffolded residues. By this point both
    copies are finished: designed with the ligand in context, folded, validated
    against their backbone and checked for clashes. Only the linker is still in
    question, so nothing that already exists should be free to move -- and
    "ALL" holds the side chains too, not merely the backbone.

    One range key per chain rather than 250 single keys. The range form is exact
    here only because chain_contig() has already refused any chain with a gap in
    it; 'A1-125' over a chain missing residue 40 would claim one that is not
    there. common/designs.py expands these the same way, so both spellings mean
    the same thing to everything downstream.
    """
    out: Dict[str, str] = {}
    for chain in sorted(residues):
        numbers = [residue.seqid.num for residue in residues[chain]]
        out[f"{chain}{min(numbers)}-{max(numbers)}"] = "ALL"
    return out


def build_json(name: str, input_path: str, contig: str,
               select_fixed: Dict[str, str]) -> dict:
    """One RFD3 design config, shaped like the stage 01 jsons.

    No `symmetry` block. Stage 01 generated a symmetric dimer and the two copies
    here are still identical in sequence, but the linker is not: it leaves one
    copy's C-terminus and arrives at the other's N-terminus, which no symmetry
    operation maps onto itself. Asking for symmetry would fight the one thing
    being generated. The result is a single continuous chain.
    """
    entry: Dict[str, object] = {
        "input": input_path,
        "contig": contig,
        "length": LINKER_TOKEN,
    }
    if select_fixed:
        entry["select_fixed_atoms"] = select_fixed
    return {name: entry}


def write_pair(stage: Path, experiment: str, group_key: str, sequence_id: str,
               pair: gemmi.Structure, protein_id: str,
               report: LinkerReport) -> None:
    protein_chains, ligand_chains = classify_chains(pair)
    if len(protein_chains) != 2:
        raise LinkerError(
            f"expected two protein chains in the pair, found {protein_chains}"
        )
    if ligand_chains:
        # align_pairs.py drops the fibre before writing, so one here means this
        # pdb was written by an older run. Refused rather than stripped: the
        # rest of that file was produced by code that has since changed, and
        # quietly fixing the symptom would leave the cause in place.
        raise LinkerError(
            f"the pair still carries {ligand_chains}; it predates the version "
            f"that drops the fibre. Delete it and re-run the alignment."
        )

    residues = {name: polymer_residues(pair[0][name]) for name in protein_chains}
    for name, chain_residues in residues.items():
        if not chain_residues:
            raise LinkerError(f"chain {name} has no amino acid residues")

    # RELATIVE TO THE CHECKOUT ROOT, pointing into stage 06 where RFD3 reads it:
    #
    #     06_rfd3_linker_generation/inputs_prepared/<experiment>/<group>/<sid>.pdb
    #
    # It was absolute for a real reason: run_one_job rewrote "input" only for a
    # design declaring a symmetry block, and a linker json deliberately declares
    # none, so the path reached RFD3 exactly as written and had to be one RFD3
    # could open. The cost was that the json only worked on the machine that
    # wrote it -- /home/<user>/1cbh_clear on the workstation is not a path the
    # cluster has, and these jsons carry hand-set linker lengths, so the only
    # remedy was to throw that work away and regenerate them.
    #
    # run_one_job now resolves "input" for EVERY design and writes the absolute
    # path back before RFD3 sees it, so the json can hold the portable spelling
    # and still reach RFD3 with a path it can open. Jsons already written with an
    # absolute path keep working: resolve_seed_path re-anchors them.
    target = (jp.pair_dir(jp.stage06_root(stage), experiment, group_key)
              / f"{sequence_id}.pdb")
    relative = str(target.resolve().relative_to(stage.parent.resolve()))

    json_dir = jp.linker_json_dir(jp.stage06_root(stage), experiment)
    json_dir.mkdir(parents=True, exist_ok=True)
    select_fixed = fixed_atoms(residues)

    first_name, second_name = protein_chains
    for orientation, (start_chain, end_chain) in (
        ("AB", (first_name, second_name)), ("BA", (second_name, first_name))
    ):
        name = f"{sequence_id}_{orientation}"
        destination = json_dir / f"{name}.json"
        if destination.is_file():
            report.already += 1
            try:
                if LINKER_TOKEN not in destination.read_text(encoding="utf-8"):
                    report.already_filled += 1
            except OSError:
                pass
            continue

        contig = (f"{chain_contig(start_chain, residues[start_chain])},"
                  f"{LINKER_TOKEN},"
                  f"{chain_contig(end_chain, residues[end_chain])}")
        destination.write_text(
            json.dumps(build_json(name, relative, contig, select_fixed), indent=2) + "\n",
            encoding="utf-8",
        )
        report.jsons += 1

        from_number, from_point = terminus(residues[start_chain], "C")
        to_number, to_point = terminus(residues[end_chain], "N")
        distance, minimum = span(from_point, to_point)
        report.rows.append({
            "sequence_id": sequence_id,
            "protein_id": protein_id,
            "experiment_name": experiment,
            "group": group_key,
            "orientation": orientation,
            "from_chain": start_chain,
            "from_residue": from_number,
            "to_chain": end_chain,
            "to_residue": to_number,
            "gap_angstrom": f"{distance:.2f}",
            "min_residues": minimum,
            "linker_length": "",
        })
        _log(f"[linker] {name}: {start_chain}{from_number} C-term to "
             f"{end_chain}{to_number} N-term is {distance:.1f} A, at least "
             f"{minimum} residue(s) -> {destination.name}")
    report.pairs += 1


def handover_members(sequence_id: str, payload: bytes
                     ) -> List[Tuple[tarfile.TarInfo, bytes]]:
    """The one structure stage 06 needs, as a flat archive member.

    Flat -- <sid>.pdb, not <sid>/<sid>.pdb. There is one file per sequence, so a
    directory per sequence would be a folder holding a single thing.
    """
    info = tarfile.TarInfo(f"{sequence_id}.pdb")
    info.size = len(payload)
    return [(info, payload)]


def pair_sources(stage: Path, experiment: str, group_key: str) -> Dict[str, bytes]:
    """{sequence_id: pdb bytes} for this group, loose files first, then archive.

    Both, because the loose pdb is transient. align_pairs.py writes it, this
    archives it, and cleanup deletes it -- so on the first run the loose file is
    the only copy and on every run after it the archive is. Reading only the
    loose files would mean a json deleted to be regenerated never came back:
    there would be no pdb to read and align_pairs would skip the sequence as
    already archived. Reading only the archive would miss the first run.
    """
    sources: Dict[str, bytes] = {}
    prepared = jp.pair_dir(stage, experiment, group_key)
    if prepared.is_dir():
        for path in sorted(prepared.glob("*.pdb")):
            sources[path.stem] = path.read_bytes()

    archive_path = jp.sorted_archive_path(stage, experiment, "passed", group_key)
    if archive_path.is_file():
        with tarfile.open(archive_path, "r:gz") as archive:
            for name in sorted(archive.getnames()):
                if not name.endswith(".pdb"):
                    continue
                sequence_id = Path(name).stem
                if sequence_id in sources:
                    continue          # the loose copy is the fresher one
                handle = archive.extractfile(name)
                if handle is not None:
                    sources[sequence_id] = handle.read()
    return sources


def build_group(stage: Path, experiment: str, group_key: str,
                report: LinkerReport) -> None:
    """Every pair this group has, loose or archived."""
    sources = pair_sources(stage, experiment, group_key)
    if not sources:
        return

    # protein_id comes from the table align_pairs.py wrote rather than from a
    # spec file copied in beside each pair. One fewer file on disk, and the two
    # tables cannot disagree about which protein a sequence belongs to.
    proteins = load_table(jp.alignment_table_path(stage, experiment),
                          key="sequence_id")

    members: List[Tuple[tarfile.TarInfo, bytes]] = []
    for sequence_id, payload in sorted(sources.items()):
        try:
            pair = gemmi.read_pdb_string(payload.decode("utf-8"))
            pair.setup_entities()
            protein_id = proteins.get(sequence_id, {}).get("protein_id", "")
            write_pair(stage, experiment, group_key, sequence_id, pair,
                       protein_id, report)
            members.extend(handover_members(sequence_id, payload))
        except (LinkerError, RuntimeError, ValueError, KeyError, OSError,
                UnicodeDecodeError, json.JSONDecodeError) as exc:
            report.problems.append(f"{experiment}/{sequence_id}: {exc}")
            _log(f"[linker] {sequence_id}: FAILED ({exc})")

    if not members:
        return
    destination = jp.sorted_archive_path(stage, experiment, "passed", group_key)
    existing: set = set()
    if destination.is_file():
        with tarfile.open(destination, "r:gz") as archived:
            existing = set(archived.getnames())
    fresh = [(info, payload) for info, payload in members
             if info.name not in existing]
    if fresh:
        added, total = merge_members_into_archive(destination, fresh)
        _log(f"[linker] passed/{group_key}: +{added} file(s), {total} total "
             f"-> {destination}")


def write_table(stage: Path, experiment: str, rows: Sequence[dict]) -> Path:
    path = jp.linker_span_table_path(stage, experiment)
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_FIELDS)
        if fresh:
            writer.writeheader()
        for row in rows:
            writer.writerow({field_: row.get(field_, "") for field_ in PAIR_FIELDS})
    return path


def run_build(stage: Path, experiment: Optional[str] = None) -> LinkerReport:
    report = LinkerReport()
    names = [experiment] if experiment else jp.inputs_experiments(stage)
    if not names:
        _log(f"[linker] nothing handed over under {jp.inputs_root(stage)}")
        return report

    for experiment_name in names:
        directory = jp.inputs_root(stage) / experiment_name
        if not directory.is_dir():
            continue
        before = len(report.rows)
        for archive_path in sorted(directory.glob("*.tar.gz")):
            group_key = archive_path.name[: -len(".tar.gz")]
            try:
                build_group(stage, experiment_name, group_key, report)
            except LinkerError as exc:
                report.problems.append(f"{experiment_name}/{group_key}: {exc}")
                _log(f"[linker] {experiment_name}/{group_key}: FAILED ({exc})")
        if len(report.rows) > before:
            path = write_table(stage, experiment_name, report.rows[before:])
            _log(f"[linker] spans recorded -> {path}")
    return report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--experiment", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    report = run_build(args.stage.resolve(), args.experiment)
    _log(f"[done] {report.summary()}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
