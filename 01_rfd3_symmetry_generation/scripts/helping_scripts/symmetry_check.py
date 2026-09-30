#!/usr/bin/env python3
"""
Structural checks on a stage 01 seed motif.
"""
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import gemmi
import numpy as np

# Maximum per-atom deviation, in Angstrom, allowed after superposing the
# two chains. Applied per atom rather than as an RMSD so a single badly
# placed atom cannot be averaged away by a few thousand good ones.
DEFAULT_TOLERANCE = 0.05

# A rotation smaller than this is reported as "no rotation", i.e. the
# motif is linear. Well below any real symmetry operation.
ANGLE_EPS_DEG = 0.5

# A screw rise smaller than this is reported as "no rise".
RISE_EPS_A = 0.05

StructureLike = Union[str, Path, gemmi.Structure]


@dataclass(frozen=True)
class AtomRecord:
    """One comparable atom. `label` is for error messages only -- it carries
    the residue number, which deliberately plays no part in matching."""

    resname: str
    label: str
    atom: str
    x: float
    y: float
    z: float


@dataclass
class MotifTransform:
    """The proper rigid motion mapping the first chain onto the second."""

    rotation_deg: float
    screw_rise_a: float
    centroid_shift_a: float
    axis: Tuple[float, float, float]
    # Chain 2 centroid minus chain 1 centroid. For a linear motif this is the
    # translation operator itself, directly comparable with the vector in
    # RFD3_T2EXACT_TRANSLATIONS.
    displacement: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def is_linear(self) -> bool:
        """True when the two chains are related by translation alone."""
        return self.rotation_deg < ANGLE_EPS_DEG

    def describe(self) -> str:
        if self.is_linear:
            vector = ", ".join(f"{v:.4f}" for v in self.displacement)
            return (f"pure translation of {self.centroid_shift_a:.3f} A (linear); "
                    f"t = ({vector})")
        if abs(self.screw_rise_a) < RISE_EPS_A:
            return f"pure rotation of {self.rotation_deg:.2f} deg"
        return (f"screw motion: {self.rotation_deg:.2f} deg about "
                f"({self.axis[0]:.3f}, {self.axis[1]:.3f}, {self.axis[2]:.3f}) "
                f"with a {self.screw_rise_a:.3f} A rise")


@dataclass
class SymmetryResult:
    is_symmetric: bool
    reason: str = ""  # empty when is_symmetric is True
    chains: Tuple[str, ...] = ()
    n_atoms: int = 0
    rmsd_a: Optional[float] = None
    max_deviation_a: Optional[float] = None
    transform: Optional[MotifTransform] = None
    altlocs_collapsed: int = 0
    # False when the file holds a single ASU, so there is no second copy to
    # compare against. That is the normal shape of a stage 01 input: the
    # symmetry frames come from RFD3_T<N>EXACT_TRANSLATIONS, not from the file.
    comparable: bool = True

    def summary(self) -> str:
        """One line describing what was compared and what came out."""
        if not self.comparable:
            if len(self.chains) == 1:
                return (f"single ASU (protein chain {self.chains[0]}) - "
                        f"no second copy to compare")
            return (f"{len(self.chains)} protein chains ({', '.join(self.chains)}) - "
                    f"use --chains to pick a pair to compare")
        if not self.is_symmetric:
            return f"NOT SYMMETRIC: {self.reason}"
        detail = self.transform.describe() if self.transform else "no transform recovered"
        return (f"chains {self.chains[0]}/{self.chains[1]}, {self.n_atoms} atoms matched; "
                f"{detail}; max deviation {self.max_deviation_a:.4f} A, "
                f"rmsd {self.rmsd_a:.4f} A")


class StructureError(ValueError):
    """The file cannot be compared at all (wrong chain count, unreadable)."""


# Step 1. Loading.

def load_structure(source: StructureLike) -> gemmi.Structure:
    """Parse a structure, or pass an already-parsed one straight through."""
    if isinstance(source, gemmi.Structure):
        return source
    structure = gemmi.read_structure(str(source))
    structure.setup_entities()
    return structure


def _is_polymer_residue(residue: gemmi.Residue) -> bool:
    """Polymer residues only. HETATM ligands and waters are excluded here
    rather than at chain level, so a ligand modelled inside a protein
    chain is skipped instead of breaking the atom-composition match."""
    return residue.het_flag != "H" and not residue.is_water()


def protein_chains(
    structure: gemmi.Structure,
    chain_names: Optional[Sequence[str]] = None,
) -> List[gemmi.Chain]:
    """The chains to compare: the two containing polymer residues, or the
    ones named explicitly if the seed holds more than two."""
    model = structure[0]
    if chain_names:
        by_name = {chain.name: chain for chain in model}
        missing = [name for name in chain_names if name not in by_name]
        if missing:
            raise StructureError(
                f"requested chain(s) {missing} not in file; "
                f"available: {[c.name for c in model]}"
            )
        return [by_name[name] for name in chain_names]

    candidates = [c for c in model if any(_is_polymer_residue(r) for r in c)]
    if not candidates:
        raise StructureError(
            f"no protein chains found; all chains in file: {[c.name for c in model]}"
        )
    return candidates


def chain_atoms(chain: gemmi.Chain) -> Tuple[List[AtomRecord], int]:
    """Ordered comparable atoms for one chain, plus how many alternate
    conformers were collapsed."""
    records: List[AtomRecord] = []
    collapsed = 0
    for residue in chain:
        if not _is_polymer_residue(residue):
            continue
        best: Dict[str, gemmi.Atom] = {}
        for atom in residue:
            previous = best.get(atom.name)
            if previous is None:
                best[atom.name] = atom
                continue
            collapsed += 1
            if atom.occ > previous.occ:
                best[atom.name] = atom
        icode = residue.seqid.icode.strip()
        label = f"{residue.name}{residue.seqid.num}{icode}"
        for name in sorted(best):
            position = best[name].pos
            records.append(
                AtomRecord(residue.name, label, name, position.x, position.y, position.z)
            )
    return records, collapsed


# Step 2. Matching.

def _composition_mismatch(
    first: List[AtomRecord],
    second: List[AtomRecord],
    name_a: str,
    name_b: str,
) -> Optional[str]:
    """Positional comparison of residue names and atom names. Residue
    *numbers* are never compared, so differing numbering is fine."""
    if len(first) != len(second):
        return (f"atom count mismatch: chain {name_a} has {len(first)} comparable "
                f"atom(s), chain {name_b} has {len(second)}")
    for index, (rec_a, rec_b) in enumerate(zip(first, second)):
        if rec_a.resname != rec_b.resname or rec_a.atom != rec_b.atom:
            return (f"composition diverges at matched position {index}: "
                    f"chain {name_a} has {rec_a.label}:{rec_a.atom}, "
                    f"chain {name_b} has {rec_b.label}:{rec_b.atom}")
    return None


def _coordinates(records: List[AtomRecord]) -> np.ndarray:
    return np.array([[r.x, r.y, r.z] for r in records], dtype=float)


def _best_fit(
    first: np.ndarray, second: np.ndarray, allow_reflection: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Kabsch fit: the rigid motion mapping 'first' onto 'second'.

    Coordinates are centred before the SVD, which keeps the fit well
    conditioned for motifs far from the origin or with an extreme aspect
    ratio."""
    centroid_a, centroid_b = first.mean(axis=0), second.mean(axis=0)
    covariance = (first - centroid_a).T @ (second - centroid_b)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if not allow_reflection and np.linalg.det(rotation) < 0:
        rotation = vt.T @ np.diag([1.0, 1.0, -1.0]) @ u.T
    translation = centroid_b - rotation @ centroid_a
    return rotation, translation


def _deviations(
    first: np.ndarray, second: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    return np.linalg.norm(first @ rotation.T + translation - second, axis=1)


def _describe_transform(
    matrix: np.ndarray, vector: np.ndarray, first: np.ndarray, second: np.ndarray
) -> MotifTransform:
    """Recover rotation angle, axis and screw rise from the superposition.

    The rise is the component of the translation along the rotation axis,
    which -- unlike the translation itself -- does not depend on where the
    origin sits.
    """
    cosine = (float(np.trace(matrix)) - 1.0) / 2.0
    angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    offset = second.mean(axis=0) - first.mean(axis=0)
    centroid_shift = float(np.linalg.norm(offset))
    displacement = tuple(float(v) for v in offset)

    if angle < ANGLE_EPS_DEG:
        # No rotation: the axis is undefined, so report the translation
        # direction instead and treat its full length as the rise.
        norm = float(np.linalg.norm(vector))
        axis = vector / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
        return MotifTransform(
            angle, norm, centroid_shift, tuple(float(v) for v in axis), displacement
        )

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    axis = np.real(eigenvectors[:, int(np.argmin(np.abs(eigenvalues - 1.0)))])
    norm = float(np.linalg.norm(axis))
    axis = axis / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
    rise = float(np.dot(vector, axis))
    if rise < 0:  # axis sign is arbitrary; report a positive rise
        axis, rise = -axis, -rise
    return MotifTransform(
        angle, rise, centroid_shift, tuple(float(v) for v in axis), displacement
    )


def check_symmetry(
    source: StructureLike,
    tolerance: float = DEFAULT_TOLERANCE,
    chain_names: Optional[Sequence[str]] = None,
) -> SymmetryResult:
    """Verify the seed's two protein chains are related by a proper rigid
    motion, and report which motion it is."""
    structure = load_structure(source)
    chains = protein_chains(structure, chain_names)

    if len(chains) != 2:
        # A single ASU is the normal stage 01 input; more than two copies needs
        # --chains to say which pair to compare. Neither is an error here.
        return SymmetryResult(
            is_symmetric=True, comparable=False,
            chains=tuple(c.name for c in chains),
            reason="" if len(chains) == 1 else
                   f"{len(chains)} protein chains; use --chains to pick a pair",
        )

    chain_a, chain_b = chains
    records_a, collapsed_a = chain_atoms(chain_a)
    records_b, collapsed_b = chain_atoms(chain_b)
    collapsed = collapsed_a + collapsed_b
    names = (chain_a.name, chain_b.name)

    if not records_a:
        return SymmetryResult(
            False, f"chain {chain_a.name} has no polymer atoms to compare",
            chains=names, altlocs_collapsed=collapsed,
        )

    mismatch = _composition_mismatch(records_a, records_b, chain_a.name, chain_b.name)
    if mismatch:
        return SymmetryResult(False, mismatch, chains=names, altlocs_collapsed=collapsed)

    coords_a, coords_b = _coordinates(records_a), _coordinates(records_b)
    rotation, translation = _best_fit(coords_a, coords_b, allow_reflection=False)
    deviations = _deviations(coords_a, coords_b, rotation, translation)
    max_deviation = float(deviations.max())
    rmsd = float(np.sqrt((deviations ** 2).mean()))
    transform = _describe_transform(rotation, translation, coords_a, coords_b)

    if max_deviation > tolerance:
        worst = int(np.argmax(deviations))
        # Would the chains fit if a reflection were allowed? If so they are
        # mirror images of each other, which is worth saying outright.
        free_rotation, free_translation = _best_fit(coords_a, coords_b, allow_reflection=True)
        mirrored = (
            np.linalg.det(free_rotation) < 0
            and float(_deviations(coords_a, coords_b, free_rotation, free_translation).max())
            <= tolerance
        )
        diagnosis = (
            "the chains are mirror images of each other, not copies related by a "
            "rotation or translation"
            if mirrored
            else "the second chain is not a rigid copy of the first"
        )
        return SymmetryResult(
            False,
            f"chains are not superimposable: {diagnosis}. Worst atom "
            f"{records_a[worst].label}:{records_a[worst].atom} is off by "
            f"{max_deviation:.4f} A (tol={tolerance} A, rmsd={rmsd:.4f} A)",
            chains=names, n_atoms=len(records_a),
            rmsd_a=rmsd, max_deviation_a=max_deviation,
            transform=transform, altlocs_collapsed=collapsed,
        )

    return SymmetryResult(
        True, "", chains=names, n_atoms=len(records_a),
        rmsd_a=rmsd, max_deviation_a=max_deviation,
        transform=transform, altlocs_collapsed=collapsed,
    )


def polymer_residues(source: StructureLike) -> Dict[str, "set"]:
    """{chain name: set of residue numbers} for polymer residues only."""
    structure = load_structure(source)
    out: Dict[str, set] = {}
    for chain in structure[0]:
        nums = {r.seqid.num for r in chain if _is_polymer_residue(r)}
        if nums:
            out[chain.name] = nums
    return out


def referenced_residues(config: dict) -> "set":
    """(chain, residue number) pairs a design config points at.

    Covers contig motif segments ('A24-36', 'A5', 'B28-32'), select_fixed_atoms
    keys and select_exposed entries. Bare numbers in a contig are designed-region
    lengths rather than residues, so they are skipped.
    """
    wanted = set()

    for segment in str(config.get("contig", "")).split(","):
        match = re.fullmatch(r"\s*([A-Za-z])(\d+)(?:-(\d+))?\s*", segment)
        if match:
            chain, first = match.group(1), int(match.group(2))
            last = int(match.group(3)) if match.group(3) else first
            wanted |= {(chain, n) for n in range(first, last + 1)}

    for key in config.get("select_fixed_atoms", {}) or {}:
        match = re.fullmatch(r"([A-Za-z])(\d+)", str(key))
        if match:
            wanted.add((match.group(1), int(match.group(2))))

    for key in str(config.get("select_exposed", "")).split(","):
        match = re.fullmatch(r"\s*([A-Za-z])(\d+)\s*", key)
        if match:
            wanted.add((match.group(1), int(match.group(2))))

    return wanted


def missing_residues(source: StructureLike, config: dict) -> List[Tuple[str, int]]:
    """Residues the config references that the structure does not contain.

    This is the check that matters before dispatch: a contig naming residues
    absent from the seed fails inside RFD3 after the model is loaded, which is
    an expensive way to learn about a typo or a swapped seed file.
    """
    present = polymer_residues(source)
    return sorted(
        item for item in referenced_residues(config)
        if item[1] not in present.get(item[0], ())
    )


def copy_transforms(
    source: StructureLike,
    asu_chain: str,
    tolerance: float = DEFAULT_TOLERANCE,
) -> List[Tuple[np.ndarray, Tuple[float, float, float]]]:
    """(rotation, translation) mapping `asu_chain` onto every protein copy.

        copy = R @ asu + t

    which is the convention the overlay's frames already use, so what is
    measured here can be handed over unchanged.

    The ASU itself comes first as (identity, zero), and the rest are ordered by
    distance along the motion, so the result maps directly onto
    RFD3_T<N>EXACT_TRANSLATIONS.

    A ROTATION IS NOT REFUSED. copy_translations() below is this function with
    the rotation required to be identity, which was the only thing the frames
    could carry until they learned to carry a full rigid motion. A chitin seed
    whose two copies are related by a 45 degree screw has a perfectly good
    operator; it simply is not a translation.

    What is still refused is a copy that is not RIGID -- a different
    conformation, not a different placement -- because no single frame can
    express that however it is written.
    """
    structure = load_structure(source)
    chains = {c.name: c for c in protein_chains(structure)}
    if asu_chain not in chains:
        raise StructureError(
            f"ASU chain {asu_chain!r} not among the protein chains {sorted(chains)}"
        )

    reference, _ = chain_atoms(chains[asu_chain])
    reference_coords = _coordinates(reference)

    found: List[Tuple[float, np.ndarray, Tuple[float, float, float]]] = []
    for name, chain in chains.items():
        if name == asu_chain:
            continue
        records, _ = chain_atoms(chain)
        mismatch = _composition_mismatch(reference, records, asu_chain, name)
        if mismatch:
            raise StructureError(f"copy {name} does not match the ASU: {mismatch}")

        coords = _coordinates(records)
        rotation, translation = _best_fit(reference_coords, coords, allow_reflection=False)
        deviation = float(_deviations(reference_coords, coords, rotation, translation).max())
        if deviation > tolerance:
            raise StructureError(
                f"chain {name} is not a rigid copy of {asu_chain} "
                f"(worst atom off by {deviation:.4f} A, tol={tolerance} A)"
            )
        transform = _describe_transform(rotation, translation, reference_coords, coords)
        found.append((transform.centroid_shift_a, rotation,
                      tuple(float(v) for v in translation)))

    found.sort(key=lambda item: item[0])
    return ([(np.eye(3), (0.0, 0.0, 0.0))]
            + [(rotation, translation) for _, rotation, translation in found])


def copy_translations(
    source: StructureLike,
    asu_chain: str,
    tolerance: float = DEFAULT_TOLERANCE,
) -> List[Tuple[float, float, float]]:
    """Displacements from `asu_chain` to every protein copy in the file.

    Returns one vector per copy, starting with (0,0,0) for the ASU itself and
    ordered by distance along the translation, so the result maps directly onto
    RFD3_T<N>EXACT_TRANSLATIONS.

    Every copy must be related to the ASU by a *pure translation*: the overlay's
    exact-translation frames carry no rotation, so a rotated copy cannot be
    expressed and is refused rather than silently flattened.
    """
    structure = load_structure(source)
    chains = {c.name: c for c in protein_chains(structure)}
    if asu_chain not in chains:
        raise StructureError(
            f"ASU chain {asu_chain!r} not among the protein chains {sorted(chains)}"
        )

    reference, _ = chain_atoms(chains[asu_chain])
    reference_coords = _coordinates(reference)

    offsets: List[Tuple[float, Tuple[float, float, float]]] = []
    for name, chain in chains.items():
        if name == asu_chain:
            continue
        records, _ = chain_atoms(chain)
        mismatch = _composition_mismatch(reference, records, asu_chain, name)
        if mismatch:
            raise StructureError(f"copy {name} does not match the ASU: {mismatch}")

        coords = _coordinates(records)
        rotation, translation = _best_fit(reference_coords, coords, allow_reflection=False)
        deviation = float(_deviations(reference_coords, coords, rotation, translation).max())
        transform = _describe_transform(rotation, translation, reference_coords, coords)
        if deviation > tolerance:
            raise StructureError(
                f"chain {name} is not a rigid copy of {asu_chain} "
                f"(worst atom off by {deviation:.4f} A, tol={tolerance} A)"
            )
        if not transform.is_linear:
            raise StructureError(
                f"chain {name} is related to {asu_chain} by {transform.describe()}, "
                f"not a pure translation, so it cannot be an exact-translation frame"
            )
        offsets.append((transform.centroid_shift_a, transform.displacement))

    offsets.sort(key=lambda item: item[0])
    return [(0.0, 0.0, 0.0)] + [vector for _, vector in offsets]


def write_asu(source: StructureLike, asu_chain: str, out_path: Path) -> Path:
    """Write a one-ASU copy of the seed: `asu_chain` plus every non-polymer
    chain (ligand, fibril, solvent).

    The overlay only returns the externally supplied frames when the input has
    multiplicity 1; with more copies present it recomputes them from the atom
    array and discards the translations. Handing RFD3 a single protomer -- with
    the fibril still attached for context -- is what keeps the operator intact.
    """
    structure = load_structure(source)
    kept = gemmi.Structure()
    kept.name = structure.name
    kept.spacegroup_hm = structure.spacegroup_hm
    kept.cell = structure.cell
    model = gemmi.Model("1")

    for chain in structure[0]:
        is_polymer_chain = any(_is_polymer_residue(r) for r in chain)
        if is_polymer_chain and chain.name != asu_chain:
            continue
        copied = gemmi.Chain(chain.name)
        for residue in chain:
            copied.add_residue(residue)
        model.add_chain(copied)

    kept.add_model(model)
    kept.setup_entities()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept.write_pdb(str(out_path))
    return out_path


def has_ligand(source: StructureLike) -> List[str]:
    """Labels (chain:resname resnum) of non-water HETATM residues.
    Empty list means no ligand present."""
    structure = load_structure(source)
    ligands: List[str] = []
    for chain in structure[0]:
        for residue in chain:
            if residue.het_flag == "H" and not residue.is_water():
                ligands.append(f"{chain.name}:{residue.name}{residue.seqid.num}")
    return ligands


# Step 3. Standalone.


def parse_translation_spec(spec: str) -> np.ndarray:
    """Parse an RFD3_T2EXACT_TRANSLATIONS value into the inter-subunit vector."""
    vectors: List[np.ndarray] = []
    for part in (piece.strip() for piece in spec.split(";")):
        if not part:
            continue
        components = [c.strip() for c in part.split(",")]
        if len(components) != 3:
            raise ValueError(f"expected 3 comma-separated numbers per operator, got {part!r}")
        vectors.append(np.array([float(c) for c in components], dtype=float))
    if not vectors:
        raise ValueError("no translation operators found")
    if len(vectors) == 1:
        return vectors[0]
    return vectors[1] - vectors[0]


def format_translation_spec(displacement: Sequence[float], subunits: int = 2) -> str:
    """Render a measured displacement in RFD3_<ID>_TRANSLATIONS form."""
    if subunits < 2:
        raise ValueError(f"need at least 2 subunits, got {subunits}")
    step = np.array(displacement, dtype=float)
    rendered = []
    for index in range(subunits):
        operator = step * index
        # Render the identity as a plain '0,0,0', matching how these values are
        # written by hand in the env files.
        if not operator.any():
            rendered.append("0,0,0")
        else:
            rendered.append(",".join(f"{component:.4f}" for component in operator))
    return ";".join(rendered)


def compare_translation(
    measured: Sequence[float], expected: np.ndarray, tolerance: float
) -> Tuple[bool, np.ndarray, bool]:
    """Compare a measured displacement with an expected one."""
    measured_array = np.array(measured, dtype=float)
    forward = measured_array - expected
    reverse = -measured_array - expected
    if np.abs(reverse).max() < np.abs(forward).max():
        return bool(np.abs(reverse).max() <= tolerance), reverse, True
    return bool(np.abs(forward).max() <= tolerance), forward, False


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("structure_path", type=Path)
    parser.add_argument(
        "--tolerance", type=float, default=DEFAULT_TOLERANCE,
        help=f"max per-atom deviation in Angstrom (default: {DEFAULT_TOLERANCE})",
    )
    parser.add_argument(
        "--chains", default=None,
        help="comma-separated pair of chain names, e.g. A,B (default: the two protein chains)",
    )
    parser.add_argument(
        "--expect-translation", default=None, metavar="SPEC",
        help="cross-check the measured inter-chain translation against an expected "
             "one. Accepts the RFD3_T2EXACT_TRANSLATIONS format directly, e.g. "
             "--expect-translation \"$RFD3_T2EXACT_TRANSLATIONS\"",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    chain_names = [c.strip() for c in args.chains.split(",")] if args.chains else None
    if chain_names is not None and len(chain_names) != 2:
        print(f"ERROR: --chains needs exactly two names, got: {args.chains}")
        return 2

    try:
        structure = load_structure(args.structure_path)
        result = check_symmetry(structure, args.tolerance, chain_names)
    except (OSError, RuntimeError, StructureError) as exc:
        print(f"ERROR: {args.structure_path}: {exc}")
        return 2

    print(f"{'SYMMETRIC' if result.is_symmetric else 'NOT SYMMETRIC'}: {args.structure_path}")
    print(f"  {result.summary()}")
    if result.altlocs_collapsed:
        print(f"  note: collapsed {result.altlocs_collapsed} alternate conformer(s)")
    ligands = has_ligand(structure)
    print(f"  ligand: {', '.join(ligands) if ligands else '<none>'}")

    translation_ok = True
    if args.expect_translation and result.transform is not None:
        try:
            expected = parse_translation_spec(args.expect_translation)
        except ValueError as exc:
            print(f"ERROR: --expect-translation: {exc}")
            return 2
        matches, difference, flipped = compare_translation(
            result.transform.displacement, expected, args.tolerance
        )
        translation_ok = matches
        fmt = lambda v: ", ".join(f"{x:.4f}" for x in v)  # noqa: E731
        print(f"  expected t: ({fmt(expected)})  |t| = {float(np.linalg.norm(expected)):.4f} A")
        print(f"  measured t: ({fmt(result.transform.displacement)})  "
              f"|t| = {result.transform.centroid_shift_a:.4f} A"
              f"{'  [chain order reversed]' if flipped else ''}")
        print(f"  difference: ({fmt(difference)})  max = {float(np.abs(difference).max()):.4f} A "
              f"-> {'MATCH' if matches else 'MISMATCH'} (tol={args.tolerance} A)")

    return 0 if (result.is_symmetric and translation_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
