"""
Structure handling shared by every stage that opens a pdb or cif.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import gemmi
import numpy as np

# Any protein atom within this of a ligand atom counts as a contact when
# deciding which ligand chain the protein actually sits on.
CONTACT_CUTOFF = 8.0

# The tighter cutoff that asks whether a residue is actually ON the fibre, used
# to check that a prediction kept its binding site where the design put it.
# Deliberately not CONTACT_CUTOFF: 8 A is generous enough to pick the right
# chain out of several, which is a much weaker question than whether one
# residue is engaged.
SITE_CONTACT_CUTOFF = 6.0


class StructureError(Exception):
    """A structure could not be interpreted."""


def classify_chains(structure: gemmi.Structure) -> Tuple[List[str], List[str]]:
    """(protein chains, everything else), decided by residue type.

    A chain counts as protein if it holds any amino acid. gemmi's residue table
    knows BGL and NAG are not amino acids, so cellulose and chitin both land in
    the second list without either being named anywhere -- which is what lets
    one pipeline serve any polysaccharide.
    """
    protein: List[str] = []
    ligand: List[str] = []
    for chain in structure[0]:
        is_protein = False
        for residue in chain:
            info = gemmi.find_tabulated_residue(residue.name)
            if info is not None and info.is_amino_acid():
                is_protein = True
                break
        (protein if is_protein else ligand).append(chain.name)
    return protein, ligand


def drop_chains(structure: gemmi.Structure, names: List[str]) -> int:
    """Remove the named chains in place. Returns how many went."""
    model = structure[0]
    removed = 0
    for name in list(names):
        for index in range(len(model) - 1, -1, -1):
            if model[index].name == name:
                del model[index]
                removed += 1
    return removed


def chain_atoms(chain: gemmi.Chain) -> np.ndarray:
    """Every atom of one chain as an (n, 3) array."""
    points = [[atom.pos.x, atom.pos.y, atom.pos.z]
              for residue in chain for atom in residue]
    return np.array(points) if points else np.zeros((0, 3))


def most_contacted(structure: gemmi.Structure, protein_chains: Sequence[str],
                   ligand_chains: Sequence[str],
                   cutoff: float = CONTACT_CUTOFF) -> Tuple[str, int]:
    """(ligand chain name, contact count) for the chain the protein binds most."""
    protein_points = np.vstack(
        [chain_atoms(structure[0][name]) for name in protein_chains]
    ) if protein_chains else np.zeros((0, 3))
    if protein_points.size == 0:
        raise StructureError("the complex has no protein atoms")

    best_name, best_count = None, -1
    for name in ligand_chains:
        points = chain_atoms(structure[0][name])
        if points.size == 0:
            continue
        distances = np.linalg.norm(
            protein_points[:, None, :] - points[None, :, :], axis=2
        )
        count = int((distances <= cutoff).sum())
        if count > best_count:
            best_name, best_count = name, count
    if best_name is None:
        raise StructureError("no ligand chain in the complex")
    return best_name, best_count


def keep_one_ligand_chain(structure: gemmi.Structure,
                          cutoff: float = CONTACT_CUTOFF
                          ) -> Tuple[List[str], str, int, List[str]]:
    """Reduce the structure to its protein plus ONE ligand chain, in place.

    Returns (protein chains, the ligand chain kept, its contact count, the
    ligand chains dropped).

    A seed can sandwich several fibril chains. Design and fold both have to see
    the same one or they are not talking about the same binding site, so the
    choice is made here, once, by contact count -- and both stages call this
    rather than each deciding for itself.
    """
    protein_chains, ligand_chains = classify_chains(structure)
    if not protein_chains:
        raise StructureError("no protein chain in the complex")
    if not ligand_chains:
        raise StructureError("no ligand chain in the complex")
    kept, count = most_contacted(structure, protein_chains, ligand_chains, cutoff)
    dropped = [name for name in ligand_chains if name != kept]
    if dropped:
        drop_chains(structure, dropped)
    return protein_chains, kept, count, dropped


# ---------------------------------------------------------------------------
# How close a residue sits to the fibre
# ---------------------------------------------------------------------------

def heavy_atoms(chain: gemmi.Chain) -> np.ndarray:
    """Every non-hydrogen atom of one chain as an (n, 3) array.

    Separate from chain_atoms because hydrogens are the difference between a
    measurement that means the same thing in two files and one that does not:
    a packed seed may carry them, an AlphaFold model does not, and a hydrogen
    reaches about 1 A further than the atom it hangs off. Comparing a distance
    measured with them against one measured without is comparing two cutoffs.
    """
    points = [[atom.pos.x, atom.pos.y, atom.pos.z]
              for residue in chain for atom in residue
              if atom.element != gemmi.Element("H")
              and atom.element != gemmi.Element("D")]
    return np.array(points) if points else np.zeros((0, 3))


def site_approaches(structure: gemmi.Structure, protein_chain: str,
                    residue_numbers: Sequence[int], ligand_chain: str
                    ) -> Dict[int, float]:
    """{residue number: its closest heavy-atom approach to the ligand chain}.

    Only the residues asked for that are actually in the chain appear; a number
    with no residue is left out rather than reported as infinitely far, because
    those are two different findings and the caller can tell them apart by
    counting.

    Per residue rather than one number for the whole site: a mean would let one
    residue buried in the fibre carry four that have swung away, which is the
    failure this is meant to catch.
    """
    model = structure[0]
    ligand_points = heavy_atoms(model[ligand_chain])
    if ligand_points.size == 0:
        raise StructureError(f"ligand chain {ligand_chain} has no heavy atoms")

    wanted = set(residue_numbers)
    approaches: Dict[int, float] = {}
    for residue in model[protein_chain]:
        number = residue.seqid.num
        if number not in wanted:
            continue
        points = np.array([[atom.pos.x, atom.pos.y, atom.pos.z]
                           for atom in residue
                           if atom.element != gemmi.Element("H")
                           and atom.element != gemmi.Element("D")])
        if points.size == 0:
            continue
        distances = np.linalg.norm(
            points[:, None, :] - ligand_points[None, :, :], axis=2
        )
        approaches[number] = float(distances.min())
    return approaches


def chain_approach(structure: gemmi.Structure, protein_chain: str,
                   ligand_chain: str) -> Optional[float]:
    """The closest any heavy atom of the protein chain comes to the ligand.

    The blunt question behind the per-residue one: is this monomer on the fibre
    at all, or did it fold correctly somewhere else entirely?
    """
    model = structure[0]
    protein_points = heavy_atoms(model[protein_chain])
    ligand_points = heavy_atoms(model[ligand_chain])
    if protein_points.size == 0 or ligand_points.size == 0:
        return None
    distances = np.linalg.norm(
        protein_points[:, None, :] - ligand_points[None, :, :], axis=2
    )
    return float(distances.min())

