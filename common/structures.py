"""
Structure handling shared by every stage that opens a pdb or cif.
"""
from __future__ import annotations

from typing import List, Tuple

import gemmi


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

