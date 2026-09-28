"""
Building AlphaFold3 input jsons, shared by every stage that folds.

Stage 04 folds one designed chain with the fibril it was designed against;
stage 08 folds the finished construct twice, once with the fibril and once
without. The ligand block, the bond search and the json shape are the same
work in both places, and a second copy would be a second thing to get wrong.

The ligand is read out of a structure, never assumed, and by now that means its
atoms rather than its name. The bonds are whichever atom pair is closest under
BOND_CUTOFF; the CCD code is whichever sugar ring the heavy-atom composition
says it is. Cellulose and chitin both come out right without either being named
at a call site, and a seed whose residues are called something else comes out
right too -- see LIGAND_CLASSES for why that matters.

SMILES would be the other way to describe a ligand, and it cannot be used here:
AF3's own documentation is explicit that "SMILES ligands don't support bonds:
there is no atom name that could be used to define the bond", and a fibre of
twenty rings is nineteen bonds. Writing the whole chain as one SMILES string
would avoid that, at the cost of having RDKit generate the fibre's conformation
from scratch -- which is the crystalline geometry the design was built against,
so predicting it rather than supplying it defeats the point of folding with a
fibre at all. userCCD is the documented route for a covalent ligand the
dictionary does not have, if one ever turns up.

The json's shape is copied from a file that is known to have produced a correct
cellulose complex on this cluster, field for field, rather than from the schema
documentation. Where the two disagree, the file that ran wins.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import gemmi

BOND_CUTOFF = 1.8      # angstroms, a C-O covalent bond is about 1.43

PROTEIN_CHAIN_ID = "A"
LIGAND_CHAIN_ID = "D"  # only a fallback; see ligand_chain_id below
CHAIN_SEPARATOR = ":"  # LigandMPNN's


class Af3JsonError(Exception):
    """A json could not be built."""


def ligand_chain_id(chain_name: str) -> str:
    """The id the ligand gets in the json: the seed's own chain name.

    This was hardcoded to "D", which works -- AF3 only needs the id to be
    consistent between the ligand entry and bondedAtomPairs, and not to collide
    with the protein. Carrying the seed's own name through costs nothing and
    means a json built here can be diffed line-for-line against one written by
    hand from the same seed, which is how the BGL fibre was finally caught.

    A name that would collide with the protein chain, or that is not a single
    character AF3 will take as a chain id, falls back to "D".
    """
    name = (chain_name or "").strip()
    if len(name) == 1 and name.isalnum() and name != PROTEIN_CHAIN_ID:
        return name
    return LIGAND_CHAIN_ID


# ---------------------------------------------------------------------------
# What the fibre actually is, read from its atoms
# ---------------------------------------------------------------------------
#
# The CCD code used to come straight from the residue name, which is how a
# cellulose seed whose residues are called BGL produced twenty molecules of
# 2-O-octyl-beta-D-glucopyranose -- a detergent. A residue NAME and a CCD CODE
# are different namespaces that happen to share spellings, and nothing checked.
#
# The first fix was a list of permitted names. That is not much better: it still
# decides from the name, and it needs extending every time a seed is written by
# a new tool. So the code is now decided from the ATOMS, which are the one thing
# in the file that cannot be misspelt:
#
#     count the heavy atoms by element, look at the atom names, and see which
#     kind of sugar ring that is. A hexose ring is C6 with five or six oxygens.
#     An N-acetylhexosamine has eight carbons and a nitrogen. An octyl glucoside
#     has fourteen carbons and a row of primed atom names, and matches nothing.
#
# The residue name is then used only to pick between codes WITHIN the class the
# atoms established. A name that is not a code for that class is ignored and
# said to be ignored, rather than being trusted or refusing the run.
#
# The honest limit, stated because it matters: composition cannot tell
# stereoisomers apart. BGC, GLC, BMA, MAN and GAL are all C6 H12 O6 and differ
# only in chirality, which is in the coordinates but not in the atom counts. So
# within a class the default is a policy of this pipeline -- cellulose, hence
# BGC -- not a measurement, and the log says so every time it is used.


class LigandClass:
    """A kind of sugar ring, recognised by composition rather than by name."""

    def __init__(self, label: str, default_code: str, codes: set,
                 carbons: int, nitrogens: int, oxygens: set, description: str):
        self.label = label
        self.default_code = default_code
        self.codes = codes
        self.carbons = carbons
        self.nitrogens = nitrogens
        self.oxygens = oxygens
        self.description = description

    def matches(self, counts: dict) -> bool:
        return (counts.get("C", 0) == self.carbons
                and counts.get("N", 0) == self.nitrogens
                and counts.get("O", 0) in self.oxygens
                and not set(counts) - {"C", "N", "O"})


# Oxygen counts are ranges because the two ends of a chain are not the same as
# its middle: an internal ring has lost the oxygen that became its glycosidic
# bond, the reducing end still carries O1. A dp20 cellulose chain is therefore
# nineteen rings of C6 O5 and one of C6 O6, and both are the same molecule.
LIGAND_CLASSES = (
    LigandClass(
        label="hexopyranose",
        default_code="BGC",
        codes={"BGC", "GLC", "BMA", "MAN", "GAL", "GLA", "ALL", "GXL"},
        carbons=6, nitrogens=0, oxygens={5, 6},
        description="a six-carbon sugar ring -- glucose, mannose or galactose; "
                    "cellulose is beta-D-glucose, BGC",
    ),
    LigandClass(
        label="N-acetylhexosamine",
        default_code="NAG",
        codes={"NAG", "NDG", "NGA", "A2G"},
        carbons=8, nitrogens=1, oxygens={5, 6, 7},
        description="an acetamido sugar ring; chitin is "
                    "N-acetyl-D-glucosamine, NAG",
    ),
    LigandClass(
        label="pentopyranose",
        default_code="XYP",
        codes={"XYP", "XYS", "ARA", "ARB"},
        carbons=5, nitrogens=0, oxygens={4, 5},
        description="a five-carbon sugar ring; xylan is beta-D-xylose, XYP",
    ),
)


def heavy_composition(residue: gemmi.Residue) -> dict:
    """{element symbol: count} over the residue's heavy atoms."""
    counts: dict = {}
    for atom in residue:
        symbol = atom.element.name
        if symbol in ("H", "D"):
            continue
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts


ELEMENT_ORDER = {"C": 0, "N": 1, "O": 2}


def describe_composition(counts: dict) -> str:
    """'C6 O5', in a fixed order so two of these can be compared by eye.

    Anything that is not C, N or O goes last rather than first, because it is
    the surprise in the string and the reader should reach it having already
    seen what the ring is: 'C6 O6 S1' says sulfated glucose, 'S1 C6 O6' says
    very little.
    """
    ordered = sorted(counts.items(),
                     key=lambda pair: (ELEMENT_ORDER.get(pair[0], 99), pair[0]))
    return " ".join(f"{symbol}{count}" for symbol, count in ordered) or "nothing"


def classify_residue(residue: gemmi.Residue) -> Tuple[Optional[LigandClass], dict]:
    """(the class its atoms put it in, its heavy-atom composition)."""
    counts = heavy_composition(residue)
    for candidate in LIGAND_CLASSES:
        if candidate.matches(counts):
            return candidate, counts
    return None, counts


def choose_code(residue: gemmi.Residue,
                override: Optional[str] = None) -> Tuple[str, str]:
    """(the CCD code to write, a sentence saying how it was decided).

    The sentence is returned rather than logged here so the caller can print it
    once per chain instead of once per ring, and so this stays testable without
    capturing output.
    """
    found, counts = classify_residue(residue)
    composition = describe_composition(counts)
    name = residue.name.strip().upper()

    if override:
        code = override.strip().upper()
        if found is None:
            return code, (f"{code} forced; the atoms ({composition}) match no sugar "
                          f"ring this pipeline knows, so nothing here can check it")
        if code not in found.codes:
            return code, (f"{code} forced, but the atoms ({composition}) are "
                          f"{found.label} and {code} is not one -- "
                          f"{found.description}. Folding it anyway because it was "
                          f"asked for explicitly")
        return code, f"{code} forced, and consistent with {found.label} ({composition})"

    if found is None:
        raise Af3JsonError(
            f"cannot tell what residue {residue.name!r} is. Its heavy atoms are "
            f"{composition}, which matches no sugar ring this pipeline knows "
            f"(hexose C6 O5-6, N-acetylhexosamine C8 N1 O5-7, pentose C5 O4-5). "
            f"A residue name is not a CCD code and is not trusted here, so pass "
            f"the right code explicitly with --ccd-code if you know it. "
            f"For reference, 2-O-octyl-beta-D-glucopyranose (BGL) is C14 O6 -- "
            f"if that is what this is, the seed is built from a detergent rather "
            f"than from cellulose."
        )

    if name in found.codes:
        return name, (f"{name}, taken from the residue name and confirmed by its "
                      f"atoms ({found.label}, {composition})")

    return found.default_code, (
        f"{found.default_code}, chosen from the atoms ({found.label}, "
        f"{composition}); the residue name {name!r} is not a {found.label} code "
        f"and was ignored. Stereochemistry is not read, so this is the "
        f"pipeline's default for this ring -- pass --ccd-code to say otherwise"
    )


def linking_atoms(first: gemmi.Residue, second: gemmi.Residue,
                  cutoff: float = BOND_CUTOFF) -> Optional[Tuple[str, str]]:
    """The atom names bonding two consecutive ligand residues, by distance."""
    best: Optional[Tuple[float, str, str]] = None
    for atom_a in first:
        for atom_b in second:
            distance = atom_a.pos.dist(atom_b.pos)
            if distance <= cutoff and (best is None or distance < best[0]):
                best = (distance, atom_a.name, atom_b.name)
    return (best[1], best[2]) if best else None


def odd_linkage(first_atom: str, second_atom: str) -> bool:
    """True if this pair is not the C-O a glycosidic bond always is.

    The bond is found by distance, which is what lets any polysaccharide work
    without naming its linkage. The cost is that slightly-off coordinates can
    put some other pair nearest -- C1 to C1, say -- and AF3 will fold that
    without complaint. A sugar chain bonded C-C is wrong, so it is worth saying
    so; it is not raised, because a future ligand may legitimately not be a
    sugar at all.
    """
    elements = {first_atom[:1], second_atom[:1]}
    return elements != {"C", "O"}


def bond_direction(bonds: Sequence[list]) -> str:
    """'O4->C1' or 'C1->O4': which way round the glycosidic bonds were found.

    Worth printing, because both are valid and they are not the same json. A
    beta-1,4 glucan bonds C1 of one ring to O4 of another; which of the two
    carries the lower residue number depends on which end of the fibre the seed
    was numbered from, and it is the seed that decides. Measuring it keeps the
    json honest about the coordinates it came from -- writing the other
    direction in would describe a polymer the input does not contain.
    """
    if not bonds:
        return "no bonds"
    first, second = bonds[0]
    return f"{first[2]}->{second[2]}"


def ligand_block(structure: gemmi.Structure, chain_name: str,
                 chain_id: Optional[str] = None,
                 ccd_code: Optional[str] = None
                 ) -> Tuple[dict, List[list], int, List[str], str]:
    """(the ligand entry, the bondedAtomPairs, how many residues, odd linkages,
    how the CCD code was decided).

    chain_id defaults to the seed's own chain name; see ligand_chain_id.
    ccd_code forces a code for every unit; without it the code is read from the
    residues' atoms, as choose_code explains.

    The last element is a sentence for the caller to log. It is returned rather
    than printed because this is the one decision in the file that has already
    silently produced twenty molecules of the wrong compound, and a decision
    like that should be in the run log whether or not anyone thought to ask.
    """
    if chain_id is None:
        chain_id = ligand_chain_id(chain_name)
    residues = list(structure[0][chain_name])
    if not residues:
        raise Af3JsonError(f"ligand chain {chain_name} is empty")

    # Decided per residue, then required to agree: a chain whose rings are not
    # all the same thing is a heteropolymer, and writing one code for all of
    # them would fold something the seed does not contain.
    decisions = [choose_code(residue, ccd_code) for residue in residues]
    codes = [code for code, _ in decisions]
    distinct = sorted(set(codes))
    if len(distinct) > 1:
        raise Af3JsonError(
            f"ligand chain {chain_name} is not one kind of ring: its residues "
            f"resolve to {distinct}. This pipeline writes one code per unit of a "
            f"uniform fibre; a mixed chain needs its json written by hand."
        )
    reason = decisions[0][1]

    bonds: List[list] = []
    missing: List[int] = []
    odd: List[str] = []
    for index in range(len(residues) - 1):
        pair = linking_atoms(residues[index], residues[index + 1])
        if pair is None:
            missing.append(index + 1)
            continue
        if odd_linkage(pair[0], pair[1]):
            odd.append(f"{index + 1}{pair[0]}-{index + 2}{pair[1]}")
        bonds.append([[chain_id, index + 1, pair[0]],
                      [chain_id, index + 2, pair[1]]])
    if missing:
        raise Af3JsonError(
            f"ligand chain {chain_name}: no atom pair within {BOND_CUTOFF} A between "
            f"residue(s) {missing[:4]} and the next -- the chain would be folded as "
            f"disconnected fragments"
        )
    return {"id": chain_id, "ccdCodes": codes}, bonds, len(residues), odd, reason


def build_json(job_name: str, sequence: str, ligand: Optional[dict] = None,
               bonds: Sequence[list] = ()) -> dict:
    """The AF3 input, shaped like the file that is known to have run here.

    Every field below is copied from a json that produced a correct cellulose
    complex on this cluster, rather than from the schema docs:

        version 2         what the installed AF3 wants; a 1 is a different
                          dialect and an earlier guess of mine.
        id "A"            a string, not a one-element list.
        modifications []  present and empty.
        unpairedMsa ""    empty, which is AF3's "fold this sequence alone, do
                          not search". An earlier version of this function put
                          the sequence here as a one-entry MSA instead, which
                          is a different instruction.
        userCCD null      present even with nothing to define.

    ligand=None gives a protein-only fold. The ligand entry and bondedAtomPairs
    are then left out altogether rather than written empty: AF3 reads an empty
    bondedAtomPairs list as a claim that there are no bonds to make, which is
    a different statement from there being no ligand.
    """
    sequences: List[dict] = [
        {
            "protein": {
                "id": PROTEIN_CHAIN_ID,
                "sequence": sequence,
                "modifications": [],
                "unpairedMsa": "",
                "pairedMsa": "",
                "templates": [],
            }
        }
    ]
    payload = {
        "dialect": "alphafold3",
        "version": 2,
        "name": job_name,
        "sequences": sequences,
        "modelSeeds": [1],
    }
    if ligand is not None:
        sequences.append({"ligand": ligand})
        payload["bondedAtomPairs"] = list(bonds)
    payload["userCCD"] = None
    return payload


# ---------------------------------------------------------------------------
# Self-test: run `python3 common/af3_json.py` to check an unpacked copy
# ---------------------------------------------------------------------------

# A glucose ring in local coordinates, laid out so that O4 sits 1.43 A from the
# next ring's C1 and nothing else crosses BOND_CUTOFF. Eleven heavy atoms: an
# internal residue of a chain has no O1, which is why the mass ratio against
# BGC's tabulated 180.2 lands near 0.84 rather than 1.0.
_RING = [
    ("C1", 0.00, 0.00, 0.00), ("C2", 1.30, 0.60, 0.00),
    ("C3", 2.20, 0.00, 1.00), ("C4", 2.90, 0.90, 0.00),
    ("C5", 1.60, -0.80, 1.20), ("C6", 2.20, -1.60, 2.30),
    ("O2", 1.00, 1.50, -1.00), ("O3", 3.00, -0.70, 1.80),
    ("O4", 3.77, 0.00, 0.00), ("O5", 0.60, -0.60, 1.00),
    ("O6", 3.00, -2.30, 1.90),
]
_REPEAT = 5.20

# C1 and O4 keep their places in every variant below, so the bond search finds
# the same linkage and only the composition changes -- which is the one thing
# these tests are about.

# N-acetylhexosamine: O2 becomes N2, and the acetyl group adds C7, C8, O7.
_RING_NAG = ([entry for entry in _RING if entry[0] != "O2"]
             + [("N2", 1.00, 1.50, -1.00), ("C7", 0.40, 2.60, -1.40),
                ("C8", 1.00, 3.40, -2.50), ("O7", -0.70, 2.90, -0.90)])

# Pentose: no C6, and therefore no O6.
_RING_XYP = [entry for entry in _RING if entry[0] not in ("C6", "O6")]

# 2-O-octyl-beta-D-glucopyranose: a hexose ring wearing an eight-carbon tail.
# This is what BGL actually means, and what AF3 built twenty of.
_RING_OCTYL = _RING + [(f"C{index}'", 1.00 + index * 1.25, 2.60, -1.00 - index * 0.4)
                       for index in range(1, 9)]


def _fibre(count: int, name: str, chain: str, reverse: bool = False,
           ring=None):
    structure = gemmi.Structure()
    model = gemmi.Model("1")
    sugar = gemmi.Chain(chain)
    order = range(count - 1, -1, -1) if reverse else range(count)
    for number, index in enumerate(order, start=1):
        residue = gemmi.Residue()
        residue.name, residue.seqid = name, gemmi.SeqId(str(number))
        for atom_name, x, y, z in (ring if ring is not None else _RING):
            atom = gemmi.Atom()
            atom.name = atom_name
            atom.element = gemmi.Element(atom_name[0])
            atom.pos = gemmi.Position(x + index * _REPEAT, y, z)
            residue.add_atom(atom)
        sugar.add_residue(residue)
    model.add_chain(sugar)
    structure.add_model(model)
    structure.setup_entities()
    return structure


def test_shape() -> None:
    """The json shape, against the file that is known to have run here.

    Not a style check. Every one of these was wrong at some point and AF3 either
    refused the file or folded something other than what was meant.
    """
    ligand, bonds, size, odd, _ = ligand_block(_fibre(4, "BGC", "M"), "M")
    payload = build_json("demo_ligand", "MKTAYIAK", ligand, bonds)

    assert list(payload) == ["dialect", "version", "name", "sequences",
                             "modelSeeds", "bondedAtomPairs", "userCCD"], list(payload)
    assert payload["version"] == 2, payload["version"]
    assert payload["userCCD"] is None
    protein = payload["sequences"][0]["protein"]
    assert protein["id"] == "A", protein["id"]          # a string, not ["A"]
    assert protein["unpairedMsa"] == "", protein["unpairedMsa"]
    assert protein["modifications"] == []
    assert payload["sequences"][1]["ligand"]["id"] == "M"  # the seed's own name
    assert (size, odd) == (4, [])

    apo = build_json("demo_apo", "MKTAYIAK")
    assert "bondedAtomPairs" not in apo
    assert len(apo["sequences"]) == 1


def test_direction() -> None:
    """The bond direction follows the seed's numbering, not a choice here.

    The same polymer numbered from the other end gives the other direction, and
    both are correct. Writing one in would describe a fibre the coordinates do
    not contain, so it is measured; this asserts the measurement tracks.
    """
    _, forward, _, _, _ = ligand_block(_fibre(4, "BGC", "M"), "M")
    _, backward, _, _, _ = ligand_block(_fibre(4, "BGC", "M", reverse=True), "M")
    assert bond_direction(forward) == "O4->C1", bond_direction(forward)
    assert bond_direction(backward) == "C1->O4", bond_direction(backward)
    assert len(forward) == len(backward) == 3


def test_identification() -> None:
    """The code comes from the atoms, and the name cannot override them.

    Every case here is one the name-based version got wrong or could have: the
    seed named BGL that built a detergent, a chitin seed named anything at all,
    and a seed that really does hold something exotic and must not be guessed at.
    """
    # cellulose named correctly: the name is used, and confirmed
    ligand, _, _, _, why = ligand_block(_fibre(4, "BGC", "M"), "M")
    assert ligand["ccdCodes"] == ["BGC"] * 4
    assert "confirmed by its atoms" in why, why

    # THE BUG: cellulose whose residues are named BGL. No flag, no whitelist --
    # the atoms are a hexose ring, BGL is not a hexose code, so BGL is dropped.
    ligand, _, _, _, why = ligand_block(_fibre(4, "BGL", "M"), "M")
    assert ligand["ccdCodes"] == ["BGC"] * 4, ligand["ccdCodes"]
    assert "ignored" in why and "BGL" in why, why

    # a name that is a legal hexose code but not the default is kept, because
    # the seed knows its own stereochemistry and this file does not read it
    ligand, _, _, _, why = ligand_block(_fibre(4, "GAL", "M"), "M")
    assert ligand["ccdCodes"] == ["GAL"] * 4, ligand["ccdCodes"]

    # chitin, identified by its nitrogen, whatever the residues are called
    for name in ("NAG", "NDG", "WHATEVER"):
        ligand, _, _, _, _ = ligand_block(_fibre(4, name, "M", ring=_RING_NAG), "M")
        expected = name if name in ("NAG", "NDG") else "NAG"
        assert ligand["ccdCodes"] == [expected] * 4, (name, ligand["ccdCodes"])

    # xylan
    ligand, _, _, _, _ = ligand_block(_fibre(4, "XYP", "M", ring=_RING_XYP), "M")
    assert ligand["ccdCodes"] == ["XYP"] * 4

    # a residue that really IS an octyl glucoside is refused, not guessed at --
    # and the refusal says what the atoms are rather than what the name is
    try:
        ligand_block(_fibre(4, "BGL", "M", ring=_RING_OCTYL), "M")
    except Af3JsonError as exc:
        assert "C14" in str(exc), exc
    else:
        raise AssertionError("an octyl glucoside was accepted as a sugar ring")

    # ... unless it is asked for by name, which is what the override is for
    ligand, _, _, _, why = ligand_block(
        _fibre(4, "BGL", "M", ring=_RING_OCTYL), "M", ccd_code="BGL")
    assert ligand["ccdCodes"] == ["BGL"] * 4
    assert "forced" in why, why

    # an override that contradicts the atoms is obeyed, and says so loudly
    _, _, _, _, why = ligand_block(_fibre(4, "BGC", "M"), "M", ccd_code="NAG")
    assert "not one" in why and "hexopyranose" in why, why

    # A decorated ring is not a plain one. Counting C, N and O and ignoring
    # everything else would call a sulfated or phosphorylated glucose BGC and
    # fold it without its substituent, which is a different molecule binding a
    # different way -- so any other element has to refuse.
    decorated = _RING + [("S1", 1.00, 2.60, -1.20), ("O1S", 1.80, 3.40, -1.60)]
    try:
        ligand_block(_fibre(4, "BGC", "M", ring=decorated), "M")
    except Af3JsonError as exc:
        assert "S1" in str(exc), exc
    else:
        raise AssertionError("a sulfated ring was accepted as plain glucose")


def test_reducing_end() -> None:
    """The two ends of a chain are not the middle, and are the same molecule.

    An internal ring has lost the oxygen that became its glycosidic bond; the
    reducing end still carries O1. A composition check with a single exact
    oxygen count would reject every real fibre on its last residue.
    """
    structure = _fibre(4, "BGC", "M")
    last = list(structure[0]["M"])[-1]
    atom = gemmi.Atom()
    atom.name, atom.element = "O1", gemmi.Element("O")
    atom.pos = gemmi.Position(-1.2, 0.6, 0.0)
    last.add_atom(atom)
    ligand, _, _, _, _ = ligand_block(structure, "M")
    assert ligand["ccdCodes"] == ["BGC"] * 4, ligand["ccdCodes"]


if __name__ == "__main__":
    test_shape()
    test_direction()
    test_identification()
    test_reducing_end()
    print("af3_json: shape, bond direction and ligand identification all pass")

