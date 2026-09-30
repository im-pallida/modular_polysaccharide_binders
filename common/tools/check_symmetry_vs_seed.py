#!/usr/bin/env python3
"""
Does a design's two copies sit the way its seed's two copies do?

    check_symmetry_vs_seed.py <seed.pdb> <design.cif|.tar.gz> [more ...]

Stage 02 sorts on clashes and contacts, neither of which notices a second copy
rotated the wrong way about the right axis -- the fault that shipped while the
exact frames were handed to RFD3 untransposed. This measures the thing that was
wrong: the rotation, the rise and the copy-to-copy spread of the operator
relating the two protein chains, compared with the seed's own.

Correspondence-free. Within one structure the two chains hold the same residues
in the same order, so no alignment between seed and design is needed -- only
each structure's own A-to-B operator, which is what has to match.
"""
from __future__ import annotations

import sys
import tarfile
import tempfile
from pathlib import Path

import gemmi
import numpy as np

TOLERANCE_DEG = 2.0
TOLERANCE_RISE_A = 1.0


def protein_chains(structure):
    out = []
    for chain in structure[0]:
        if any((info := gemmi.find_tabulated_residue(r.name)) and info.is_amino_acid()
               for r in chain):
            out.append(chain)
    return out


def alpha_carbons(chain) -> np.ndarray:
    points = [[a.pos.x, a.pos.y, a.pos.z]
              for r in chain for a in r if a.name == "CA"]
    return np.array(points)


def kabsch(first, second):
    ca, cb = first.mean(0), second.mean(0)
    u, _, vt = np.linalg.svd((first - ca).T @ (second - cb))
    R = vt.T @ u.T
    if np.linalg.det(R) < 0:
        R = vt.T @ np.diag([1.0, 1.0, -1.0]) @ u.T
    return R, cb - R @ ca


def screw_parameters(R, t):
    """(angle, signed rise) with the axis oriented so the rotation is +angle.

    THE SIGN IS THE WHOLE POINT. A rotation and its inverse have the SAME angle,
    so an angle alone cannot tell a faithful copy from one turned the wrong way
    -- and an axis taken from an eigenvector has an arbitrary sign, which makes
    the rise look the same too. Both of those fooled the first version of this
    file, and the first analysis that used it.

    So the axis comes from the ANTISYMMETRIC part of R, which fixes its
    handedness: vec(R - R.T)/2 = sin(angle) * axis. The rotation is then always
    +angle about that axis, and the rise -- the component of t along it --
    carries a sign that FLIPS when the rotation inverts. That is the number that
    separates the two.
    """
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    sin_a = float(np.linalg.norm(w))
    cos_a = float(np.clip((np.trace(R) - 1) / 2, -1, 1))
    angle = float(np.degrees(np.arctan2(sin_a, cos_a)))
    if sin_a < 1e-6:
        # No turn (a pure translation) or a half-turn: no handedness to compare.
        return angle, None
    return angle, float((w / sin_a) @ t)


def operator(structure):
    """(angle, signed rise, min copy-to-copy, max copy-to-copy, fit rmsd)."""
    chains = protein_chains(structure)
    if len(chains) != 2:
        raise ValueError(f"{len(chains)} protein chain(s), expected 2")
    a, b = alpha_carbons(chains[0]), alpha_carbons(chains[1])
    if a.shape != b.shape or not len(a):
        raise ValueError(f"chains hold {len(a)} and {len(b)} CA atoms")
    R, t = kabsch(a, b)
    angle, rise = screw_parameters(R, t)
    spread = np.linalg.norm(a - b, axis=1)
    rmsd = float(np.sqrt(((a @ R.T + t - b) ** 2).sum(1).mean()))
    return angle, rise, spread.min(), spread.max(), rmsd


def _rise(value) -> str:
    return "     n/a" if value is None else f"{value:+8.3f} A"


def structures(path: Path):
    """(name, structure) for a cif/pdb, or for every one inside a tarball."""
    if path.suffix in (".gz", ".tar", ".tgz") or ".tar." in path.name:
        with tarfile.open(path, "r:*") as archive:
            for member in sorted(archive.getnames()):
                if not member.endswith((".cif", ".pdb")):
                    continue
                with tempfile.TemporaryDirectory() as tmp:
                    # filter= only exists on newer pythons; the cluster venv is 3.9
                    try:
                        archive.extract(member, tmp, filter="data")
                    except TypeError:
                        archive.extract(member, tmp)
                    yield member, gemmi.read_structure(str(Path(tmp) / member))
    else:
        yield path.name, gemmi.read_structure(str(path))


def main(argv) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    seed_path, *designs = argv
    _, seed = next(structures(Path(seed_path)))
    s_angle, s_rise, s_min, s_max, _ = operator(seed)
    print(f"seed {Path(seed_path).name}")
    print(f"  operator {s_angle:7.2f} deg, rise {_rise(s_rise)}, "
          f"copy-to-copy {s_min:.2f} - {s_max:.2f} A")
    if s_rise is None:
        print("  (no turn in this operator, so there is no handedness to compare)")
    print(f"  tolerance: {TOLERANCE_DEG} deg, {TOLERANCE_RISE_A} A\n")

    bad = 0
    for target in designs:
        for name, structure in structures(Path(target)):
            try:
                angle, rise, lo, hi, rmsd = operator(structure)
            except ValueError as exc:
                print(f"  {'SKIP':6} {name}: {exc}")
                continue
            off_angle = abs(angle - s_angle)
            # SIGNED comparison. An inverted copy matches on |rise| and differs
            # by 2x the rise once the sign is honest.
            off_rise = (abs(rise - s_rise) if (rise is not None and s_rise is not None)
                        else 0.0)
            ok = off_angle <= TOLERANCE_DEG and off_rise <= TOLERANCE_RISE_A
            bad += not ok
            print(f"  {'ok' if ok else 'WRONG':6} {name}")
            print(f"         {angle:7.2f} deg ({off_angle:+.2f}), "
                  f"rise {_rise(rise)} ({off_rise:+.3f}), "
                  f"copy-to-copy {lo:.2f} - {hi:.2f} A, fit rmsd {rmsd:.3f} A")
    if bad:
        print(f"\n{bad} structure(s) do NOT reproduce the seed's operator. A copy "
              f"turned the wrong way\nabout the right axis keeps the rise and "
              f"changes the separation, so it survives a\nclash-and-contact "
              f"filter -- do not design sequences for these.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
