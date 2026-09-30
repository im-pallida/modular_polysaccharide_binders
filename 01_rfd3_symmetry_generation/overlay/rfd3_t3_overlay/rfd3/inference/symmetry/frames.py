import numpy as np
import torch


def _normalize_symmetry_id(symmetry_id):
    """
    RFD3 sometimes passes a SymmetryConfig object here instead of a plain
    string. Unwrap it once, here, so every branch below only ever deals
    with a plain string id. Replaces two separate, inconsistent unwrap
    attempts that existed in earlier overlay patches.
    """
    if hasattr(symmetry_id, "id"):
        return symmetry_id.id
    return symmetry_id


def get_symmetry_frames_from_symmetry_id(symmetry_id):
    """
    Get symmetry frames from a symmetry id.
    Arguments:
        symmetry_id: string of the symmetry id (or a SymmetryConfig-like
            object exposing an `.id` attribute)
    Returns:
        frames: list of (rotation_matrix, translation_vector) tuples
    """
    from rfd3.inference.symmetry.symmetry_utils import SymmetryConfig

    sym_conf = {}
    if isinstance(symmetry_id, SymmetryConfig):
        sym_conf = symmetry_id

    symmetry_id = _normalize_symmetry_id(symmetry_id)
    sid = str(symmetry_id).lower()

    # ------------------------------------------------------------------
    # T3_LINEAR_OVERLAY — custom additions, not present in stock RFD3.
    #
    #   T<N>       : N copies, evenly spaced along one axis, no rotation
    #                copy_i = unit + i * rise * axis
    #   SCREW<N>   : N copies, evenly spaced + rotated per step
    #                copy_i = R(i * theta) @ unit + i * rise * axis
    #   T<N>EXACT  : N copies, exact user-supplied translations, read
    #                from RFD3_T<N>EXACT_TRANSLATIONS
    #
    # Frames always follow RFD3's existing format: frame_i = (R_i, T_i)
    #
    # If none of these match, execution falls through unchanged to
    # stock RFD3 behavior below (C<N>, D<N>, input_defined) — so any
    # symmetry_id that isn't one of the custom types above is handled
    # identically to the real installed rfd3 package.
    # ------------------------------------------------------------------

    if sid.startswith("t") and sid[1:].isdigit():
        return _translational_frames(order=int(sid[1:]))

    if sid.startswith("screw") and sid[5:].isdigit():
        return _screw_frames(order=int(sid[5:]))

    if sid.startswith("t") and sid.endswith("exact") and sid[1:-5].isdigit():
        return _exact_translational_frames(order=int(sid[1:-5]))

    # ------------------------------------------------------------------
    # Stock RFD3 behavior — unmodified.
    # ------------------------------------------------------------------

    if symmetry_id.lower().startswith("c"):
        order = int(symmetry_id[1:])
        frames = get_cyclic_frames(order)
    elif symmetry_id.lower().startswith("d"):
        order = int(symmetry_id[1:])
        frames = get_dihedral_frames(order)
    elif symmetry_id.lower() == "input_defined":
        assert (
            sym_conf.symmetry_file is not None
        ), "symmetry_file is required for input_defined symmetry"
        frames = get_frames_from_file(sym_conf.symmetry_file)
    else:
        raise ValueError(
            f"Symmetry id {symmetry_id} not supported. This overlay routes:\n"
            f"  T<N>EXACT  exact frames from RFD3_T<N>EXACT_TRANSLATIONS; each\n"
            f"             frame is x,y,z or r11..r33,x,y,z, so a screw about\n"
            f"             ANY axis fits here\n"
            f"  T<N>       even spacing, RFD3_T3_RISE / RFD3_T3_AXIS\n"
            f"  SCREW<N>   even spacing plus rotation, RFD3_SCREW_RISE /\n"
            f"             RFD3_SCREW_ANGLE_DEG / RFD3_SCREW_AXIS -- axis must\n"
            f"             be x, y or z\n"
            f"  C<N>, D<N> stock RFD3 cyclic and dihedral\n"
            f"Note the order is part of the id: 'SCREW' alone does not route, "
            f"'SCREW2' does. A screw measured off a seed belongs in T<N>EXACT, "
            f"not SCREW<N>, because SCREW<N> cannot express an arbitrary axis."
        )

    for R, _ in frames:
        assert is_valid_rotation_matrix(R), f"Frame {R} is not a valid rotation matrix"

    return frames


def _axis_vector(axis_name, env_var_name):
    axes = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }
    if axis_name not in axes:
        raise ValueError(f"{env_var_name} must be x, y, or z, got {axis_name}")
    return axes[axis_name]


def _axis_vector_and_rotator(axis_name, env_var_name):
    def rot_x(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)

    def rot_y(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)

    def rot_z(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)

    axes = {
        "x": (np.array([1.0, 0.0, 0.0]), rot_x),
        "y": (np.array([0.0, 1.0, 0.0]), rot_y),
        "z": (np.array([0.0, 0.0, 1.0]), rot_z),
    }
    if axis_name not in axes:
        raise ValueError(f"{env_var_name} must be x, y, or z, got {axis_name}")
    return axes[axis_name]


def _translational_frames(order):
    import os
    rise = float(os.environ.get("RFD3_T3_RISE", "35.0"))
    axis_name = os.environ.get("RFD3_T3_AXIS", "x").lower()
    axis = _axis_vector(axis_name, "RFD3_T3_AXIS")

    R = np.eye(3)
    return [(R.copy(), (i * rise * axis).copy()) for i in range(order)]


def _screw_frames(order):
    import os
    rise = float(os.environ.get("RFD3_SCREW_RISE", "35.0"))
    theta_deg = float(os.environ.get("RFD3_SCREW_ANGLE_DEG", "0.0"))
    axis_name = os.environ.get("RFD3_SCREW_AXIS", "x").lower()
    axis, rot_fn = _axis_vector_and_rotator(axis_name, "RFD3_SCREW_AXIS")
    theta = np.deg2rad(theta_deg)

    frames = []
    for i in range(order):
        R = rot_fn(i * theta)
        T = i * rise * axis
        frames.append((R.copy(), T.copy()))
    return frames


def _exact_translational_frames(order):
    """
    Replaces the old separate T2EXACT / T3EXACT hardcoded blocks.
    Works for any T<N>EXACT — the digit in the symmetry id now
    authoritatively determines how many translations are required,
    fixing the earlier inconsistency where T3EXACT's name implied 3
    frames but a patch had silently loosened it to "at least 2".

    T3_LINEAR_OVERLAY — each frame may now be EITHER

        x,y,z                  three numbers: a translation, identity rotation
        r11,...,r33,x,y,z      twelve numbers: a full rigid motion, rotation
                               row-major, applied as  copy = R @ unit + t

    The twelve-number form exists because SCREW<N> can only turn about x, y or
    z, and a real seed's screw axis is wherever the crystal put it: a chitin
    pair related by 45 deg about (0.428, -0.601, -0.675) has no axis-aligned
    spelling. Measuring the operator off the seed and writing the matrix out
    avoids having to name an axis at all.

    Three numbers stays the default spelling for a pure translation, so every
    existing RFD3_T<N>EXACT_TRANSLATIONS keeps working untouched.
    """
    import os
    env_name = f"RFD3_T{order}EXACT_TRANSLATIONS"
    trans_text = os.environ.get(env_name, "")
    if not trans_text:
        raise ValueError(f"{env_name} is required, e.g. 0,0,0;0,0,41.52")

    frames = []
    for item in trans_text.split(";"):
        item = item.strip()
        if not item:
            continue
        vals = [float(x) for x in item.split(",")]
        if len(vals) == 3:
            R = np.eye(3, dtype=np.float32)
            T = np.array(vals, dtype=np.float32)
        elif len(vals) == 12:
            R = np.array(vals[:9], dtype=np.float32).reshape(3, 3)
            T = np.array(vals[9:], dtype=np.float32)
            if not is_valid_rotation_matrix(np.asarray(R, dtype=float)):
                raise ValueError(
                    f"Bad frame {item!r} in {env_name}: the first nine numbers are "
                    f"not a rotation matrix (R @ R.T is not the identity). They are "
                    f"read row-major, r11,r12,r13,r21,..."
                )
            # TRANSPOSED HERE, AT THE BOUNDARY.
            #
            # The spec is written the way the operator is measured off the seed
            # and the way anyone would write it down:  copy = R @ asu + t.
            # RFD3 applies a frame in the ROW-vector convention, coords @ R + t,
            # so handing it R unchanged applies R-transpose -- which for a
            # rotation is its inverse.
            #
            # Measured on a real chitin pair: the produced copy-to-copy rotation
            # came out 89.93 deg away from the emitted operator and 0.14 deg away
            # from its transpose, and reapplying coords @ R + t reproduced the
            # produced chain to 0.058 A against 18.0 A for R @ coords + t.
            #
            # Pure translations were immune, the identity being its own
            # transpose, which is why every translational seed worked and this
            # only surfaced on the first seed with a rotation in its operator.
            R = R.T.copy()
        else:
            raise ValueError(
                f"Bad frame {item!r} in {env_name}: expected 3 numbers (x,y,z) or "
                f"12 (r11..r33,x,y,z), got {len(vals)}"
            )
        frames.append((R, T))

    if len(frames) != order:
        raise ValueError(
            f"{env_name} declares {order} frames but got {len(frames)} translations"
        )

    return frames


def get_symmetry_frames_from_atom_array(src_atom_array, input_frames):
    """
    Get symmetry frames from an atom array. Adapted from code from FD
    Arguments:
        src_atom_array: atom array with coordinates and chain/residue information
        input_frames: list of (rotation_matrix, translation_vector) tuples
    Returns:
        computed_frames: list of (rotation_matrix, translation_vector) tuples (updated)
    """
    # import within the function to avoid circular import
    from rfd3.inference.symmetry.checks import (
        check_input_frames_match_symmetry_frames,
        check_max_rmsds,
        check_max_transforms,
        check_min_atoms_to_align,
        check_valid_multiplicity,
        check_valid_subunit_size,
    )

    # remove non-protein elements
    src_atom_array = src_atom_array[src_atom_array.chain_type == 6]

    # get entities and ids from the src atom array
    pn_unit_ent = src_atom_array.get_annotation("pn_unit_entity")
    pn_unit_id = src_atom_array.get_annotation("pn_unit_iid")
    unique_entities = np.unique(pn_unit_ent)
    nids_by_entity = {
        i: np.unique(pn_unit_id[pn_unit_ent == i]) for i in unique_entities
    }

    # get coordinates
    coords = src_atom_array.coord

    # ------------------------------------------------------------------
    # T3_LINEAR_OVERLAY — ASU-only bypass for T<N>EXACT motif conditioning.
    #
    # For exact-translation symmetry, the input PDB may intentionally
    # contain only ONE asymmetric unit (e.g. chain A only). The N exact
    # translational frames are supplied externally via
    # RFD3_T<N>EXACT_TRANSLATIONS, so in that case we must NOT require
    # the input atom array to already contain multiple symmetric copies.
    #
    # Generalized flag: RFD3_ALLOW_EXACT_ASU_MOTIF (checked first).
    # Older, per-order flags are still honored for backward compatibility:
    #   RFD3_ALLOW_T2EXACT_ASU_MOTIF, RFD3_ALLOW_T3EXACT_ASU_MOTIF
    # ------------------------------------------------------------------
    import os

    def _flag_is_true(name):
        return os.environ.get(name, "").lower() in {"1", "true", "yes", "y", "on"}

    multiplicity = min([len(i) for i in nids_by_entity.values()])

    allow_asu_exact = (
        _flag_is_true("RFD3_ALLOW_EXACT_ASU_MOTIF")
        or _flag_is_true("RFD3_ALLOW_T2EXACT_ASU_MOTIF")
        or _flag_is_true("RFD3_ALLOW_T3EXACT_ASU_MOTIF")
    )

    if multiplicity == 1 and allow_asu_exact:
        return input_frames

    check_valid_multiplicity(nids_by_entity)

    multiplicity = min([len(i) for i in nids_by_entity.values()])
    n_per_asu = {i: len(j) // multiplicity for i, j in nids_by_entity.items()}

    # check that the subunits in the input are of the same size
    check_valid_subunit_size(nids_by_entity, pn_unit_id)

    # align the largest set of entities
    natm_per_unique = {
        i: (pn_unit_id == nids_by_entity[i][0]).sum()
        for i in unique_entities
        if n_per_asu[i] == 1
    }
    reference_entity = max(natm_per_unique, key=natm_per_unique.get)

    # check that we have enough atoms to align
    check_min_atoms_to_align(natm_per_unique, reference_entity)

    # chains for the alignment (will generate complete set of frames)
    chains_to_consider = nids_by_entity[reference_entity]
    reference_molecule = nids_by_entity[reference_entity][0]

    # check that we are not exceeding the max number of transforms
    check_max_transforms(chains_to_consider)

    # align reference molecule to all others
    xforms = {
        i: _align(coords[pn_unit_id == i], coords[pn_unit_id == reference_molecule])
        for i in chains_to_consider
    }
    rmsds = {
        i: _rms(coords[pn_unit_id == i], coords[pn_unit_id == reference_molecule], *j)
        for i, j in xforms.items()
    }

    # check that there is not too big of a RMSD difference between subunits
    check_max_rmsds(rmsds)

    # check that the frames are valid rotation matrices
    Rs = [R for _, R, _ in xforms.values()]
    for R in Rs:
        assert is_valid_rotation_matrix(
            R
        ), f"Computed frame {R} is not a valid rotation matrix"
    computed_frames = [(R, np.array([0, 0, 0])) for R in Rs]

    # check that the computed frames match the input frames
    check_input_frames_match_symmetry_frames(
        computed_frames, input_frames, nids_by_entity
    )

    return computed_frames


def _align(X_fixed, X_moving):
    """
    Align two sets of coordinates using Kabsch algorithm.
    Arguments:
        X_fixed: fixed coordinates
        X_moving: moving coordinates
    Returns:
        u_X_moving: mean of the moving coordinates
        R: rotation matrix
        u_X_fixed: mean of the fixed coordinates
    """
    is_torch = isinstance(X_fixed, torch.Tensor)

    def _mean_along_dim(X, dim):
        if is_torch:
            return X.mean(dim=dim)
        else:
            return X.mean(axis=dim)

    assert X_fixed.shape == X_moving.shape

    if X_fixed.ndim == 2:
        X_fixed = X_fixed[None, ...]
        X_moving = X_moving[None, ...]
    B = X_fixed.shape[0]

    if is_torch:
        mask = (~torch.isnan(X_fixed) & ~torch.isnan(X_moving)).all(dim=-1).all(dim=0)
    else:
        mask = (~np.isnan(X_fixed) & ~np.isnan(X_moving)).all(axis=-1).all(axis=0)

    X_fixed = X_fixed[:, mask]
    X_moving = X_moving[:, mask]

    u_X_fixed = _mean_along_dim(X_fixed, dim=-2)
    u_X_moving = _mean_along_dim(X_moving, dim=-2)

    X_fixed_centered = X_fixed - u_X_fixed[..., None, :]
    X_moving_centered = X_moving - u_X_moving[..., None, :]

    if is_torch:
        C = torch.einsum("...ji,...jk->...ik", X_fixed_centered, X_moving_centered)
        U, S, V = torch.linalg.svd(C, full_matrices=False)
    else:
        C = np.einsum("...ji,...jk->...ik", X_fixed_centered, X_moving_centered)
        U, S, V = np.linalg.svd(C, full_matrices=False)

    R = U @ V
    if is_torch:
        F = torch.eye(3, 3, device=R.device).expand(B, 3, 3).clone()
        F[..., -1, -1] = torch.sign(torch.linalg.det(R))
    else:
        F = np.broadcast_to(np.eye(3, 3), (B, 3, 3)).copy()
        F[..., -1, -1] = np.sign(np.linalg.det(R))
    R = U @ F @ V

    if R.shape[0] == 1:
        return u_X_moving[0], R[0], u_X_fixed[0]

    return u_X_moving, R, u_X_fixed


def _rms(X_fixed, X_moving, t_pre, R, t_post):
    """
    Calculate the RMSD between two sets of coordinates.
    Arguments:
        X_fixed: fixed coordinates
        X_moving: moving coordinates
        t_pre: pre-rotation translation
        R: rotation matrix
        t_post: post-rotation translation
    Returns:
        rms: RMSD
    """
    mask = (~np.isnan(X_fixed) & ~np.isnan(X_moving)).all(axis=-1)
    X_fixed = X_fixed[mask]
    X_moving = X_moving[mask]

    X_moving_aln = np.einsum("ij,bj->bi", R, (X_moving - t_pre[None])) + t_post[None]
    rms = np.sqrt(np.sum(np.square(X_moving_aln - X_fixed)) / X_moving_aln.shape[-2])
    return rms


def is_valid_rotation_matrix(R):
    """
    check if a matrix is a valid rotation matrix.
    Arguments:
        R: rotation matrix
    Returns:
        bool: True if R is a valid rotation matrix, False otherwise
    """

    return np.allclose(R @ R.T, np.eye(3), atol=1e-6)


def get_cyclic_frames(order):
    """
    Get cyclic frames from a number of subunits.
    Arguments:
        order: number of subunits
    Returns:
        frames: list of rotation matrices
    """

    frames = []
    for i in range(order):
        angle = 2 * np.pi * i / order
        R = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        frames.append((R, np.array([0, 0, 0])))

    return frames


def get_dihedral_frames(order):
    """
    Get dihedral frames from a number of subunits.
    Arguments:
        order: number of subunits // 2 (since each dihedral has two frames)
    Returns:
        frames: list of rotation matrices
    """

    frames = []

    for i in range(order):
        angle = 2 * np.pi * i / order
        R = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )

        # 180 degree rotation in the xy-plane
        phi = angle + np.pi / order
        u = np.array([np.cos(phi), np.sin(phi), 0])
        flip = -np.eye(3) + 2 * np.outer(u, u)

        # add both frames for the dihedral
        frames.append((R, np.array([0, 0, 0])))
        frames.append((R @ flip, np.array([0, 0, 0])))

    return frames


def get_frames_from_file(file_path):
    raise NotImplementedError("Input defined symmetry not implemented")


###################################
# Kinematics
###################################


# fd - two routines that convert between:
#    a) a "virtual frame" consisting of three atoms; and
#    b) a translation and rotation
# uses Gram-Schmidt orthogonalziation, handles stacked/unstacked
# support np and torch inputs
def RTs_to_framecoords(Rs, ts, sig=1.0):
    if isinstance(Rs, np.ndarray):
        Rs = torch.from_numpy(Rs)
        ts = torch.from_numpy(ts)
    Ori = ts
    X = Ori + sig * Rs[..., 0, :] / (
        torch.norm(Rs[..., 0, :], dim=-1, keepdim=True) + 1e-6
    )
    Y = Ori + sig * Rs[..., 1, :] / (
        torch.norm(Rs[..., 1, :], dim=-1, keepdim=True) + 1e-6
    )
    return Ori, X, Y


# RTs_to_framecoords is used in loss and expects torch inputs
# (and must support backwards)
def framecoords_to_RTs(Ori, X, Y, eps=1e-6):
    R1 = X - Ori
    R1 = (R1 + torch.tensor([eps, 0, 0], device=R1.device)) / (
        torch.linalg.norm(R1, axis=-1, keepdims=True) + eps
    )

    Y_rel = Y - Ori
    proj = torch.sum(Y_rel * R1, axis=-1, keepdims=True) * R1
    R2 = Y_rel - proj
    R2 = (R2 + torch.tensor([0, eps, 0], device=R1.device)) / (
        torch.linalg.norm(R2, axis=-1, keepdims=True) + eps
    )

    R3 = torch.cross(R1, R2, dim=-1)

    # Stack into rotation matrix
    R = torch.stack([R1, R2, R3], axis=-2)  # shape (..., 3, 3)
    T = Ori

    return R, T


def pack_vector(v: np.ndarray) -> np.ndarray:
    """
    v: 1-D array of shape (3,) and arbitrary dtype
    returns: 1-element of shape 1
    """
    dt = np.dtype([("x", v.dtype, (3,))])
    a = np.zeros(1, dtype=dt)
    a["x"][0] = v
    return a


def unpack_vector(a: np.ndarray) -> np.ndarray:
    """
    a: stuctured array of shape (1,)
    returns: original vector
    """
    return a["x"]


def decompose_symmetry_frame(frame):
    R, T = frame
    Ori, X, Y = RTs_to_framecoords(R, T)
    Ori, X, Y = pack_vector(Ori.numpy()), pack_vector(X.numpy()), pack_vector(Y.numpy())
    return Ori, X, Y
