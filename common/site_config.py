#!/usr/bin/env python3
"""
Where this machine keeps its tools, and whether it has a queue.

THE PROBLEM THIS SOLVES

The pipeline already runs from any directory -- every script resolves its own
location -- but it did not run under any USER. Three paths were written down
with a username in them, one of them pointing into somebody else's home, and
"am I on the cluster?" was answered eight separate times by asking whether
sbatch happened to be on PATH, with no way to say otherwise.

So: one place that answers both questions, and one file per machine for the
answers discovery cannot work out.

PRECEDENCE, highest first

    1. the environment the caller already exported
    2. common/site.local.env          per machine, per user, NOT tracked
    3. discovery                      the candidate lists below

Never the other way round. Someone who exports MPNN_ROOT means it, and a
config file that overrides a deliberate export is a config file that wastes an
afternoon.

BOTH LANGUAGES, ONE IMPLEMENTATION

Half the pipeline is bash -- the sbatch scripts and the AF3 runners -- so the
settings have to reach a shell too. Rather than write the discovery twice and
watch the two drift, common/site.env simply evaluates this file:

    eval "$(python3 common/site_config.py --export)"

so there is one owner. RFD3 is the exception: 01_.../scripts/env/rfd3.env
already resolves RFD3_EXE and CKPT this way and is already portable, so it
keeps its own settings and this file only reports them.

Usage:
    python3 common/site_config.py              # what this machine looks like
    python3 common/site_config.py --export     # shell assignments, for site.env
    python3 common/site_config.py --mode       # "cluster" or "workstation"
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ENV = Path(__file__).resolve().parent / "site.local.env"

MODES = ("cluster", "workstation")


class SiteError(Exception):
    """A required setting could not be resolved."""


@dataclass(frozen=True)
class Setting:
    """One machine-specific value, and where to look for it.

    `marker` is a file that must exist INSIDE a directory for it to count --
    otherwise an empty folder of the right name resolves happily and the
    failure lands three stages later.
    """
    name: str
    what: str
    kind: str                      # "dir" | "file" | "value"
    candidates: Tuple[str, ...]
    stages: Tuple[str, ...]
    marker: Optional[str] = None
    modes: Tuple[str, ...] = MODES
    required: bool = True
    default: str = ""

    def needed_here(self, mode: str) -> bool:
        return mode in self.modes


# Candidate templates. $PROJECT is this checkout; $HOME, $USER, $SCRATCH and
# anything else come from the environment and are skipped when unset. A '*'
# makes it a glob -- which is what carries "same machine, different user":
# /home/*/software/alphafold3 finds a shared install without naming whose home
# it sits in.
SETTINGS: Tuple[Setting, ...] = (
    Setting(
        "MPNN_ROOT", "vanilla ProteinMPNN checkout", "dir",
        ("$HOME/software/ProteinMPNN", "$HOME/ProteinMPNN",
         "$PROJECT/../software/ProteinMPNN", "/home/*/software/ProteinMPNN",
         "/opt/ProteinMPNN"),
        stages=("07",), marker="protein_mpnn_run.py",
    ),
    Setting(
        # protein_mpnn_run.py needs torch, exactly as LigandMPNN's run.py does,
        # and for the same reason it cannot be the pipeline's own interpreter on
        # a cluster: that one carries gemmi, numpy and biopython, not torch.
        #
        # The ligandmpnn venv is among the candidates on purpose. Both are torch,
        # and a second venv holding the same 2 GB of wheels earns nothing.
        "MPNN_PYTHON", "an interpreter that can import torch", "file",
        ("$PROJECT/../software/proteinmpnn_venv/bin/python",
         "$MPNN_ROOT/.venv/bin/python",
         "$PROJECT/../software/ligandmpnn_venv/bin/python",
         "$HOME/software/proteinmpnn_venv/bin/python",
         "$HOME/software/ligandmpnn_venv/bin/python"),
        stages=("07",), required=False,
    ),
    Setting(
        "LIGANDMPNN_ROOT", "LigandMPNN checkout", "dir",
        ("$HOME/software/LigandMPNN", "$HOME/LigandMPNN",
         "$PROJECT/../software/LigandMPNN", "/home/*/software/LigandMPNN",
         "/opt/LigandMPNN"),
        stages=("03",), marker="run.py",
    ),
    Setting(
        # LigandMPNN needs torch; the pipeline's own interpreter needs gemmi and
        # numpy. On a workstation one conda env holds both and sys.executable
        # was enough. On a cluster the environments are split on purpose, and
        # running run.py under the pipeline interpreter finds no torch at all.
        "LIGANDMPNN_PYTHON", "an interpreter that can import torch", "file",
        ("$PROJECT/../software/ligandmpnn_venv/bin/python",
         "$LIGANDMPNN_ROOT/.venv/bin/python",
         "$PROJECT/../software/rfd3/envs/*/bin/python",
         "$HOME/software/ligandmpnn_venv/bin/python"),
        stages=("03",), required=False,
    ),
    Setting(
        "LIGANDMPNN_CHECKPOINT", "LigandMPNN weights", "file",
        ("$LIGANDMPNN_ROOT/model_params/ligandmpnn_v_32_010_25.pt",
         "$HOME/.foundry/checkpoints/ligandmpnn_v_32_010_25.pt"),
        stages=("03",), required=False,
    ),
    Setting(
        "AF3_ROOT", "AlphaFold3 checkout", "dir",
        ("$HOME/software/alphafold3", "$HOME/alphafold3",
         "$PROJECT/../software/alphafold3", "/home/*/software/alphafold3",
         "/opt/alphafold3"),
        stages=("04", "08"), marker="run_alphafold.py",
        modes=("workstation",),
    ),
    Setting(
        "MODEL_DIR", "AlphaFold3 model parameters", "dir",
        ("$HOME/software/alphafold3_model_params",
         "$AF3_ROOT/model_params",
         "/home/*/software/alphafold3_model_params",
         "/opt/alphafold3_model_params"),
        stages=("04", "08"), modes=("workstation",),
    ),
    Setting(
        "AF3_CONDA_ENV", "conda environment AlphaFold3 lives in", "value",
        (), stages=("04", "08"), modes=("workstation",), default="alphafold3",
    ),
    Setting(
        "AF3_WORKDIR", "AlphaFold3 working tree on the cluster", "dir",
        # $PROJECT/.. comes first deliberately. On a cluster the checkout sits
        # in a project's scratch space -- /shared/scratch/<project>/<someone>/
        # -- and the tools sit beside it. Anchoring to the checkout finds them
        # whoever is logged in, which globbing $USER does not: the account
        # running the job is often not the account the directory is named for.
        ("$PROJECT/../software/alphafold", "$SCRATCH/software/alphafold",
         "$HOME/software/alphafold",
         "/shared/scratch/*/*/software/alphafold",
         "/scratch/*/*/software/alphafold"),
        stages=("04", "08"), modes=("cluster",),
    ),
    Setting(
        "DB_DIR", "sequence databases; unset means --norun_data_pipeline", "dir",
        (), stages=("04", "08"), required=False,
    ),
    # RFD3 used to be resolved only by scripts/env/rfd3.env, from $CONDA_PREFIX
    # or PATH. That works in an activated shell and not on a compute node,
    # where neither is set and every submitted job died with "RFD3_EXE and/or
    # CKPT not set". Discovery lives here now and rfd3.env consumes it, so the
    # answer is the same whoever asks and wherever they ask from.
    Setting(
        "RFD3_EXE", "the rfd3 executable", "file",
        ("$CONDA_PREFIX/bin/rfd3",
         "$PROJECT/../software/rfd3/envs/*/bin/rfd3",
         "$PROJECT/../software/*/bin/rfd3",
         "$HOME/software/rfd3/envs/*/bin/rfd3",
         "$HOME/.foundry/bin/rfd3"),
        stages=("01", "06"),
    ),
    Setting(
        "CKPT", "the RFD3 checkpoint", "file",
        ("$HOME/.foundry/checkpoints/rfd3_latest.ckpt",
         "$PROJECT/01_rfd3_symmetry_generation/checkpoints/rfd3_latest.ckpt",
         "$PROJECT/../software/rfd3/checkpoints/rfd3_latest.ckpt",
         "$PROJECT/../software/rfd3/*/rfd3_latest.ckpt",
         "$HOME/software/rfd3/checkpoints/rfd3_latest.ckpt"),
        stages=("01", "06"),
    ),
)


def expand(template: str) -> List[str]:
    """Candidate paths from one template, or nothing if it cannot be expanded.

    A template naming a variable this machine has not set expands to a literal
    '$SCRATCH/...', which is not a path and must not be offered as one -- so it
    is dropped rather than tested.
    """
    text = template.replace("$PROJECT", str(PROJECT_ROOT))
    text = os.path.expandvars(text)
    if "$" in text:
        return []
    # normpath, not resolve: it collapses the '..' in $PROJECT/../software so
    # the reported path is readable, without following symlinks. Cluster
    # scratch is often a symlink, and resolving it would print a path the
    # person does not recognise as their own.
    if any(character in text for character in "*?["):
        return sorted(os.path.normpath(match) for match in glob.glob(text))
    return [os.path.normpath(text)]


def acceptable(path: Path, setting: Setting) -> bool:
    if setting.kind == "dir":
        if not path.is_dir():
            return False
        return setting.marker is None or (path / setting.marker).is_file()
    if setting.kind == "file":
        return path.is_file()
    return True


def read_local(path: Path = LOCAL_ENV) -> Dict[str, str]:
    """KEY=value lines from the per-machine file, if there is one.

    Deliberately not a shell: no substitution, no command execution, no
    sourcing. A config file that can run commands is a config file that can
    surprise you, and every value here is a path.
    """
    found: Dict[str, str] = {}
    if not path.is_file():
        return found
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export "):].lstrip()
        if "=" not in stripped:
            raise SiteError(f"{path.name} line {number}: expected KEY=value, got {line!r}")
        key, _, value = stripped.partition("=")
        value = value.strip()
        if len(value) > 1 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        found[key.strip()] = os.path.expandvars(value.replace("$PROJECT", str(PROJECT_ROOT)))
    return found


def mode_and_why() -> Tuple[str, str]:
    """("cluster" | "workstation", how it was decided).

    PIPELINE_MODE decides when it is set, because sbatch on PATH is a good
    guess and not a fact: a workstation can have Slurm installed, and a compute
    node already inside a job has sbatch but must not submit more.

    It follows the same precedence as every other setting -- the environment,
    then site.local.env, then the guess -- because a mode that could only be
    set by exporting it would be the one setting you cannot write down.
    """
    asked = os.environ.get("PIPELINE_MODE", "").strip().lower()
    source = "PIPELINE_MODE in the environment"
    if not asked:
        asked = read_local().get("PIPELINE_MODE", "").strip().lower()
        source = f"PIPELINE_MODE in {LOCAL_ENV.name}"
    if asked in MODES:
        return asked, source
    if asked and asked != "auto":
        raise SiteError(
            f"PIPELINE_MODE={asked!r} is not one of {', '.join(MODES)} or 'auto'"
        )
    if shutil.which("sbatch") is not None:
        return "cluster", "sbatch is on PATH"
    return "workstation", "no sbatch on PATH"


def mode() -> str:
    return mode_and_why()[0]


@dataclass
class Resolved:
    setting: Setting
    value: str = ""
    source: str = "missing"        # "environment" | "site.local.env" | path | "default"

    @property
    def ok(self) -> bool:
        return bool(self.value) or not self.setting.required


def resolve_all(current_mode: Optional[str] = None,
                apply_to_environ: bool = False) -> List[Resolved]:
    """Every setting, in order, so later candidates can use earlier answers.

    LIGANDMPNN_CHECKPOINT's first candidate lives under LIGANDMPNN_ROOT, so the
    order here is load-bearing: each resolved value is put where the next one's
    expansion can see it.
    """
    here = current_mode or mode()
    local = read_local()
    seen: Dict[str, str] = {}
    out: List[Resolved] = []

    for setting in SETTINGS:
        record = Resolved(setting)
        exported = os.environ.get(setting.name, "").strip()
        if exported:
            record.value, record.source = exported, "environment"
        elif local.get(setting.name):
            record.value, record.source = local[setting.name], LOCAL_ENV.name
        elif setting.kind == "value":
            if setting.default:
                record.value, record.source = setting.default, "default"
        else:
            # Earlier answers are visible to later templates, without leaking
            # into the caller's environment unless they asked for that.
            restore = {key: os.environ.get(key) for key in seen}
            os.environ.update(seen)
            try:
                for template in setting.candidates:
                    match = next((path for path in expand(template)
                                  if acceptable(Path(path), setting)), None)
                    if match is not None:
                        record.value, record.source = match, "found"
                        break
            finally:
                for key, previous in restore.items():
                    if previous is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = previous

        if record.value:
            seen[setting.name] = record.value
            if apply_to_environ:
                os.environ.setdefault(setting.name, record.value)
        out.append(record)
    return out


def apply() -> str:
    """Fill in whatever this machine has not exported, and return the mode.

    Called once at the top of a launcher. Anything the caller set is left
    exactly as it was -- os.environ.setdefault, never assignment.
    """
    here = mode()
    resolve_all(here, apply_to_environ=True)
    os.environ.setdefault("PIPELINE_MODE", here)
    return here


def require(name: str) -> Path:
    """One setting, or a message naming the variable to set."""
    for record in resolve_all():
        if record.setting.name != name:
            continue
        if record.value:
            return Path(record.value)
        raise SiteError(
            f"{name} is not set and could not be found ({record.setting.what}).\n"
            f"  Set it in {LOCAL_ENV}, or export it before running."
        )
    raise SiteError(f"{name} is not a known setting")


def export_lines(here: Optional[str] = None) -> List[str]:
    """Shell assignments for site.env to evaluate."""
    current = here or mode()
    lines = [f"export PIPELINE_MODE={current}",
             f"export PIPELINE_ROOT={_quote(str(PROJECT_ROOT))}"]
    for record in resolve_all(current):
        if record.value and record.source != "environment":
            lines.append(f"export {record.setting.name}={_quote(record.value)}")
    return lines


def _quote(text: str) -> str:
    return "'" + text.replace("'", "'\\''") + "'"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--export", action="store_true",
                        help="print shell assignments instead of a report")
    parser.add_argument("--mode", action="store_true",
                        help="print just cluster or workstation")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        here = mode()
        if args.mode:
            print(here)
            return 0
        if args.export:
            print("\n".join(export_lines(here)))
            return 0
        print(f"project : {PROJECT_ROOT}")
        print(f"mode    : {here}")
        for record in resolve_all(here):
            mark = "  " if record.ok else "!!"
            print(f"{mark} {record.setting.name:<22} {record.value or '-'}  "
                  f"[{record.source}]")
    except SiteError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
