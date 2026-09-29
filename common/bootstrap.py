#!/usr/bin/env python3
"""
Start under an interpreter that can actually run the pipeline.

THE PROBLEM

On the cluster there is no conda. The pipeline's dependencies live in a plain
venv beside the checkout, and the only thing that knew where was a line in
somebody's shell history:

    export RFD3_PIPELINE_PYTHON=$SW/pipeline_venv/bin/python
    "$RFD3_PIPELINE_PYTHON" scripts/stage_01_launcher.py ...

Typed correctly it works; typed as `python stage_01_launcher.py` it dies on
`import gemmi` before a single line of the launcher runs. The jobs were already
protected -- run_cluster.sbatch and friends read RFD3_PIPELINE_PYTHON and
refuse to start under an interpreter that cannot import gemmi -- but the
launcher you type yourself was not.

WHAT THIS DOES

Called at the top of a launcher, BEFORE the imports that need gemmi: if this
interpreter can import them, it returns and nothing happens. If it cannot, it
finds one that can and re-execs the same script, same arguments, under that.
It says so on stderr; a program that silently restarts itself under a different
interpreter is a program that wastes somebody's afternoon.

RFD3_PIPELINE_PYTHON is honoured first, because the sbatch scripts already use
that name and two names for one thing is one too many.

GUARDS

    a marker in the environment      so a re-exec can never re-exec again,
                                     whatever goes wrong
    every candidate is TESTED        by running `import gemmi, numpy` in it,
                                     not by existing at the right path
    nothing found -> a clear error   naming what was tried and what to set

Standard library only, on purpose: this runs before the dependencies exist.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

# What a launcher needs to START, which is not everything the pipeline needs.
# biopython is deliberately absent: only stage 04's scoring imports it, and
# requiring it here would stop stages 01 to 03 running on a machine that has no
# reason to need it. The complete set is check_site.PACKAGES, which reports the
# gap instead of refusing -- and which now asks the re-exec target itself rather
# than assuming that finding one makes every missing package somebody else's
# problem.
REQUIRED = ("gemmi", "numpy")
GUARD = "PIPELINE_BOOTSTRAP_REEXEC"
OVERRIDE = ("RFD3_PIPELINE_PYTHON", "PIPELINE_PYTHON")

# Anchored to the checkout first, for the same reason AF3_WORKDIR is: on a
# cluster the account running the job is often not the account the directory is
# named for, so $USER finds nothing while $PROJECT/.. finds everything.
CANDIDATES = (
    "$PROJECT/../software/pipeline_venv/bin/python",
    "$PROJECT/../software/pipeline_venv/bin/python3",
    "$PROJECT/.venv/bin/python",
    "$PROJECT/../pipeline_venv/bin/python",
    "$HOME/software/pipeline_venv/bin/python",
    "/shared/scratch/*/*/software/pipeline_venv/bin/python",
    "/scratch/*/*/software/pipeline_venv/bin/python",
)


class BootstrapError(Exception):
    """No interpreter on this machine can run the pipeline."""


def _expand(template: str) -> List[str]:
    """Candidate paths, reusing site_config's expansion so the two agree."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import site_config  # noqa: E402  (stdlib-only, safe to import here)
    return site_config.expand(template)


def importable_here(packages: Sequence[str] = REQUIRED) -> bool:
    """Can THIS interpreter import them? Asked in-process, not by subprocess."""
    import importlib.util
    for name in packages:
        try:
            if importlib.util.find_spec(name) is None:
                return False
        except (ImportError, ValueError):
            return False
    return True


def usable(python: str, packages: Sequence[str] = REQUIRED) -> bool:
    """Can that interpreter import them? Proven by running it.

    A path that exists proves nothing -- a venv can be half-built, or built for
    a python that no longer exists on the node.
    """
    if not (os.path.isfile(python) and os.access(python, os.X_OK)):
        return False
    try:
        done = subprocess.run(
            [python, "-c", "import " + ", ".join(packages)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def find_interpreter(packages: Sequence[str] = REQUIRED) -> Optional[str]:
    """The first interpreter that can run the pipeline, or None."""
    for name in OVERRIDE:
        value = os.environ.get(name, "").strip()
        if value and usable(value, packages):
            return value
    for template in CANDIDATES:
        for path in _expand(template):
            if usable(path, packages):
                return path
    return None


def ensure(packages: Sequence[str] = REQUIRED) -> None:
    """Return under a usable interpreter, or re-exec into one, or explain.

    Call it at the top of a launcher, before importing anything that needs the
    dependencies. It is a no-op on a machine that is already set up, which is
    every workstation run.
    """
    if importable_here(packages):
        # Nothing to re-exec, but the submitted jobs still need to be TOLD which
        # interpreter this is. The .sbatch wrappers read RFD3_PIPELINE_PYTHON
        # and otherwise fall back to whatever python3 the compute node has --
        # the system one, without gemmi. Recording it here closes the backwards
        # gap where starting the launcher under the WRONG python worked (the
        # re-exec exported the name) and starting it under the right one did
        # not. setdefault, so an explicit export still wins.
        os.environ.setdefault("RFD3_PIPELINE_PYTHON", sys.executable)
        return

    missing = ", ".join(packages)
    if os.environ.get(GUARD):
        raise BootstrapError(
            f"re-executed under {sys.executable} and still cannot import "
            f"{missing}. That interpreter reported it could, so something "
            f"changed underneath it -- check the venv is intact."
        )

    python = find_interpreter(packages)
    if python is None:
        tried = "\n".join(f"      {template}" for template in CANDIDATES)
        raise BootstrapError(
            f"this interpreter ({sys.executable}) cannot import {missing}, and "
            f"no other one on this machine could either.\n"
            f"  Set RFD3_PIPELINE_PYTHON to an interpreter that can, or create "
            f"the venv:\n"
            f"      python3 -m venv <somewhere>/pipeline_venv\n"
            f"      <somewhere>/pipeline_venv/bin/pip install -r "
            f"{Path(__file__).resolve().parent / 'requirements.txt'}\n"
            f"  Looked for, in order, RFD3_PIPELINE_PYTHON, PIPELINE_PYTHON, "
            f"then:\n{tried}"
        )

    print(f"[bootstrap] {Path(sys.executable).name} cannot import {missing}; "
          f"re-running under {python}", file=sys.stderr, flush=True)
    environment = dict(os.environ, **{GUARD: "1"})
    # Pass the interpreter on so the submitted jobs use the same one; their
    # sbatch scripts already read this name.
    environment.setdefault("RFD3_PIPELINE_PYTHON", python)
    script = os.path.abspath(sys.argv[0])
    os.execve(python, [python, script] + sys.argv[1:], environment)


def main() -> int:
    """`python3 common/bootstrap.py` reports rather than re-execs."""
    here = importable_here()
    print(f"this interpreter : {sys.executable}")
    print(f"  can import {', '.join(REQUIRED)}: {'yes' if here else 'no'}")
    if here:
        return 0
    found = find_interpreter()
    print(f"  usable alternative: {found or 'none found'}")
    return 0 if found else 1


if __name__ == "__main__":
    raise SystemExit(main())
