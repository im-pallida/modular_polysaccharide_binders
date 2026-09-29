#!/usr/bin/env python3
"""
What this machine can and cannot run, in one pass.

    python3 common/check_site.py

Run it once on a machine you have not run on before. It reports the mode, every
tool the pipeline needs, and for anything missing, the exact variable to set --
instead of the alternative, which is discovering the gaps one stage failure at
a time, with stage 08's missing pieces surfacing only after stages 01 to 07
have already run.

It only looks. Nothing here writes, submits or folds.

WHAT IT CHECKS

    the mode           cluster or workstation, and which of the two decided it
    the tools          common/site.py's settings, resolved the way the stages
                       will resolve them
    RFD3               by sourcing stage 01's rfd3.env, which owns RFD3_EXE and
                       CKPT and stays the owner
    committed config   the composition bias stage 07 will not run without
    python             the packages every stage imports

Exit status is 0 when everything the CURRENT mode needs is present. A setting
only the other mode uses is reported and not counted against you -- AF3_WORKDIR
is a cluster path and its absence on a workstation means nothing.
"""
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
# site_config, not site: the standard library owns that name, and a module
# beside it on sys.path would either shadow it or silently lose to it.
import site_config as site  # noqa: E402

PROJECT_ROOT = site.PROJECT_ROOT
RFD3_ENV = PROJECT_ROOT / "01_rfd3_symmetry_generation" / "scripts" / "env" / "rfd3.env"
BIAS_FILE = PROJECT_ROOT / "07_proteinmpnn_linker" / "scripts" / "jsonls" / "bias_AA.jsonl"
REQUIREMENTS = PROJECT_ROOT / "common" / "requirements.txt"
# import name -> (what to pip install, what needs it)
#
# The import name is not always the package name -- Bio installs as biopython --
# and reporting "pip install Bio" sends somebody to a different, abandoned
# package. The third column is here because a missing package should say which
# stage it costs you, not just that it is absent.
PACKAGES = {
    "gemmi": ("gemmi", "every stage that reads a structure"),
    "numpy": ("numpy", "scoring and geometry"),
    "Bio": ("biopython", "stage 04 scoring: the alignment before superposition"),
}

OK, MISSING, SKIP = "  ok  ", " MISS ", " n/a  "


def line(status: str, name: str, detail: str) -> None:
    print(f"[{status}] {name:<24} {detail}")


def check_mode() -> str:
    here, why = site.mode_and_why()
    print(f"project : {PROJECT_ROOT}")
    print(f"mode    : {here}   ({why})")
    if here == "cluster":
        print("          export PIPELINE_MODE=workstation to run in place instead")
    print()
    return here


def check_settings(here: str) -> List[str]:
    problems: List[str] = []
    for record in site.resolve_all(here):
        setting = record.setting
        stages = "stages " + ", ".join(setting.stages)
        if not setting.needed_here(here):
            line(SKIP, setting.name, f"{setting.what} -- {setting.modes[0]} only")
            continue
        if record.value:
            line(OK, setting.name, f"{record.value}  [{record.source}]")
        elif setting.required:
            line(MISSING, setting.name, f"{setting.what} ({stages})")
            # WHERE IT LOOKED. "MISS" on its own says the tool was not found and
            # leaves you to guess whether it looked anywhere sensible. Printing
            # the patterns turns that into a question you can answer: if yours
            # is in none of these places, say so in site.local.env.
            for pattern in setting.candidates[:4]:
                print(f"{' ' * 8}tried  {pattern}")
            problems.append(setting.name)
        else:
            line(SKIP, setting.name, f"{setting.what} -- optional")
    return problems


def suggested_value(setting) -> str:
    """The candidate worth offering as a starting point.

    Not simply the first. site.local.env expands $HOME and $PROJECT and nothing
    else, so offering $CONDA_PREFIX/bin/rfd3 hands someone a line that cannot
    work; and a candidate with a '*' in it is a search pattern, not a path
    anybody can type. So: the first candidate that is neither.
    """
    usable = [c for c in setting.candidates
              if "*" not in c
              and all(part.startswith(("$HOME", "$PROJECT")) or not part.startswith("$")
                      for part in [c.split("/")[0]])]
    return (usable or list(setting.candidates) or [""])[0]


def local_env_block(here: str, missing: Sequence[str]) -> List[str]:
    """The lines to paste into site.local.env, one per setting not found."""
    entries = []
    for record in site.resolve_all(here):
        setting = record.setting
        if setting.name not in missing or record.value:
            continue
        entries.append((f"{setting.name}={suggested_value(setting)}", setting.what))
    if not entries:
        return []
    width = max(len(assignment) for assignment, _ in entries)
    return [f"{assignment:<{width}}   # {what}" for assignment, what in entries]


def check_rfd3() -> List[str]:
    """RFD3_EXE and CKPT as a COMPUTE NODE gets them.

    site_config resolves these too, and they are reported above. This is not a
    duplicate: that pair is what Python works out, this pair is what the shell
    ends up with after sourcing rfd3.env -- which is the path a submitted job
    actually takes, and the one that used to come back empty because it looked
    only at $CONDA_PREFIX and PATH. Both are shown because both can fail on
    their own.
    """
    if not RFD3_ENV.is_file():
        line(MISSING, "rfd3.env", f"not at {RFD3_ENV}")
        return ["rfd3.env"]
    script = f'source "{RFD3_ENV}" >/dev/null 2>&1; echo "$RFD3_EXE"; echo "$CKPT"'
    try:
        done = subprocess.run(["bash", "-c", script], capture_output=True,
                              text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        line(MISSING, "rfd3.env", f"could not be sourced: {exc}")
        return ["rfd3.env"]
    values = (done.stdout.splitlines() + ["", ""])[:2]
    problems: List[str] = []
    for name, value in (("RFD3_EXE", values[0]), ("CKPT", values[1])):
        label = f"{name} via rfd3.env"
        path = Path(value) if value else None
        if path is None or not path.is_file():
            line(MISSING, label,
                 f"stages 01, 06 -- set it in {site.LOCAL_ENV.name}. This is "
                 f"the shell path a submitted job takes.")
            problems.append(name)
        else:
            line(OK, label, str(path))
    return problems


def check_committed() -> List[str]:
    if BIAS_FILE.is_file():
        line(OK, "bias_AA.jsonl", str(BIAS_FILE))
        return []
    line(MISSING, "bias_AA.jsonl", f"stage 07 will not start without {BIAS_FILE}")
    return ["bias_AA.jsonl"]


def active_environment() -> str:
    """The conda env or virtualenv this interpreter belongs to, if any."""
    conda = os.environ.get("CONDA_DEFAULT_ENV", "").strip()
    if conda:
        return conda
    venv = os.environ.get("VIRTUAL_ENV", "").strip()
    return Path(venv).name if venv else ""


def check_packages() -> List[str]:
    """The imports every stage needs -- IN THIS INTERPRETER.

    Which is the part worth saying out loud. This script reports on whichever
    python ran it, so a check run from the wrong environment reports the wrong
    machine: the packages are missing from the env you are standing in, not
    from the box. Same for RFD3_EXE, which rfd3.env looks for under
    $CONDA_PREFIX. Naming the environment turns three confusing failures into
    one obvious one.
    """
    problems: List[str] = []
    where = active_environment()
    line(OK, "python", f"{sys.version.split()[0]} at {sys.executable}"
                       + (f"   [env: {where}]" if where else ""))
    for name, (pip_name, needed_by) in PACKAGES.items():
        try:
            module = importlib.import_module(name)
        except ImportError:
            line(MISSING, name, f"pip install {pip_name}  ({needed_by})")
            problems.append(name)
        else:
            line(OK, module.__name__, getattr(module, "__version__", "present"))
    if problems and where:
        print(f"         ^ missing from the '{where}' environment. If the "
              f"pipeline normally runs somewhere else,")
        print(f"           activate that environment and run this again before "
              f"installing anything.")
    if problems:
        # The launchers do not simply fail here -- they re-exec. Saying which
        # interpreter they would land on turns "it is broken" into "it will
        # work, just not under the python you are standing in".
        import bootstrap
        found = bootstrap.find_interpreter()
        if not found:
            line(MISSING, "launcher interpreter",
                 "no interpreter on this machine can import "
                 + ", ".join(bootstrap.REQUIRED))
            return problems

        line(OK, "launcher interpreter", f"{found}  (the launchers re-exec into this)")
        # Each one ASKED OF THAT INTERPRETER, not assumed.
        #
        # This used to clear every missing package the moment a launcher
        # interpreter existed, on the reasoning that the launchers re-exec into
        # it. But bootstrap only requires gemmi and numpy of a candidate, so an
        # interpreter missing biopython is still chosen -- and this reported
        # "ready: everything stage 01 to 08 needs is present" right up until
        # stage 04 refused to start. A package the venv also lacks is still
        # missing, and has to be named.
        still: List[str] = []
        for name in problems:
            if bootstrap.usable(found, (name,)):
                continue
            _, needed_by = PACKAGES[name]
            line(MISSING, f"{name} in the venv",
                 f"{found} cannot import it either  ({needed_by})")
            still.append(name)
        if still:
            # One command for all of them, rather than one per stage failure.
            print(f"         ^ fix the lot:")
            print(f"           {Path(found).parent / 'pip'} install -r {REQUIREMENTS}")
        if not still:
            # Otherwise the report reads as a contradiction: three MISS lines
            # and then "ready". Both are true -- they are missing from the
            # python you typed, and present in the one the launchers re-exec
            # into -- but only if somebody says so.
            print(f"         ^ missing from the interpreter above, present in "
                  f"that one. Nothing to install.")
        problems = still
    return problems


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=site.MODES, default=None,
                        help="report as if on this kind of machine")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        here = args.mode or check_mode()
        if args.mode:
            print(f"project : {PROJECT_ROOT}")
            print(f"mode    : {here}   (forced by --mode)\n")
        problems = check_settings(here)
        print()
        problems += check_rfd3()
        problems += check_committed()
        print()
        problems += check_packages()
    except site.SiteError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print()
    if not problems:
        print(f"ready: everything stage 01 to 08 needs on a {here} is present")
        return 0
    print(f"NOT READY on a {here}: {len(problems)} missing -- "
          f"{', '.join(problems)}")
    print()

    # The lines to paste, not just the name of the file to paste them into.
    #
    # This used to end at "Set them in <path>", which tells someone setting up a
    # new account the one thing they already knew. What they do not know is the
    # spelling of each key and where the pipeline expects to find the tool, and
    # both are right here.
    block = local_env_block(here, problems)
    if not block:
        print(f"Set them in {site.LOCAL_ENV}, one KEY=value per line, or export "
              f"them before running.")
        return 1

    print(f"Put these in {site.LOCAL_ENV} -- it is per machine AND per user, it "
          f"is not tracked,")
    print(f"and it is read before discovery, so nothing else has to change:")
    print()
    if not site.LOCAL_ENV.is_file():
        print(f"    cp {site.LOCAL_ENV.name}.example {site.LOCAL_ENV.name}")
    for entry in block:
        print(f"    {entry}")
    print()
    print(f"$PROJECT is this checkout ({PROJECT_ROOT}) and $HOME is yours, both "
          f"expanded when read,")
    print(f"so the values above are a starting point -- edit each to where YOUR "
          f"copy actually lives.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
