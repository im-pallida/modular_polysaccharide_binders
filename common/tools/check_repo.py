#!/usr/bin/env python3
"""
Would a fresh clone of this repo actually run the pipeline?

    check_repo.py                       # report
    check_repo.py --must <path> ...     # and fail if those are not committed

"git status" answers a different question. It says whether your working tree
matches your last commit; it says nothing about whether that commit contains the
files the pipeline needs, nothing about whether the commit reached the remote,
and -- the one that loses work -- nothing about a source file .gitignore is
quietly swallowing. A file that is ignored never appears as untracked, so the
tree looks clean and the file exists only on that machine.

That case is live here. The overlay under
01_rfd3_symmetry_generation/overlay/ is a vendored copy of RFD3's own package,
so it is the sort of directory people ignore wholesale -- and one file inside it
carries the rotation transpose, without which every symmetric design comes out
with its second copy inverted. Ignored, it survives exactly as long as the
machine does.

WHAT IT CHECKS

    source not tracked        it would be missing from a clone
    source IGNORED            worse: it will never show up as untracked either
    source tracked but dirty  the remote has an OLDER version than this machine
    HEAD not pushed           committed is not the same as uploaded
    data tracked              archives, structures, checkpoints and tables in
                              the repo -- not dangerous, but it is about to be
                              public and 16k production runs are large

Classification is by extension and by directory, and anything it cannot place
is listed as unclassified rather than assumed harmless. Read-only: it runs git
queries and nothing else.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

SOURCE_SUFFIXES = {
    ".py", ".sbatch", ".sh", ".env", ".example", ".txt", ".md",
    ".cfg", ".toml", ".yaml", ".yml", ".ini",
}
# Source that stays source inside a results or job_runs directory. Text files do
# not get this treatment: a .txt or .md under outputs/ is a run's own log or
# notice, and calling every TERMS_OF_USE.md a missing source file is noise.
EXECUTABLE_SUFFIXES = {".py", ".sbatch", ".sh"}
DATA_SUFFIXES = {
    ".tar", ".gz", ".tgz", ".zip", ".cif", ".pdb", ".ent", ".mmcif",
    ".ckpt", ".pt", ".pth", ".csv", ".tsv", ".jsonl", ".fa", ".fasta",
    ".npz", ".npy", ".pkl", ".png", ".jpg", ".pdf", ".log", ".pyc", ".bak",
}
# Produced by a run. Everything under one of these is data whatever it is called.
DATA_DIRS = {
    "outputs", "outputs_raw", "outputs_clean", "inputs", "inputs_prepared",
    "job_runs", "af3_runs", "logs", "tables", "sorted_raw", "sorted_clean",
    "results", "checkpoints", "raw", "scratch",
}
NEVER_WALK = {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache"}

SOURCE, DATA, EXPERIMENT, UNKNOWN = "source", "data", "experiment json", "unclassified"


def git(repo: Path, *args: str) -> Tuple[int, str]:
    """Run git here. Output is returned RAW, deliberately.

    It used to .strip(), which is right for rev-parse and wrong for
    `status --porcelain`, where the first two columns are a status code that may
    begin with a space: ' M common/archives.py' arrived as 'M common/...' and
    every dirty filename came out a character short -- reported as
    'ommon/archives.py', and quietly missed by --must, which looked the name up
    in a dict keyed by the mangled version. Callers that want it tidy say so.
    """
    done = subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True)
    return done.returncode, done.stdout


def classify(relative: Path) -> str:
    parts = set(relative.parts[:-1])
    suffix = relative.suffix.lower()
    if relative.name.endswith(".bak") or ".bak" in relative.name:
        return DATA
    # Something executable is source WHEREVER it sits. The data-directory rule
    # below is otherwise stronger than the extension rule, which made
    # results/save_monomer_passed.py "data" -- harmless as a mislabel, but it
    # means an untracked script under results/, job_runs/ or scripts/ nested in
    # one of those would not have been reported as missing from a clone. A tool
    # whose whole job is "do not miss a source file" cannot have that gap.
    if suffix in EXECUTABLE_SUFFIXES:
        return SOURCE
    if parts & DATA_DIRS:
        return DATA
    if suffix in DATA_SUFFIXES:
        return DATA
    if suffix == ".json":
        return EXPERIMENT
    if suffix in SOURCE_SUFFIXES or relative.name.startswith("."):
        return SOURCE
    return UNKNOWN


def walk(repo: Path) -> List[Path]:
    found = []
    for path in repo.rglob("*"):
        if any(part in NEVER_WALK for part in path.parts):
            continue
        if path.is_file():
            found.append(path.relative_to(repo))
    return sorted(found)


def tracked_set(repo: Path) -> Set[str]:
    code, out = git(repo, "ls-files")
    return set(out.strip().splitlines()) if code == 0 else set()


def ignored_set(repo: Path, candidates: Sequence[Path]) -> Set[str]:
    """Which of these git would ignore. Asked in one call, not one per file."""
    if not candidates:
        return set()
    done = subprocess.run(
        ["git", "-C", str(repo), "check-ignore", "--stdin"],
        input="\n".join(str(path) for path in candidates),
        capture_output=True, text=True,
    )
    return set(done.stdout.split("\n")) - {""}


def dirty_set(repo: Path) -> Dict[str, str]:
    """Tracked files whose committed version differs from what is on disk."""
    code, out = git(repo, "status", "--porcelain")
    if code != 0:
        return {}
    # splitlines on the RAW output: rstrip only, never lstrip.
    changed: Dict[str, str] = {}
    for line in out.rstrip("\n").splitlines():
        if len(line) < 4:
            continue
        state, name = line[:2], line[3:]
        if state == "??":
            continue
        if " -> " in name:                    # a rename
            name = name.split(" -> ", 1)[1]
        changed[name.strip('"')] = state.strip() or "M"
    return changed


def push_state(repo: Path) -> str:
    code, upstream = git(repo, "rev-parse", "--abbrev-ref",
                         "--symbolic-full-name", "@{u}")
    upstream = upstream.strip()
    if code != 0 or not upstream:
        return ("no upstream branch is configured, so nothing here has a remote "
                "to be behind -- `git push -u origin <branch>` sets one")
    code, counts = git(repo, "rev-list", "--left-right", "--count",
                       f"HEAD...{upstream}")
    if code != 0:
        return (f"upstream is {upstream}, but it could not be compared -- "
                f"run `git fetch` first")
    ahead, _, behind = counts.strip().partition("\t")
    ahead, behind = ahead.strip() or "0", behind.strip() or "0"
    if ahead == "0" and behind == "0":
        return f"level with {upstream}: everything committed is also pushed"
    parts = []
    if ahead != "0":
        parts.append(f"{ahead} commit(s) NOT PUSHED")
    if behind != "0":
        parts.append(f"{behind} commit(s) on {upstream} not here")
    return f"{upstream}: " + ", ".join(parts)


def grouped(paths: Sequence[str], limit: int = 12) -> List[str]:
    """By top two directory levels, so a vendored tree is one line not four hundred."""
    buckets: Dict[str, List[str]] = defaultdict(list)
    for name in sorted(paths):
        parts = Path(name).parts
        key = "/".join(parts[:2]) if len(parts) > 2 else str(Path(name).parent or ".")
        buckets[key].append(name)
    lines = []
    for key, names in sorted(buckets.items()):
        if len(names) > limit:
            lines.append(f"    {key}/  --  {len(names)} file(s), including:")
            lines.extend(f"        {name}" for name in names[:3])
            lines.append(f"        ... and {len(names) - 3} more")
        else:
            lines.extend(f"    {name}" for name in names)
    return lines


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=Path.cwd(),
                        help="the checkout (default: current directory)")
    parser.add_argument("--must", action="append", default=[], metavar="PATH",
                        help="fail unless this path is tracked and committed clean; "
                             "repeatable")
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    code, top = git(repo, "rev-parse", "--show-toplevel")
    if code != 0:
        print(f"ERROR: {repo} is not a git repository", file=sys.stderr)
        return 2
    repo = Path(top.strip())
    print(f"repository: {repo}")
    code, head = git(repo, "log", "-1", "--format=%h %s")
    print(f"HEAD      : {head.strip() or '<no commits yet>'}")
    print(f"remote    : {push_state(repo)}")

    files = walk(repo)
    tracked = tracked_set(repo)
    dirty = dirty_set(repo)
    untracked = [str(path) for path in files if str(path) not in tracked]
    ignored = ignored_set(repo, [Path(name) for name in untracked])

    by_class: Dict[str, Dict[str, List[str]]] = {
        name: {"tracked": [], "untracked": [], "ignored": []}
        for name in (SOURCE, DATA, EXPERIMENT, UNKNOWN)
    }
    for path in files:
        name = str(path)
        kind = classify(path)
        if name in tracked:
            by_class[kind]["tracked"].append(name)
        elif name in ignored:
            by_class[kind]["ignored"].append(name)
        else:
            by_class[kind]["untracked"].append(name)

    print("\n--- what is in this checkout ---")
    for kind in (SOURCE, EXPERIMENT, DATA, UNKNOWN):
        counts = by_class[kind]
        total = sum(len(v) for v in counts.values())
        if not total:
            continue
        print(f"  {kind:<16} {total:>5} file(s):  "
              f"{len(counts['tracked'])} tracked, "
              f"{len(counts['untracked'])} untracked, "
              f"{len(counts['ignored'])} ignored")

    problems = 0

    dirty_source = sorted(name for name in dirty
                          if classify(Path(name)) in (SOURCE, EXPERIMENT))
    if dirty_source:
        problems += 1
        print(f"\n!!! {len(dirty_source)} tracked source file(s) differ from the last "
              f"commit. The remote has the OLDER version:")
        print("\n".join(grouped(dirty_source)))

    if by_class[SOURCE]["ignored"]:
        problems += 1
        print(f"\n!!! {len(by_class[SOURCE]['ignored'])} source file(s) are IGNORED. "
              f"They exist only on this machine and will never show as untracked:")
        print("\n".join(grouped(by_class[SOURCE]["ignored"])))

    if by_class[SOURCE]["untracked"]:
        problems += 1
        print(f"\n!!! {len(by_class[SOURCE]['untracked'])} source file(s) are not "
              f"tracked -- missing from a fresh clone:")
        print("\n".join(grouped(by_class[SOURCE]["untracked"])))

    if by_class[UNKNOWN]["untracked"] or by_class[UNKNOWN]["ignored"]:
        loose = sorted(by_class[UNKNOWN]["untracked"] + by_class[UNKNOWN]["ignored"])
        print(f"\n--- {len(loose)} file(s) this tool could not classify; "
              f"decide for yourself ---")
        print("\n".join(grouped(loose)))

    if by_class[DATA]["tracked"]:
        print(f"\n--- {len(by_class[DATA]['tracked'])} data file(s) ARE tracked "
              f"(fine, but they go public with the repo) ---")
        print("\n".join(grouped(by_class[DATA]["tracked"])))

    for wanted in args.must:
        name = str(Path(wanted))
        if name not in tracked:
            problems += 1
            state = "ignored" if name in ignored else "not tracked"
            print(f"\n!!! --must {name}: {state}. This one was named as required.")
        elif name in dirty:
            problems += 1
            print(f"\n!!! --must {name}: tracked but modified since the last commit "
                  f"({dirty[name]}), so the remote has an older copy.")
        else:
            print(f"\nok  --must {name}: tracked and committed clean.")

    if problems:
        print(f"\n{problems} problem group(s) above. Nothing has been changed.")
        return 1
    print("\nEvery source file is tracked and committed. "
          "Check the remote line above for whether it is pushed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
