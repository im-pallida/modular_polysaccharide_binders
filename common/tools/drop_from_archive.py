#!/usr/bin/env python3
"""
Take one structure out of an archive, or move it to another one.

    drop_from_archive.py --id 001008 <archive.tar.gz> [more ...]
    drop_from_archive.py --id 001008 --apply <archive.tar.gz> [more ...]
    drop_from_archive.py --id 001008 --to <dir> --apply <archive.tar.gz>

Without --apply it only says what would go. That is the default on purpose.

WHY THIS EXISTS

`tar` cannot remove a member from a .tar.gz -- --delete refuses a compressed
archive -- so the usual answer is to unpack a group, delete one file, re-roll
it, and do that once per stage by hand. That is a lot of hand-typed destruction
to withdraw one structure, and a slip loses the group.

Withdrawing one is not hypothetical. 2cwr_1xchitin_dp13_001008 came out of RFD3
with its second copy rotated the wrong way about the right axis. Stage 02 passed
it -- that filter sorts on chainbreaks and non-loop fraction, which an inverted
copy satisfies perfectly -- and it would have spent a LigandMPNN run and an AF3
run before anyone looked at its geometry.

WHAT IT GUARANTEES

    nothing is written          without --apply
    with --to, the member is    in the new archive, verified, BEFORE it leaves
                                the old one
    every survivor is compared  by sha256 after the rewrite, not by name: the
                                point of the exercise is that the rest is
                                untouched
    the old archive is kept     as a .bak beside it, for you to delete
    an empty result refused     unless --allow-empty
    nothing matched             writes nothing and exits non-zero

REMOVING IT FROM ONE ARCHIVE IS NOT ENOUGH

Membership is what the pipeline reads, and it reads it in both directions:

    the stage 02 -> 03 transfer  asks the DESTINATION which protein_ids it
                                 already holds and sends everything else. Drop
                                 one from stage 03's inputs alone and the next
                                 stage 02 run puts it straight back, because an
                                 id that is missing looks new.
    stage 02's filter            skips what its results table already lists.
                                 That row is what stops the structure being
                                 sorted out of stage 01 a second time -- so
                                 LEAVE THE ROW. It records what happened.

So a structure leaves at the stage it must not reach AND at the passed/ archive
that feeds it. Stage 01's outputs_clean keeps it: that is the record of what
RFD3 actually produced, nothing reads it without going through the filter's
table first, and keeping it means this operation destroys no structure at all.

It calls archives._write_and_swap rather than rolling its own temp-file dance,
because that is the one implementation of read, write, verify, replace in this
pipeline and a second one would be a second thing to get wrong.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import archives  # noqa: E402

Member = Tuple[tarfile.TarInfo, bytes]

# Short enough to match by accident. "8" would match every member whose name
# ends _8, across every group in the archive.
MIN_ID_LENGTH = 3


def normalise(name: str) -> str:
    """Drop the './' that `tar czf ... -C dir .` writes."""
    return name[2:] if name.startswith("./") else name


def member_key(member_name: str) -> str:
    """The id a member is filed under.

    Members come in three shapes across the stages:

        001008.cif                      one file per structure
        2cwr_1xchitin_dp13_001008.cif   the same, named for its group
        001008/seq_1.fa                 stage 03's fastas, a directory each

    so the key is the stem of the FIRST path component -- which makes a nested
    fasta belong to its structure rather than to itself.
    """
    cleaned = normalise(member_name).strip("/")
    if not cleaned:
        return ""
    return Path(cleaned.split("/")[0]).stem


def matches(member_name: str, wanted: str) -> bool:
    """Does this member belong to `wanted`?

    Either the key is the id, or the id is the tail of a name that carries its
    group in front of it. The underscore is required, so 001008 does not match
    1001008.
    """
    key = member_key(member_name)
    return key == wanted or key.endswith("_" + wanted)


def digests(members: Sequence[Member]) -> Dict[str, str]:
    return {normalise(info.name): hashlib.sha256(payload).hexdigest()
            for info, payload in members}


def describe(info: tarfile.TarInfo) -> str:
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(info.mtime))
    return f"{normalise(info.name):<44} {info.size / 1024:9.1f} KiB   {when}"


def free_backup_path(archive_path: Path) -> Path:
    """`x.tar.gz.bak`, or .bak2, .bak3 ... so no earlier backup is overwritten."""
    candidate = archive_path.with_name(archive_path.name + ".bak")
    number = 2
    while candidate.exists():
        candidate = archive_path.with_name(f"{archive_path.name}.bak{number}")
        number += 1
    return candidate


@dataclass
class Plan:
    """What one archive would lose and keep."""
    archive: Path
    going: List[Member] = field(default_factory=list)
    staying: List[Member] = field(default_factory=list)

    @property
    def going_ids(self) -> List[str]:
        return sorted({member_key(info.name) for info, _ in self.going})

    @property
    def staying_ids(self) -> List[str]:
        return sorted({member_key(info.name) for info, _ in self.staying})


def plan_for(archive_path: Path, wanted: Sequence[str]) -> Plan:
    plan = Plan(archive_path)
    for member in archives.read_archive_members(archive_path):
        info, _ = member
        if any(matches(info.name, one) for one in wanted):
            plan.going.append(member)
        else:
            plan.staying.append(member)
    return plan


def move_to(destination: Path, going: Sequence[Member]) -> None:
    """Put the members in `destination` first, so nothing is deleted uncopied.

    A re-run after an interrupted move finds them already there. That is only
    allowed when they are byte-for-byte the same members; a same-named member
    with different contents is a different structure and stops everything.
    """
    already = {}
    if destination.is_file():
        already = digests(archives.read_archive_members(destination))
    wanted = digests(going)

    conflict = sorted(name for name, value in wanted.items()
                      if name in already and already[name] != value)
    if conflict:
        raise archives.ArchiveError(
            f"{destination}: already holds {conflict} with different contents. "
            f"Refusing to move onto them -- check which is the structure you want."
        )

    outstanding = [m for m in going if normalise(m[0].name) not in already]
    if not outstanding:
        print(f"  already at {destination} -- nothing to add")
    else:
        archives.merge_members_into_archive(destination, outstanding)

    arrived = digests(archives.read_archive_members(destination))
    missing = sorted(name for name, value in wanted.items()
                     if arrived.get(name) != value)
    if missing:
        raise archives.ArchiveError(
            f"{destination}: {missing} are not in it after the move. Nothing has "
            f"been removed from the source."
        )
    print(f"  moved to {destination}")


def apply_plan(plan: Plan, destination_dir: Optional[Path], allow_empty: bool) -> bool:
    if not plan.going:
        print(f"{plan.archive}: nothing matched -- left alone")
        return False
    if not plan.staying and not allow_empty:
        print(f"{plan.archive}: every member matched. Refusing to write an empty "
              f"archive -- pass --allow-empty if that is what you meant, or "
              f"delete the file yourself.")
        return False

    print(f"{plan.archive}")
    if destination_dir is not None:
        move_to(destination_dir / plan.archive.name, plan.going)

    before = digests(plan.staying)
    backup = free_backup_path(plan.archive)
    shutil.copy2(plan.archive, backup)

    archives._write_and_swap(
        plan.archive, plan.staying, lambda out: None, set(before)
    )

    # Names were checked by the swap. Contents are what matters: the promise is
    # that everything except the withdrawn structure came through unchanged.
    after = digests(archives.read_archive_members(plan.archive))
    changed = sorted(name for name, value in before.items() if after.get(name) != value)
    left_over = sorted(name for name in after if name not in before)
    if changed or left_over:
        raise archives.ArchiveError(
            f"{plan.archive}: rewrite is wrong -- changed {changed}, unexpected "
            f"{left_over}. The previous archive is at {backup}."
        )

    print(f"  removed {len(plan.going)} member(s) for {', '.join(plan.going_ids)}; "
          f"{len(plan.staying)} kept ({len(plan.staying_ids)} structure(s))")
    print(f"  previous archive kept at {backup}")
    return True


def report(plan: Plan) -> None:
    print(f"{plan.archive}")
    if not plan.going:
        print(f"  nothing matched. It holds: "
              f"{', '.join(plan.staying_ids) or '<no members>'}")
        return
    for info, _ in sorted(plan.going, key=lambda m: m[0].name):
        print(f"  going    {describe(info)}")
    kept = plan.staying_ids
    preview = ", ".join(kept[:8]) + (f" (+{len(kept) - 8} more)" if len(kept) > 8 else "")
    print(f"  staying  {len(plan.staying)} member(s), "
          f"{len(kept)} structure(s): {preview or '<none>'}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("archives", nargs="+", type=Path,
                        help="the .tar.gz files to take it out of")
    parser.add_argument("--id", dest="ids", action="append", required=True,
                        metavar="STRUCTURE_ID",
                        help="structure id to remove; repeatable")
    parser.add_argument("--to", type=Path, default=None, metavar="DIR",
                        help="move the members into DIR/<same archive name> "
                             "before removing them, instead of only removing")
    parser.add_argument("--apply", action="store_true",
                        help="actually write. Without it, nothing changes.")
    parser.add_argument("--allow-empty", action="store_true",
                        help="permit an archive to end up with no members")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    bad = [one for one in args.ids if len(one) < MIN_ID_LENGTH or "/" in one]
    if bad:
        print(f"ERROR: {bad} is not a usable structure id -- at least "
              f"{MIN_ID_LENGTH} characters and no '/'. A short id matches by "
              f"accident, and this tool deletes what it matches.", file=sys.stderr)
        return 2

    missing = [path for path in args.archives if not path.is_file()]
    if missing:
        for path in missing:
            print(f"ERROR: {path} is not a file", file=sys.stderr)
        return 2

    if args.to is not None and not args.to.is_dir():
        print(f"ERROR: --to {args.to} is not a directory", file=sys.stderr)
        return 2

    plans = []
    for path in args.archives:
        try:
            plans.append(plan_for(path, args.ids))
        except (OSError, tarfile.TarError, archives.ArchiveError) as exc:
            print(f"ERROR: {path}: {exc}", file=sys.stderr)
            return 1

    if not any(plan.going for plan in plans):
        print(f"nothing in {len(plans)} archive(s) matches "
              f"{', '.join(args.ids)} -- nothing to do")
        return 1

    if not args.apply:
        for plan in plans:
            report(plan)
        print("\nthis was a dry run. Add --apply to write, and the archives are "
              "each backed up before they are rewritten.")
        return 0

    touched = 0
    for plan in plans:
        try:
            if apply_plan(plan, args.to, args.allow_empty):
                touched += 1
        except (OSError, tarfile.TarError, archives.ArchiveError) as exc:
            print(f"ERROR: {plan.archive}: {exc}", file=sys.stderr)
            print("Stopping here. Archives already rewritten are listed above, "
                  "each with its backup.", file=sys.stderr)
            return 1

    print(f"\n{touched} archive(s) rewritten.")
    return 0 if touched else 1


if __name__ == "__main__":
    raise SystemExit(main())
