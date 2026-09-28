"""
Hand a stage's passed structures to the next stage.

Every stage does this identically:

    sorted_clean/<experiment>/passed/<group>.tar.gz     <- this stage wrote
    <next stage>/inputs/<experiment>/<group>.tar.gz     <- this writes

Same <group>.tar.gz filename, just relocated. rejected/ is never read: only
passed structures move on.

Incremental, with no transfer log. A protein counts as transferred when it is
already a member of the destination archive, so the destination is the single
record of what has moved and cannot drift from a separate file. That works
because regroup_and_archive is cumulative -- a group's passed archive holds
every structure that ever passed, not just the latest run's -- so source and
destination can be compared directly.

Destination archives are extended, never overwritten: the next stage needs
every passed protein ever produced.

A protein moves only with BOTH its metadata json and its structure file. An
incomplete pair is reported and left behind rather than half-copied.

A stage supplies only where its output goes; everything else is here.
"""
from __future__ import annotations

import argparse
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import job_paths as jp  # noqa: E402
from archives import (  # noqa: E402
    ArchiveError,
    group_members_by_protein,
    merge_members_into_archive,
)

PASSED = "passed"

# (stage root, experiment, group_key) -> the archive this group belongs in
DestinationFor = Callable[[Path, str, str], Path]

# member names -> {protein_id: [member names belonging to it]}. Stages differ in
# what one protein's files look like -- a json plus a structure for stages 01
# and 02, a folder of fastas for stage 03 -- so the rule is supplied rather than
# assumed, and everything else about the transfer stays shared.
MembersOf = Callable[[Sequence[str]], Dict[str, List[str]]]


def paired_members(member_names: Sequence[str]) -> Dict[str, List[str]]:
    """The default: a protein moves only with BOTH its json and its structure."""
    grouped: Dict[str, List[str]] = {}
    for protein_id, entry in group_members_by_protein(member_names).items():
        if "json" in entry and "structure" in entry:
            grouped[protein_id] = [entry["json"], entry["structure"]]
        else:
            grouped[protein_id] = []      # incomplete: reported, not moved
    return grouped


class TransferError(Exception):
    """A transfer could not be completed -- not a structure being skipped."""


def _log(*parts: object) -> None:
    print(*parts, flush=True)


@dataclass
class TransferResult:
    experiment: str
    group_key: str
    source: Path
    destination: Path
    added: int = 0
    total: int = 0


@dataclass
class TransferReport:
    transferred: int = 0
    already_present: int = 0
    incomplete: List[str] = field(default_factory=list)
    results: List[TransferResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """An incomplete pair means a passed structure cannot move on."""
        return not self.incomplete

    def summary(self) -> str:
        if not self.transferred:
            return f"transfer: nothing new ({self.already_present} already there)"
        return (f"transfer: {self.transferred} structure(s) moved on, "
                f"{self.already_present} already there, "
                f"{len(self.results)} archive(s) updated")


def passed_archives(stage: Path, experiment: Optional[str] = None) -> Dict[str, List[Path]]:
    """{experiment: [passed/<group>.tar.gz, ...]}.

    Defaults to every filtered experiment, matching the filters' own
    sweep-everything behaviour, so a batch an interrupted run left behind is
    picked up by the next launch whichever experiment that launch was for.
    """
    names = [experiment] if experiment else jp.sorted_experiments(stage)
    found: Dict[str, List[Path]] = {}
    for name in names:
        passed_dir = jp.sorted_clean_dir(stage, name, PASSED)
        if not passed_dir.is_dir():
            continue
        archives = sorted(passed_dir.glob("*.tar.gz"))
        if archives:
            found[name] = archives
    return found


def transferred_ids(destination: Path, members_of: MembersOf = paired_members) -> Set[str]:
    """The protein_ids already in a destination archive -- the record of what
    has moved. Reuses the shared grouping so the rule for what makes a
    protein's file pair is defined in exactly one place."""
    if not destination.is_file():
        return set()
    with tarfile.open(destination, "r:gz") as archive:
        return set(members_of(archive.getnames()))


def _read_new_members(
    source_path: Path, already: Set[str], members_of: MembersOf
) -> Tuple[List[Tuple[tarfile.TarInfo, bytes]], List[str], int]:
    """Members for proteins not yet at the destination, the ids whose file set
    is incomplete, and how many were already transferred."""
    members: List[Tuple[tarfile.TarInfo, bytes]] = []
    incomplete: List[str] = []

    with tarfile.open(source_path, "r:gz") as source:
        grouped = members_of(source.getnames())
        candidates = sorted(pid for pid in grouped if pid not in already)
        skipped = len(grouped) - len(candidates)

        for protein_id in candidates:
            names = grouped[protein_id]
            if not names:
                incomplete.append(protein_id)
                _log(f"  [incomplete] {protein_id}: file set incomplete in "
                     f"{source_path.name}; not transferred")
                continue
            for member_name in names:
                member_info = source.getmember(member_name)
                if member_info.isdir():
                    # Directory entries carry no data. An archive written by a
                    # stage has none, but one made by hand with `tar czf` does,
                    # and they must not look like unreadable files.
                    continue
                extracted = source.extractfile(member_info)
                if extracted is None:
                    raise TransferError(
                        f"could not read member {member_name!r} from {source_path}"
                    )
                # Normalise the './' that `tar czf ... -C dir .` writes, so the
                # destination holds plain names. Archive verification compares
                # normalised names, so carrying the prefix through would make a
                # correctly written archive fail its own check.
                if member_info.name.startswith("./"):
                    member_info.name = member_info.name[2:]
                members.append((member_info, extracted.read()))

    return members, incomplete, skipped


def run_transfer(
    stage: Path, destination_for: DestinationFor, label: str,
    experiment: Optional[str] = None, members_of: MembersOf = paired_members,
) -> TransferReport:
    """Move every passed structure not already at the destination."""
    report = TransferReport()
    by_experiment = passed_archives(stage, experiment)
    if not by_experiment:
        # Distinguish "the filter has not run" from "it ran and nothing passed".
        # regroup_and_archive only creates an outcome directory when something
        # lands in it, so a zero-yield experiment has rejected/ but no passed/,
        # and reporting that as "nothing under sorted_clean" sends people
        # looking for a bug that is not there.
        filtered = jp.sorted_experiments(stage)
        if filtered:
            _log(f"[transfer] no passed/ archive in any of {len(filtered)} filtered "
                 f"experiment(s) ({', '.join(filtered)}) -- nothing has passed the "
                 f"filter yet, so {label} has nothing to receive")
        else:
            _log(f"[transfer] nothing filtered yet under {jp.sorted_clean_root(stage)}")
        return report

    for experiment_name, sources in by_experiment.items():
        for source_path in sources:
            group_key = source_path.name[: -len(".tar.gz")]
            destination = destination_for(stage, experiment_name, group_key)
            if not report.results:
                _log(f"[transfer] {label} inputs -> {destination.parent.parent}")

            already = transferred_ids(destination, members_of)
            members, incomplete, skipped = _read_new_members(
                source_path, already, members_of)
            report.already_present += skipped
            report.incomplete.extend(incomplete)
            if not members:
                continue

            added, total = merge_members_into_archive(destination, members)
            structures = len(set(
                member_info.name.split("/")[0] if "/" in member_info.name
                else Path(member_info.name).stem
                for member_info, _ in members))
            report.transferred += structures
            report.results.append(
                TransferResult(experiment_name, group_key, source_path,
                               destination, added, total)
            )
            _log(f"[transfer] {experiment_name}/{group_key}: +{structures} structure(s) "
                 f"({added} new member(s), {total} total) -> {destination}")

    return report


def cli(destination_for: DestinationFor, label: str, default_stage: Path,
        doc: str, argv: Optional[List[str]] = None,
        members_of: MembersOf = paired_members) -> int:
    """The whole command-line side, so a stage's own file is just its
    destination plus a shebang."""
    parser = argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=Path, default=default_stage,
                        help=f"stage root directory (default: {default_stage})")
    parser.add_argument("--experiment", default=None,
                        help="only transfer this one experiment (default: all of them)")
    args = parser.parse_args(argv)
    try:
        report = run_transfer(args.stage.resolve(), destination_for, label,
                              args.experiment, members_of)
    except (OSError, ArchiveError, TransferError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    _log(f"[done] {report.summary()}")
    return 0 if report.ok else 1
