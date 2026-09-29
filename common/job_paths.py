"""
The funcs which help to keep the consistency for the paths search.
"""
from __future__ import annotations
 
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
 
RUN_LIST_SUFFIX = "_run.tsv"
COUNTER_FILENAME = "global_seq_counter.txt"
 
# A run-list row: the json filename relative to json/<experiment>/, and the
# global sequence number assigned to that one structure.
RunRow = Tuple[str, int]
 
 
# Step 1. Naming.
 
 
def group_key_from_json_rel(json_rel: str) -> str:
    """'small.json' -> 'small'. json/<experiment>/ has no subfolders, so this
    is just the filename with the extension stripped."""
    return json_rel.removesuffix(".json")
 
 
# job_name() zero-pads to this width and group_key_from_job_name() undoes it.
# They must agree: if they ever drift, every structure is filed into the
# wrong archive without anything raising.
JOB_SEQ_DIGITS = 6


def job_name(group_key: str, global_seq: int) -> str:
    """The stem shared by a structure's .cif, its .json, and its archive
    members: 'small' + 42 -> 'small_000042'."""
    return f"{group_key}_{global_seq:0{JOB_SEQ_DIGITS}d}"
 
 
#Step 2. Inputs.
 
 
def json_root(stage: Path) -> Path:
    return stage / "json"
 
 
def experiment_json_dir(stage: Path, experiment: str) -> Path:
    return json_root(stage) / experiment
 
 
def json_path(stage: Path, experiment: str, json_rel: str) -> Path:
    return experiment_json_dir(stage, experiment) / json_rel


def design_entries(data: object) -> "dict[str, dict]":
    """The design configs inside an RFD3 input json.
    Raises ValueError if the file is not shaped like either."""
    if not isinstance(data, dict):
        raise ValueError(f"top level must be an object, got {type(data).__name__}")
    if "input" in data or "contig" in data:
        return {"": data}
    if not data:
        raise ValueError("file contains no design entries")
    bad = [name for name, cfg in data.items() if not isinstance(cfg, dict)]
    if bad:
        raise ValueError(
            f"expected each top-level key to name a design config; "
            f"these are not objects: {bad}"
        )
    return dict(data)
 
 
REFERENCE_DIRNAME = "00_reference_structures"


def resolve_seed_path(stage: Path, raw_input: str, json_path_: Path) -> Optional[Path]:
    """Locate the seed structure a json's "input" field points at.

    An absolute path is honoured when it exists, and a relative one is resolved
    against the stage and then the json's own folder -- both as before.

    What is new is the last resort. Design jsons are configuration and are
    committed, but their "input" was written on whichever machine created them,
    so a json made on the cluster carries /scratch/<project>/<someone>/... and
    is useless on a workstation, and the reverse. The seed itself is sitting in
    this checkout the whole time, under 00_reference_structures, so the file is
    looked for there by name before giving up.

    That makes one committed json serve both machines, and it is why a cluster
    whose scratch is reachable as both /scratch and /shared/scratch stops
    mattering.
    """
    candidate = Path(raw_input)
    project = stage.parent
    if candidate.is_absolute():
        if candidate.is_file():
            return candidate
        # Written by another checkout of this same project. Rebuild it onto
        # this one: find the first component of the stored path that names a
        # directory here -- '06_rfd3_linker_generation', say -- and keep
        # everything from there on.
        #
        #   stored  /home/karina/1cbh_clear/06_.../inputs_prepared/e/g/x.pdb
        #   here    /shared/scratch/SCWF00146/karina/1cbh_clear
        #   found   <here>/06_.../inputs_prepared/e/g/x.pdb
        #
        # Left to right, so the longest tail wins and the most specific match
        # is the one taken. This is what lets a json written on the workstation
        # run on the cluster without being regenerated -- and stage 06's linker
        # lengths are set by hand, so regenerating is not a free operation.
        rebuilt = _reanchor(project, candidate)
        if rebuilt is not None:
            return rebuilt
    else:
        # project first: a path stored relative to the checkout root, which is
        # what a stage writing into ANOTHER stage's tree has to use.
        for base in (project, stage, json_path_.parent):
            resolved = (base / candidate).resolve()
            if resolved.is_file():
                return resolved

    reference = project / REFERENCE_DIRNAME
    if candidate.name and reference.is_dir():
        for match in sorted(reference.rglob(candidate.name)):
            if match.is_file():
                return match
    return None


def _reanchor(project: Path, candidate: Path) -> Optional[Path]:
    """An absolute path from another checkout, rebuilt onto this project."""
    parts = candidate.parts
    for index in range(len(parts) - 1):
        if not (project / parts[index]).is_dir():
            continue
        rebuilt = project.joinpath(*parts[index:])
        if rebuilt.is_file():
            return rebuilt
    return None
 
 
# Step 3. Outputs.
 
 
def raw_root(stage: Path) -> Path:
    """Where a generating stage's untidied output lands, before archiving."""
    return stage / "outputs_raw"


def raw_dir(stage: Path, experiment: str, group_key: str) -> Path:
    return raw_root(stage) / experiment / group_key
 
 
def raw_cif_path(stage: Path, experiment: str, group_key: str, name: str) -> Path:
    return raw_dir(stage, experiment, group_key) / f"{name}.cif"
 
 
def raw_json_path(stage: Path, experiment: str, group_key: str, name: str) -> Path:
    return raw_dir(stage, experiment, group_key) / f"{name}.json"
 
 
def tmp_work_dir(stage: Path, experiment: str, group_key: str, name: str) -> Path:
    return stage / "outputs_raw" / ".tmp" / experiment / group_key / name
 
 
def clean_dir(stage: Path, experiment: str) -> Path:
    return stage / "outputs_clean" / experiment
 
 
def clean_archive_path(stage: Path, experiment: str, group_key: str) -> Path:
    return clean_dir(stage, experiment) / f"{group_key}.tar.gz"
 
 
def expected_raw_cif(stage: Path, experiment: str, json_rel: str, global_seq: int) -> Path:
    """Where a given run-list row's structure will land. Used to check whether
    a job produced its output without re-deriving the naming rules."""
    group_key = group_key_from_json_rel(json_rel)
    return raw_cif_path(stage, experiment, group_key, job_name(group_key, global_seq))
 
 
# Step 4. Run list.
 
 
def run_list_path(stage: Path, experiment: str) -> Path:
    return stage / "job_runs" / f"{experiment}{RUN_LIST_SUFFIX}"
 
 
def counter_path(stage: Path) -> Path:
    return stage / "job_runs" / COUNTER_FILENAME
 
 
def write_run_list(path: Path, rows: Sequence[RunRow]) -> None:
    """One '<json_filename>\\t<global_seq>' row per structure to generate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for json_rel, global_seq in rows:
            handle.write(f"{json_rel}\t{global_seq}\n")
 
 
def read_run_list(path: Path) -> List[RunRow]:
    """Parse a run list, reporting the offending line number on bad input.
 
    Blank lines and '#' comments are skipped so a run list stays hand-editable.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"could not read run list {path}: {exc}") from exc
 
    rows: List[RunRow] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split("\t")
        if len(fields) != 2:
            raise ValueError(
                f"{path}:{lineno}: expected '<json_filename><TAB><global_seq>', "
                f"got {line!r}"
            )
        json_rel, raw_seq = fields[0].strip(), fields[1].strip()
        if not raw_seq.isdecimal():
            raise ValueError(f"{path}:{lineno}: global_seq must be an integer, got {raw_seq!r}")
        rows.append((json_rel, int(raw_seq)))
    return rows
 
 
def group_rows(rows: Iterable[RunRow]) -> "dict[str, List[RunRow]]":
    """Run-list rows bucketed by group_key, preserving order within a group.
    Archiving works one group at a time because each group has its own tarball."""
    grouped: "dict[str, List[RunRow]]" = {}
    for json_rel, global_seq in rows:
        grouped.setdefault(group_key_from_json_rel(json_rel), []).append((json_rel, global_seq))
    return grouped


# Step 5. Filtering: sorted outputs and per-experiment result tables.

SORTED_RAW_DIRNAME = "sorted_raw"
SORTED_CLEAN_DIRNAME = "sorted_clean"
TABLES_DIRNAME = "tables"

# The two buckets the geometry filter sorts structures into. "rejected"
# matches the REJECTED status written in the results table, and only
# "passed" is ever read by the stage-02 transfer.
OUTCOMES = ("passed", "rejected")

def group_key_from_job_name(name: str) -> str:
    """'small_000042' -> 'small'. The inverse of job_name().

    Splits on the *last* underscore, so a group key containing underscores
    survives: 'chitin_trial_000001' -> 'chitin_trial'.

    Raises ValueError rather than guessing. A name this cannot parse is a
    structure from a different naming scheme (the 16k production ids looked
    like 'hsl014024'), and inventing a group for it would file the structure
    into the wrong archive silently.
    """
    group_key, separator, sequence = name.rpartition("_")
    if not separator or not group_key:
        raise ValueError(
            f"{name!r} is not '<group_key>_<{JOB_SEQ_DIGITS} digits>': no sequence suffix"
        )
    if not sequence.isdecimal() or len(sequence) < JOB_SEQ_DIGITS:
        raise ValueError(
            f"{name!r} is not '<group_key>_<{JOB_SEQ_DIGITS} digits>': "
            f"trailing {sequence!r} is not a {JOB_SEQ_DIGITS}-digit number"
        )
    return group_key


def clean_root(stage: Path) -> Path:
    return stage / "outputs_clean"


def clean_experiments(stage: Path) -> List[str]:
    """Every experiment that has an outputs_clean/ folder, oldest name first."""
    root = clean_root(stage)
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def sorted_scratch_dir(stage: Path, experiment: str) -> Path:
    """Where members are extracted before their outcome is known."""
    return stage / SORTED_RAW_DIRNAME / experiment / "_scratch"


def sorted_raw_dir(stage: Path, experiment: str, outcome: str) -> Path:
    """Loose routed files, before they are folded into a tarball."""
    return stage / SORTED_RAW_DIRNAME / experiment / outcome


def sorted_raw_group_dir(stage: Path, experiment: str, outcome: str, group_key: str) -> Path:
    """Loose routed files, kept under the group they came from.

    The group a structure belongs to is the archive it arrived in, which the
    caller already knows. Putting it in the path means archiving never has to
    recover it by parsing a protein id -- names from older runs (hsx015039)
    do not carry one, and deriving it there crashed the whole regrouping step.
    """
    return sorted_raw_dir(stage, experiment, outcome) / group_key


def sorted_clean_dir(stage: Path, experiment: str, outcome: str) -> Path:
    return stage / SORTED_CLEAN_DIRNAME / experiment / outcome


def sorted_archive_path(stage: Path, experiment: str, outcome: str, group_key: str) -> Path:
    """clean_archive_path() with an outcome level inserted:

        outputs_clean/<experiment>/<group>.tar.gz            <- diffusion writes
        sorted_clean/<experiment>/passed/<group>.tar.gz      <- the filter writes
        sorted_clean/<experiment>/rejected/<group>.tar.gz
    """
    return sorted_clean_dir(stage, experiment, outcome) / f"{group_key}.tar.gz"


def stage_label(stage: Path) -> str:
    """'01' from '01_rfd3_symmetry_generation', '02' from '02_geometry_filtering'.

    Taken from the directory rather than hardcoded so one copy of these helpers
    serves every stage. A directory that does not start with digits falls back
    to its own name, which is wrong-looking rather than silently wrong.
    """
    name = stage.name
    digits = ""
    for character in name:
        if not character.isdigit():
            break
        digits += character
    return digits or name


def results_table_path(stage: Path, experiment: str) -> Path:
    return stage / TABLES_DIRNAME / f"stage_{stage_label(stage)}_results_{experiment}.csv"


def passed_table_path(stage: Path, experiment: str) -> Path:
    """Stage 08 only: the round's deliverable, one row per construct that
    passed -- which construct, its unit sequence, and the linker holding the
    two copies of that unit together. The results table beside it keeps every
    measurement and every rejection; this one is the answer."""
    return stage / TABLES_DIRNAME / f"stage_{stage_label(stage)}_passed_{experiment}.csv"


def legacy_results_table_path(stage: Path) -> Path:
    """The single shared table the 16k production run wrote, before results
    were split per experiment. Read as a fallback so those structures are
    still recognised as already filtered; never written to."""
    return stage / TABLES_DIRNAME / "stage_01_results.csv"


# Step 6. Handing passed structures to stage 02.

STAGE02_DIRNAME = "02_geometry_filtering"
STAGE02_INPUTS_DIRNAME = "inputs"


def project_root(stage: Path) -> Path:
    """The folder holding all the numbered stage directories."""
    return stage.parent


def stage02_root(stage: Path) -> Path:
    return project_root(stage) / STAGE02_DIRNAME


def stage02_inputs_dir(stage: Path, experiment: str) -> Path:
    return stage02_root(stage) / STAGE02_INPUTS_DIRNAME / experiment


def stage02_archive_path(stage: Path, experiment: str, group_key: str) -> Path:
    """Where a group's passed structures land for stage 02. Deliberately the
    same <group>.tar.gz filename as the source, just relocated:

        sorted_clean/<experiment>/passed/<group>.tar.gz          <- stage 01 writes
        ../02_geometry_filtering/inputs/<experiment>/<group>.tar.gz
    """
    return stage02_inputs_dir(stage, experiment) / f"{group_key}.tar.gz"


def sorted_clean_root(stage: Path) -> Path:
    return stage / SORTED_CLEAN_DIRNAME


def sorted_experiments(stage: Path) -> List[str]:
    """Every experiment that has been through the filter."""
    root = sorted_clean_root(stage)
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


# Step 7. Stage 02: geometry filtering.
#
# The same shapes as stage 01, one stage along. A stage's own root is always
# passed in, so these work whichever stage directory calls them.

STAGE01_DIRNAME = "01_rfd3_symmetry_generation"
STAGE03_DIRNAME = "03_ligandmpnn"
INPUTS_DIRNAME = "inputs"


def inputs_root(stage: Path) -> Path:
    """Where the previous stage's transfer wrote its archives."""
    return stage / INPUTS_DIRNAME


def inputs_experiments(stage: Path) -> List[str]:
    root = inputs_root(stage)
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def input_archive_path(stage: Path, experiment: str, group_key: str) -> Path:
    return inputs_root(stage) / experiment / f"{group_key}.tar.gz"


def stage01_root(stage: Path) -> Path:
    return project_root(stage) / STAGE01_DIRNAME


def stage01_json_path(stage: Path, experiment: str, group_key: str) -> Path:
    """The stage-01 json a group was generated from.

    group_key is the json's filename stem (job_name() encodes it, and the
    transfer keeps the archive named after it), so a structure's generating
    config -- and through it its seed -- is recoverable from the archive path
    alone, with nothing extra recorded anywhere.
    """
    return experiment_json_dir(stage01_root(stage), experiment) / f"{group_key}.json"


def stage03_root(stage: Path) -> Path:
    return project_root(stage) / STAGE03_DIRNAME


def stage03_inputs_dir(stage: Path, experiment: str) -> Path:
    return stage03_root(stage) / INPUTS_DIRNAME / experiment


def stage03_archive_path(stage: Path, experiment: str, group_key: str) -> Path:
    return stage03_inputs_dir(stage, experiment) / f"{group_key}.tar.gz"


def fixed_residues_path(stage: Path, experiment: str) -> Path:
    """Where stage 01 records which residues it held fixed, one JSONL line per
    structure. Run state, not a result: it lives under job_runs/ and is not
    committed."""
    return stage / "job_runs" / f"fixed_residues_{experiment}.jsonl"


# Step 8. Stage 03: ProteinMPNN.

STAGE04_DIRNAME = "04_alphafold"


def stage04_root(stage: Path) -> Path:
    return project_root(stage) / STAGE04_DIRNAME


def stage04_archive_path(stage: Path, experiment: str, group_key: str) -> Path:
    return stage04_root(stage) / INPUTS_DIRNAME / experiment / f"{group_key}.tar.gz"


def prepared_dir(stage: Path, experiment: str, group_key: str) -> Path:
    """The complexes handed to LigandMPNN for one group, protein plus ligand.

    Kept per group rather than in one flat folder so provenance is structural:
    the old flat layout lost which experiment a structure came from and had to
    recover it afterwards from a committed map file.
    """
    return stage / "inputs_prepared" / experiment / group_key


def mpnn_work_dir(stage: Path, experiment: str, group_key: str) -> Path:
    """Scratch for one group's MPNN run: the jsonls it needs and the raw fastas
    it produces, before they are archived."""
    return stage / "job_runs" / "mpnn" / experiment / group_key


def fixed_positions_path(stage: Path, experiment: str, group_key: str) -> Path:
    return mpnn_work_dir(stage, experiment, group_key) / "fixed_positions.jsonl"


def tied_positions_path(stage: Path, experiment: str, group_key: str) -> Path:
    """Stage 07 only: the positions of copy A tied to their twins in copy B.

    RFD3 fuses the two copies into one chain, so the tie is same-chain --
    position i of the first lobe to position i of the second -- and it is what
    keeps the two units identical after the linker's shell is redesigned.
    """
    return mpnn_work_dir(stage, experiment, group_key) / "tied_positions.jsonl"


def linker_spec_path(stage: Path, experiment: str, group_key: str) -> Path:
    """What stage 07 found in each construct: the linker, its shell, and where
    the two copies sit. Stage 08 reads the linker back out of it rather than
    deriving it a second time."""
    return mpnn_work_dir(stage, experiment, group_key) / "linker.jsonl"


def bias_aa_path(stage: Path) -> Path:
    """Hand-chosen composition bias -- config, so it lives with the code and is
    committed, unlike everything else here."""
    return stage / "scripts" / "jsonls" / "bias_AA.jsonl"


def mpnn_outputs_path(stage: Path, experiment: str, group_key: str) -> Path:
    """Every sequence MPNN designed for a group, before the best are chosen."""
    return stage / "outputs" / experiment / f"{group_key}.tar.gz"


# Step 9. Stage 04: AlphaFold3.

LIGAND_SUFFIX = "ligand"
# Only for reading archives from the pipeline that folded without a ligand.
# Nothing writes it any more, and it is NOT the default: a call site that
# forgot the suffix used to get "_monomer" and match nothing, which reads
# exactly like an empty queue.
MONOMER_SUFFIX = "monomer"


def af3_job_name(sequence_id: str, suffix: str = LIGAND_SUFFIX) -> str:
    """The AF3 job name for a sequence.

    Stage 04 still uses '_monomer' although its dimer job is gone: the 16k
    production run's outputs are named that way, and dropping the suffix would
    make every one of them look unfolded and re-queue days of GPU time. Stage 06
    folds the same sequence again with the ligand, so it uses '_ligand' -- the
    suffix is what keeps the two stages' outputs from colliding.
    """
    return f"{sequence_id}_{suffix}"


def af3_json_experiment_dir(stage: Path, experiment: str) -> Path:
    return stage / "job_runs" / "af3" / experiment


def af3_json_dir(stage: Path, experiment: str, group_key: str) -> Path:
    """Where a group's AF3 input jsons are written. Run state, not results."""
    return af3_json_experiment_dir(stage, experiment) / group_key


def af3_json_path(stage: Path, experiment: str, group_key: str, sequence_id: str,
                  suffix: str = LIGAND_SUFFIX) -> Path:
    return (af3_json_dir(stage, experiment, group_key)
            / f"{af3_job_name(sequence_id, suffix)}.json")


def af3_output_dir(stage: Path, job_name: str) -> Path:
    """AF3's consolidated output for one job, flat by job name.

    Flat because sequence_id is globally unique, and because this is where the
    existing production outputs already live.
    """
    return stage / "outputs" / job_name


def af3_model_cif(stage: Path, job_name: str, sample_dir_name: str) -> Path:
    return af3_output_dir(stage, job_name) / sample_dir_name / f"{job_name}_{sample_dir_name}_model.cif"


def af3_ranking_path(stage: Path, job_name: str) -> Path:
    return af3_output_dir(stage, job_name) / f"{job_name}_ranking_scores.csv"


def af3_run_dir(stage: Path, job_name: str) -> Path:
    """AF3's per-job scratch (af_input/af_output), before consolidation."""
    return stage / "af3_runs" / job_name


# Step 10. Stage 05: alignment onto the seed sides, and the RFD3 inputs.
#
# Called 05_geometry_filtering until it stopped filtering. AlphaFold folds
# slightly differently from the design, so a clash measured on its prediction
# is partly AlphaFold's deviation rather than the design's geometry -- and the
# question it was asking has already been answered where it is objective, at
# stage 02, on the designed coordinates (zero protein-protein contacts under
# 1.8 A, zero protein-ligand under 2.2 A). The numbers are still measured and
# recorded here; nothing is rejected on them.

STAGE05_DIRNAME = "05_ligand_alignment"


def stage05_root(stage: Path) -> Path:
    return project_root(stage) / STAGE05_DIRNAME


def stage05_archive_path(stage: Path, experiment: str, group_key: str) -> Path:
    return stage05_root(stage) / INPUTS_DIRNAME / experiment / f"{group_key}.tar.gz"


def alignment_table_path(stage: Path, experiment: str) -> Path:
    """One row per pair built: the fit, and how close the copies came to each
    other and to the fibril. A record, not a filter."""
    return stage / TABLES_DIRNAME / f"stage_{stage_label(stage)}_alignment_{experiment}.csv"


def linker_span_table_path(stage: Path, experiment: str) -> Path:
    """One row per orientation: how far the linker has to reach, and the fewest
    residues that could do it."""
    return stage / TABLES_DIRNAME / f"stage_{stage_label(stage)}_pairs_{experiment}.csv"


# Step 10b. Stage 03's working paths, and the AF3 runner every folding stage
# shares. These sit here because that is where they have always been; they were
# briefly deleted along with a neighbouring block and are restored verbatim.


def ligand_work_dir(stage: Path, experiment: str, group_key: str) -> Path:
    return stage / "job_runs" / "ligandmpnn" / experiment / group_key


def redesign_spec_path(stage: Path, experiment: str, group_key: str) -> Path:
    """Which residues LigandMPNN may change, and which are tied to which.

    One file per group holding a record per structure, because the redesignable
    set is a property of the individual backbone, not of the group.
    """
    return ligand_work_dir(stage, experiment, group_key) / "redesign.jsonl"


def af3_shared_script(name: str) -> Path:
    """One AF3 runner script, shared by every stage that folds.

    Both stages invoke it the same way -- (json, stage root) in, a consolidated
    <stage>/outputs/<job_name>/ out -- so a second copy would only be a second
    thing to keep in step.
    """
    return Path(__file__).resolve().parent / "af3" / name


# Step 11. Stage 06: RFD3 linker generation.

STAGE06_DIRNAME = "06_rfd3_linker_generation"


def stage06_root(stage: Path) -> Path:
    return project_root(stage) / STAGE06_DIRNAME


def stage06_archive_path(stage: Path, experiment: str, group_key: str) -> Path:
    return stage06_root(stage) / INPUTS_DIRNAME / experiment / f"{group_key}.tar.gz"


def pair_dir(stage: Path, experiment: str, group_key: str) -> Path:
    """Where a stage keeps its per-sequence structures.

    ONE file per sequence: <sid>.pdb, the two aligned copies with no cellulose.
    Stage 06 unpacks the handover into the same relative path, so the "input"
    field of a json written in stage 05 resolves unchanged when RFD3 reads it
    in stage 06.
    """
    return stage / "inputs_prepared" / experiment / group_key


def pair_path(stage: Path, experiment: str, group_key: str, sequence_id: str) -> Path:
    """The aligned pair: two copies of one fold on the seed sides, cellulose
    removed.

    Both fibres are gone by the time this is written -- the one AlphaFold
    predicted, which was its own guess, and the one from the reference that the
    superposition was measured against. Neither is kept: the linker is generated
    on protein alone, and a fibre that has to come back later is regenerable by
    re-running this stage from its own input archives, which still hold the
    reference complexes.
    """
    return pair_dir(stage, experiment, group_key) / f"{sequence_id}.pdb"


def linker_json_dir(stage: Path, experiment: str) -> Path:
    """The RFD3 jsons whose linker length you fill in. Written by stage 05,
    unpacked to the same place by stage 06.

    Flat, exactly like stage 01's json/<experiment>/, so the RFD3 runner there
    drives these without adaptation: json_rel is a bare filename and
    group_key_from_json_rel() gives a key with no path separator in it.
    """
    return experiment_json_dir(stage, experiment)


# Step 12. Stage 07: ProteinMPNN on the linker.

STAGE07_DIRNAME = "07_proteinmpnn_linker"


def stage07_root(stage: Path) -> Path:
    return project_root(stage) / STAGE07_DIRNAME


def stage07_archive_path(stage: Path, experiment: str, group_key: str) -> Path:
    return stage07_root(stage) / INPUTS_DIRNAME / experiment / f"{group_key}.tar.gz"


# Step 13. Stage 08: AlphaFold3 on the finished construct.

STAGE08_DIRNAME = "08_alphafold_construct"

# The two folds stage 08 makes of every design. Both are monomers; what differs
# is whether the fibril is in the box.
# One fold, with the fibril. There were two for a while -- holo and apo, on the
# argument that a construct folding only in the fibre's presence is leaning on
# it -- but the question this stage asks is whether the linked construct still
# binds cellulose, and the apo fold answered a different one at twice the GPU.
HOLO_SUFFIX = "holo"    # with the fibril
FOLD_SUFFIXES = (HOLO_SUFFIX,)


def construct_pair_path(stage: Path, experiment: str, group_key: str,
                        sequence_id: str) -> Path:
    """The two copies as they stood BEFORE the linker was generated.

    Stage 06 keeps these -- its jsons name them by absolute path, so nothing
    deletes them -- which makes them the stable record of what the construct is
    supposed to superpose back onto. Reaching for stage 05's copy instead would
    break the moment that stage is cleaned.
    """
    return (stage06_root(stage) / "inputs_prepared" / experiment / group_key
            / f"{sequence_id}.pdb")


def sequence_id_from_design(design_name: str) -> str:
    """'p001_AB_000001' -> 'p001'. The pair a generated construct came from.

    Two things come off: the seed index job_name() appended, and the
    orientation linker_jsons.py appended before that. Neither is guessable from
    the sequence id itself, which contains underscores of its own.
    """
    group_key = group_key_from_job_name(design_name)
    for orientation in ("_AB", "_BA"):
        if group_key.endswith(orientation):
            return group_key[: -len(orientation)]
    return group_key


def stage08_root(stage: Path) -> Path:
    return project_root(stage) / STAGE08_DIRNAME


def stage08_archive_path(stage: Path, experiment: str, group_key: str) -> Path:
    return stage08_root(stage) / INPUTS_DIRNAME / experiment / f"{group_key}.tar.gz"

