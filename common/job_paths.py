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
 
 
def resolve_seed_path(stage: Path, raw_input: str, json_path_: Path) -> Optional[Path]:
    """Locate the seed structure a json's "input" field points at."""
    candidate = Path(raw_input)
    if candidate.is_absolute():
        return candidate if candidate.is_file() else None
    for base in (stage, json_path_.parent):
        resolved = (base / candidate).resolve()
        if resolved.is_file():
            return resolved
    return None
 
 
# Step 3. Outputs.
 
 
def raw_dir(stage: Path, experiment: str, group_key: str) -> Path:
    return stage / "outputs_raw" / experiment / group_key
 
 
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
STAGE03_DIRNAME = "03_protein_mpnn"
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
    """The backbones handed to ProteinMPNN for one group, ligand stripped.

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


def bias_aa_path(stage: Path) -> Path:
    """Hand-chosen composition bias -- config, so it lives with the code and is
    committed, unlike everything else here."""
    return stage / "scripts" / "jsonls" / "bias_AA.jsonl"


def mpnn_outputs_path(stage: Path, experiment: str, group_key: str) -> Path:
    """Every sequence MPNN designed for a group, before the best are chosen."""
    return stage / "outputs" / experiment / f"{group_key}.tar.gz"


# Step 9. Stage 04: AlphaFold3.

def af3_job_name(sequence_id: str) -> str:
    """The AF3 job name for a sequence.

    Still carries the '_monomer' suffix although the dimer job is gone: the
    16k production run's outputs are named this way, and dropping the suffix
    would make every one of them look unfolded and re-queue days of GPU time.
    """
    return f"{sequence_id}_monomer"


def af3_json_experiment_dir(stage: Path, experiment: str) -> Path:
    return stage / "job_runs" / "af3" / experiment


def af3_json_dir(stage: Path, experiment: str, group_key: str) -> Path:
    """Where a group's AF3 input jsons are written. Run state, not results."""
    return af3_json_experiment_dir(stage, experiment) / group_key


def af3_json_path(stage: Path, experiment: str, group_key: str, sequence_id: str) -> Path:
    return af3_json_dir(stage, experiment, group_key) / f"{af3_job_name(sequence_id)}.json"


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


# Step 10. Stage 05: LigandMPNN interface redesign.

STAGE05_DIRNAME = "05_ligandmpnn"


def stage05_root(stage: Path) -> Path:
    return project_root(stage) / STAGE05_DIRNAME


def stage05_archive_path(stage: Path, experiment: str, group_key: str) -> Path:
    return stage05_root(stage) / INPUTS_DIRNAME / experiment / f"{group_key}.tar.gz"


def complex_dir(stage: Path, experiment: str, group_key: str) -> Path:
    """The protein+ligand complexes handed to LigandMPNN, one pdb per sequence.

    Built here rather than carried from stage 04: the AF3 prediction has no
    ligand, and the ligand's placement only exists once the prediction has been
    superposed onto the reference it was designed against.
    """
    return stage / "inputs_prepared" / experiment / group_key


def complex_path(stage: Path, experiment: str, group_key: str, sequence_id: str) -> Path:
    return complex_dir(stage, experiment, group_key) / f"{sequence_id}.pdb"


def rejected_complex_dir(stage: Path, experiment: str, group_key: str) -> Path:
    """Complexes the clash check turned away, kept for inspection."""
    return stage / "inputs_rejected" / experiment / group_key


def clash_table_path(stage: Path, experiment: str) -> Path:
    return stage / TABLES_DIRNAME / f"stage_{stage_label(stage)}_clashes_{experiment}.csv"


def ligand_work_dir(stage: Path, experiment: str, group_key: str) -> Path:
    return stage / "job_runs" / "ligandmpnn" / experiment / group_key


def redesign_spec_path(stage: Path, experiment: str, group_key: str) -> Path:
    """Which residues LigandMPNN may change, and which are tied to which.

    One file per group holding a record per sequence, because the 8 A shell is
    a property of the individual complex, not of the group.
    """
    return ligand_work_dir(stage, experiment, group_key) / "redesign.jsonl"


def ligand_outputs_path(stage: Path, experiment: str, group_key: str) -> Path:
    return stage / "outputs" / experiment / f"{group_key}.tar.gz"


# Step 11. Stage 06: RFD3 linker generation.

STAGE06_DIRNAME = "06_rfd3_linker"


def stage06_root(stage: Path) -> Path:
    return project_root(stage) / STAGE06_DIRNAME


def stage06_archive_path(stage: Path, experiment: str, group_key: str) -> Path:
    return stage06_root(stage) / INPUTS_DIRNAME / experiment / f"{group_key}.tar.gz"

