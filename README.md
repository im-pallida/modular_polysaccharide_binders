# Modular Polysaccharide Binder Design

A computational pipeline for designing modular proteins that bind to repetitive polysaccharide structures, using cellulose as the primary target.

The project combines **RFDiffusion3, LigandMPNN, ProteinMPNN, and AlphaFold3** to generate protein backbones, design amino acid sequences, evaluate predicted structures, and construct larger protein assemblies connected by peptide linkers.

The pipeline was initially developed for cellulose-binding proteins based on the 1CBH reference structure and subsequently adapted for testing with chitin. Its general approach may also be applicable to other targets with suitable repeating geometry, including non-crystalline polysaccharides.

## 1. Project overview

The main objective of this project was to develop a computational workflow for designing modular protein binders that follow the repeating geometry of a polysaccharide target.

Individual protein modules can potentially form larger assemblies through non-covalent interactions or be connected covalently using designed peptide linkers.

The pipeline consists of eight computational stages, starting with symmetry-guided backbone generation and ending with structural prediction of complete protein constructs.

The workflow was designed to operate on both local workstations and HPC clusters using SLURM.

## 2. Pipeline architecture

The computational pipeline is organized as follows:

```text
Reference protein–polysaccharide structures
                    |
                    v
Stage 01 — RFDiffusion3 symmetry generation
                    |
                    v
Stage 02 — Geometry filtering
                    |
                    v
Stage 03 — LigandMPNN sequence design
                    |
                    v
Stage 04 — AlphaFold3 prediction
                    |
                    v
Stage 05 — Protein pair alignment
                    |
                    v
Stage 06 — RFDiffusion3 linker generation
                    |
                    v
Stage 07 — ProteinMPNN linker sequence design
                    |
                    v
Stage 08 — AlphaFold3 construct validation
```

### Stage descriptions

| Stage | Description |
|---|---|
| **01 — Symmetry generation** | Generates protein backbones using a custom RFDiffusion3 implementation supporting linear symmetry. |
| **02 — Geometry filtering** | Aligns generated structures with the reference and evaluates protein–protein and protein–polysaccharide contacts and clashes. |
| **03 — LigandMPNN** | Designs amino acid sequences in the presence of the polysaccharide, preserving selected binding-site residues. |
| **04 — AlphaFold3** | Predicts the structures of individual protein modules and compares them with the designed backbones. |
| **05 — Ligand alignment** | Places the validated modules in their reference arrangement and prepares inputs for linker generation. |
| **06 — Linker generation** | Uses RFDiffusion3 to generate peptide linkers connecting two protein modules. |
| **07 — ProteinMPNN linker** | Designs linker sequences and selected neighboring residues while preserving the original protein modules. |
| **08 — AlphaFold3 construct** | Predicts complete linked constructs and evaluates whether their original modular arrangement is preserved. |

Each stage contains its own launcher and supporting scripts. Common functions for job execution, path handling, structural analysis, and file transfer are located in the `common/` directory.

## 3. Custom RFDiffusion3 symmetry implementation

One of the main developments of this project was extending the symmetry handling in RFDiffusion3.

The original symmetry framework was adapted to support protein modules arranged along a repeated linear geometry.

Three symmetry modes were introduced:

- **Tn:** Copies related by equally spaced translations along a Cartesian axis.
- **SCREWn:** Copies related by repeated translation and rotation.
- **TnEXACT:** Copies positioned using explicitly defined rigid transformations, including both translations and rotations.

### Automatic symmetry extraction

For `TnEXACT`, the pipeline can automatically extract the relative symmetry transformations from a reference structure containing multiple equivalent protein copies.

The `copy_transforms()` function in `symmetry_check.py` determines the rigid transformations between protein copies.

These transformations are passed to RFDiffusion3 through the `RFD3_T<N>EXACT_TRANSLATIONS` environment variable.

The transformations may contain:

- Three components describing a translation vector.
- Twelve components describing a rotation matrix and translation vector.

This allows the pipeline to reproduce geometries that cannot be represented by a constant translational rise or rotation around a predefined Cartesian axis.

If the input seed contains only one protein copy, the required transformations must be supplied separately.

### Overlay implementation

The custom symmetry code is stored separately from the original RFDiffusion3 installation:

```text
01_rfd3_symmetry_generation/
└── overlay/
    └── rfd3_t3_overlay/
        └── rfd3/
            └── inference/
                └── symmetry/
                    ├── frames.py
                    └── checks.py
```

The overlay is activated through `PYTHONPATH` when RFDiffusion3 is executed.

This approach avoids directly modifying the original installation and limits the custom behavior to runs that use the overlay.

## 4. Requirements

The pipeline requires Python 3 and the following software:

- RFDiffusion3 — protein backbone generation.
- LigandMPNN — ligand-aware protein sequence design.
- ProteinMPNN — protein and linker sequence design.
- AlphaFold3 — structural prediction.
- SLURM — optional, for HPC cluster execution.

Each external tool must be installed separately. Their model checkpoints, databases, and software environments are not included in this repository.

The pipeline's own Python dependencies are listed in:

```text
common/requirements.txt
```

Install them using:

```bash
python3 -m pip install -r common/requirements.txt
```

The primary packages are `gemmi`, `numpy`, and `biopython`.

GPU resources are required for the generative and structure-prediction stages.

## 5. Installation and environment configuration

Clone the repository:

```bash
git clone https://github.com/im-pallida/modular_polysaccharide_binders.git

cd modular_polysaccharide_binders
```

The pipeline uses a centralized configuration system implemented in:

```text
common/site_config.py
```

Machine-specific settings can be placed in:

```text
common/site.local.env
```

Create this file from the provided example:

```bash
cp common/site.local.env.example common/site.local.env
```

Relevant settings include:

| Variable | Purpose |
|---|---|
| `RFD3_EXE` | Path to RFDiffusion3 executable |
| `CKPT` | RFDiffusion3 checkpoint |
| `LIGANDMPNN_ROOT` | LigandMPNN installation |
| `LIGANDMPNN_CHECKPOINT` | LigandMPNN model weights |
| `MPNN_ROOT` | ProteinMPNN installation |
| `AF3_ROOT` | AlphaFold3 installation |
| `MODEL_DIR` | AlphaFold3 model parameters |
| `PIPELINE_MODE` | Workstation or cluster execution |
| `PIPELINE_PARTITION` | Optional SLURM partition |

The pipeline attempts to discover software installations automatically. Explicit settings take priority when supplied.

### Check the computational environment

Before running the pipeline, execute:

```bash
python3 common/check_site.py
```

This script checks the required software paths, Python dependencies, and environment configuration.

It reports missing components and suggests the settings needed for the current machine.

To inspect whether the local repository contains untracked files or unpushed changes, use:

```bash
python3 common/tools/check_repo.py
```

These are diagnostic checks, not full end-to-end pipeline tests.

## 6. Running the pipeline

The pipeline is divided into independent stages. Each stage receives structures or sequences from the previous stage and transfers accepted outputs to the next stage.

Commands should be executed from the repository root.

### Stage 01 — Backbone generation

Place the experiment JSON configurations in:

```text
01_rfd3_symmetry_generation/json/<experiment>/
```

Example input configurations are available in:

```text
01_rfd3_symmetry_generation/json/examples/
```

For example, to prepare a small cellulose experiment:

```bash
mkdir -p 01_rfd3_symmetry_generation/json/demo_cellulose

cp 01_rfd3_symmetry_generation/json/examples/1cbh_1xcellulose_dp20.json \
01_rfd3_symmetry_generation/json/demo_cellulose/
```

Start backbone generation:

```bash
python3 01_rfd3_symmetry_generation/scripts/stage_01_launcher.py demo_cellulose --count 2
```

Here, `--count` specifies the number of structures generated for each JSON configuration.

The example JSON files contain paths from the original development environment. The pipeline attempts to resolve these paths against the current repository and its reference structures; the seed path should still be checked before running a new experiment.

### Stage 02 — Geometry filtering

```bash
python3 02_geometry_filtering/scripts/stage_02_launcher.py --experiment demo_cellulose
```

This stage evaluates generated structures according to protein–protein contacts, protein–protein clashes, and protein–polysaccharide clashes.

### Stage 03 — LigandMPNN sequence design

```bash
python3 03_ligandmpnn/scripts/stage_03_launcher.py --experiment demo_cellulose
```

LigandMPNN designs sequences with the polysaccharide included as structural context.

By default, 16 sequences are generated per backbone, and the three sequences with the highest overall confidence are selected.

### Stage 04 — AlphaFold3 prediction

```bash
python3 04_alphafold/scripts/stage_04_launcher.py --experiment demo_cellulose
```

Individual protein modules are predicted in the presence of the polysaccharide.

The predicted structures are compared with the designed backbones, and accepted structures are transferred to the next stage.

### Stage 05 — Protein pair alignment

```bash
python3 05_ligand_alignment/scripts/stage_05_launcher.py --experiment demo_cellulose
```

The selected proteins are aligned to the two reference protein positions.

Two linker orientations are prepared for each pair:

- A → B
- B → A

The distances between the corresponding termini are recorded to help select linker lengths.

### Stage 06 — Linker generation

Before running this stage, linker lengths must be specified in the generated JSON configurations.

The `set_linker.py` helper can update these lengths automatically.

For example:

```bash
python3 06_rfd3_linker_generation/scripts/helping_scripts/set_linker.py \
--experiment demo_cellulose --length 14
```

The value of 14 residues is only an example. Linker lengths should be selected based on the distances recorded in Stage 05.

Start linker generation:

```bash
python3 06_rfd3_linker_generation/scripts/stage_06_launcher.py --experiment demo_cellulose
```

The default is five generated linker backbones per JSON configuration.

### Stage 07 — ProteinMPNN linker sequence design

```bash
python3 07_proteinmpnn_linker/scripts/stage_07_launcher.py --experiment demo_cellulose
```

ProteinMPNN designs the linker and nearby residues while preserving the previously validated protein modules.

The three sequences with the lowest ProteinMPNN score are selected by default.

### Stage 08 — AlphaFold3 construct validation

```bash
python3 08_alphafold_construct/scripts/stage_08_launcher.py --experiment demo_cellulose
```

The complete linked constructs are predicted with AlphaFold3.

Predicted structures are compared with the original two-module arrangement, excluding the linker region from the structural comparison.

The final selection checks whether the relative positions of the two protein units are maintained.

## 7. Filtering and selection criteria

The main filtering and evaluation criteria are summarized below.

| Stage | Main criteria |
|---|---|
| 01 | No backbone chain breaks; non-loop fraction greater than 0.7 |
| 02 | No Cα protein–protein clashes ≤ 1.8 Å; no all-atom protein–polysaccharide clashes ≤ 2.2 Å; at least 7 inter-copy Cα contacts within 4–8 Å |
| 03 | Three highest LigandMPNN overall-confidence scores |
| 04 | RMSD below 2.5 Å; matched fraction above 0.8 |
| 05 | Alignment quality and close contacts recorded; no automatic rejection based on these values |
| 06 | Valid linker configuration and backbone generation |
| 07 | Three lowest ProteinMPNN scores; preservation of fixed residues and identical protein-module sequences |
| 08 | Whole-construct RMSD below 5.0 Å; matched fraction above 0.8 |

The selection criteria represent computational filters and do not provide direct experimental evidence of protein stability or binding affinity.

## 8. Output organization

The pipeline organizes its results by experiment name.

Depending on the stage, outputs are stored in directories such as:

```text
inputs/
inputs_prepared/
outputs_raw/
outputs_clean/
sorted_clean/
tables/
logs/
job_runs/
```

Accepted and rejected structures are generally stored separately in compressed archives:

```text
sorted_clean/<experiment>/passed/
sorted_clean/<experiment>/rejected/
```

Evaluation tables are written as CSV files containing structural metrics, prediction scores, selection status, and rejection reasons.

The final stage produces:

```text
08_alphafold_construct/tables/
    stage_08_results_<experiment>.csv
    stage_08_passed_<experiment>.csv
```

The second file contains the selected constructs and their protein-unit and linker sequences.

Most generated structures, output archives, tables, and logs are excluded from Git to avoid storing large computational datasets in the source repository.

## 9. Transferability to other polysaccharides

The pipeline was developed using cellulose as the primary target, but its general approach is based on repeated geometry rather than a particular polysaccharide.

Therefore, it could potentially be adapted to other symmetrically repeated structures, including chitin and non-crystalline polysaccharides.

For a new target, the reference structure, symmetry parameters, and validation criteria may need to be adapted.

The current repository includes cellulose and chitin reference structures as examples.

## 10. Limitations

This repository contains a research-stage computational pipeline.

Important limitations include:

- The designed proteins have not been experimentally validated for binding or structural stability.
- A suitable reference geometry is required for symmetry-guided generation.
- Automatic symmetry extraction requires multiple equivalent protein copies in the seed.
- Stage 02 currently expects a reference containing two protein copies.
- Linker lengths must be selected before Stage 06.
- External software installations and model weights are not distributed with this repository.
- A complete end-to-end automated test suite is not currently included.

Predicted structures should therefore be treated as computational candidates for further investigation.

## 11. Reproducibility

For reproducible experiments, the following information should be retained:

- Input JSON configurations and reference structures.
- Git commit and software versions.
- Model checkpoints and relevant parameters.
- Generated protein structures and sequences.
- Filtering results and prediction scores.
- Execution settings and any parameter overrides.

The source code and example configurations are provided in this repository. Large production datasets are not included by default.

## 12. Project status

This repository documents the development of a computational pipeline for modular polysaccharide-binding protein design during a Master's research internship.

It is intended to support further computational development, reproducibility, and the selection of candidate designs for future experimental investigation.
