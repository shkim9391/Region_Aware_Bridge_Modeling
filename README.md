# Region_Aware_Bridge_Modeling

Code and processed outputs supporting the manuscript:

**“Region-aware bridge modeling for mesoscale representation of spatial transcriptomic tissue sections”**

## Overview

This repository contains scripts and processed data used to construct a region-aware bridge modeling workflow for two public Visium HD spatial transcriptomic tissue sections: one colorectal cancer (CRC) section and one breast cancer (BC) section. The pipeline covers dataset loading, preprocessing, spot-level bridge-feature generation, quadrant-level aggregation, standardized design-matrix construction, region-aware ridge modeling, Bayesian regression modeling, and supporting validation analyses.

The repository is organized to support the main manuscript figures and Supplementary Data S1–S13.

## Important note on legacy filenames

Some scripts and output files retain the term `ou_branching` from an earlier internal development stage. In the submitted manuscript, these files are used only for **region-aware bridge-input and design-matrix generation** and **do not represent a separate OU-Branching analysis**.

## Public data sources

The analysis uses two publicly available datasets from the 10x Genomics datasets portal:

- **CRC**: *Visium HD Human Colon Cancer -- Gene Expression Library of Colon Cancer (Visium HD) using the Human Whole Transcriptome Probe Set*
- **BC**: *Visium HD Spatial Gene Expression Library, Human Breast Cancer (Fresh Frozen)*

Users should download the raw/source data directly from 10x Genomics before running the preprocessing pipeline.

## Repository structure

```text
repo_root/
  data/
    bridge_sample_state_breast_core4_regions4.csv
    bridge_sample_state_crc_core4_regions4.csv
    ou_branching_bridge_design_matrix_breast_...
    ou_branching_bridge_design_matrix_crc_...
    ou_branching_bridge_input_breast_core4_...
    ou_branching_bridge_input_crc_breast_...
    ou_branching_bridge_input_crc_core4_...
    ou_bridge_standardization_stats_breast_...
    ou_bridge_standardization_stats_crc_...

  scripts/
    build_ou_bridge_design_matrix.py
    build_quadrant_bridge_rows.py
    build_z_spatial.py
    evaluate_wsi_to_z.py
    fit_region_aware_bridge_bayes.py
    fit_region_aware_bridge_model.py
    load_breast.py
    load_crc.py
    make_patch_index.py
    merge_ou_branching_bridge_input.py
    run_within_image_spatial_validation.py
    scanpy_breast.py
    scanpy_crc.py
    train_wsi_to_z.py
    verification_clusters.py
    within_image_heterogeneity_metrics_by_...
    wsi_z_dataset.py

  README.md

### Data loading and preprocessing
- `load_crc.py`  
  Loads the CRC Visium HD dataset and prepares core metadata and spatial coordinates for downstream analysis.

- `load_breast.py`  
  Loads the BC Visium HD dataset and prepares core metadata and spatial coordinates for downstream analysis.

- `scanpy_crc.py`  
  Dataset-specific Scanpy preprocessing workflow for the CRC section.

- `scanpy_breast.py`  
  Dataset-specific Scanpy preprocessing workflow for the BC section.

- `verification_clusters.py`  
  Optional quality-control / exploratory cluster verification script. Not required for the core manuscript pipeline unless explicitly stated.

### Spot-level bridge-feature pipeline
- `build_z_spatial.py`  
  Builds spatial latent/target representations used for bridge-feature modeling.

- `make_patch_index.py`  
  Creates patch-level spatial indexing and coordinate mappings used to link local image/spatial units to bridge targets.

- `wsi_z_dataset.py`  
  Constructs the dataset object used for bridge-model training and evaluation.

- `train_wsi_to_z.py`  
  Trains the spot-/patch-level bridge prediction model.

- `evaluate_wsi_to_z.py`  
  Evaluates the trained bridge model and generates predicted bridge features for downstream region-aware analysis.

### Region-aware aggregation and matrix construction
- `build_quadrant_bridge_rows.py`  
  Aggregates spot-level bridge predictions into quadrant-level summaries for CRC and BC and exports region-level rows.

- `merge_ou_branching_bridge_input.py`  
  Merges CRC and BC region-level bridge summaries into the joint bridge-input matrices used for downstream analysis.  
  **Note:** filename retained for historical reasons.

- `build_ou_bridge_design_matrix.py`  
  Standardizes region-level bridge features and constructs CRC-specific, BC-specific, and joint design matrices.  
  **Note:** filename retained for historical reasons.

### Region-aware modeling
- `fit_region_aware_bridge_model.py`  
  Fits the region-aware ridge regression models and generates reduced-model selection outputs and coefficient summaries.

- `fit_region_aware_bridge_bayes.py`  
  Fits the Bayesian regression models used for posterior inference on region-level bridge-feature dependencies.

## Supplementary Data mapping

### Region-level summaries and design matrices
- **S1**: CRC regional summary table
- **S2**: CRC bridge-input matrix
- **S3**: CRC design matrix
- **S4**: CRC standardization statistics
- **S5**: BC regional summary table
- **S6**: BC bridge-input matrix
- **S7**: BC design matrix
- **S8**: BC standardization statistics
- **S9**: Joint CRC+BC bridge-input matrix
- **S10**: Joint CRC+BC design matrix

These files are generated from the region-aware aggregation and matrix-construction steps, primarily:
- `build_quadrant_bridge_rows.py`
- `merge_ou_branching_bridge_input.py`
- `build_ou_bridge_design_matrix.py`

### Validation outputs
- **S11**: within-section heterogeneity metrics across partitioning schemes
- **S12**: shuffle-null summary statistics
- **S13**: raw long-format shuffle-null outputs

If these files are generated by a separate validation script, include that script in this repository and document it here. If they are generated inside an existing script, state that explicitly.

## Recommended execution order

A typical workflow is:

1. `load_crc.py`
2. `load_breast.py`
3. `scanpy_crc.py`
4. `scanpy_breast.py`
5. `build_z_spatial.py`
6. `make_patch_index.py`
7. `wsi_z_dataset.py`
8. `train_wsi_to_z.py`
9. `evaluate_wsi_to_z.py`
10. `build_quadrant_bridge_rows.py`
11. `merge_ou_branching_bridge_input.py`
12. `build_ou_bridge_design_matrix.py`
13. `fit_region_aware_bridge_model.py`
14. `fit_region_aware_bridge_bayes.py`
15. `run_within_image_spatial_validation.py`

## Figures

- **Figure 1** is a conceptual workflow figure.
- **Figures 2–4** and **Supplementary Figures S1–S2** are supported by the processed outputs and model results generated by this repository.

## Reproducibility notes

To improve reproducibility, this repository should also include:
- `README.md`
- `requirements.txt` or `environment.yml`
- the processed CSV outputs described in Supplementary Data S1–S13

## Contact / manuscript context

This repository accompanies the manuscript on region-aware bridge modeling for mesoscale spatial characterization of CRC and BC tissue sections.
