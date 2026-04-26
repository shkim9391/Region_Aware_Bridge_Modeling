# Region_Aware_Bridge_Modeling

Code and processed outputs supporting the manuscript:

**“Region-aware bridge modeling enables interpretable mesoscale representation of spatial transcriptomic tissue sections”**

## Overview

This repository contains the analysis scripts, processed outputs, and reproducibility materials for a region-aware bridge-modeling workflow applied to public 10x Genomics Visium HD spatial transcriptomic tissue sections.

The primary proof-of-concept analysis uses:

- one colorectal cancer (**CRC**) Visium HD section
- one breast cancer Visium HD section

The supplementary external applicability analyses use:

- one lung cancer Visium HD section
- one prostate cancer Visium HD section
- one ovarian cancer Visium HD section

The workflow constructs interpretable spot-level or bin-level bridge features, aggregates them into median-quadrant regional summaries, evaluates within-section spatial heterogeneity, performs partition-sensitivity and shuffle-null validation, and fits exploratory ridge and Bayesian sensitivity models on the primary CRC–breast region-level design matrix.

The repository is organized to support the main manuscript, Supplementary Methods, Supplementary Figures S1–S4, and Supplementary Data 1–6.

## Manuscript scope

The primary CRC–breast analysis is a proof-of-concept study. Each primary dataset contains one tissue section, and each section is summarized into four median-quadrant regions. Therefore, the primary statistical design contains eight region-level observations.

Accordingly:

- spatial validation is interpreted as **within-section validation**
- ridge models are interpreted as **regularized exploratory association models**
- Bayesian models are interpreted as **uncertainty-aware sensitivity analyses**
- external lung, prostate, and ovarian cancer analyses are interpreted as **workflow-applicability tests**, not disease-specific validation studies

The framework is intended as a lightweight, reproducible bridge-to-region representation step for spatial transcriptomic analysis.

## Public data sources

The original spatial transcriptomic datasets were obtained from the 10x Genomics datasets portal:

<https://www.10xgenomics.com/datasets>

Primary sections:

- **CRC:** Visium HD Human Colon Cancer — Gene Expression Library of Colon Cancer using the Human Whole Transcriptome Probe Set
- **Breast cancer:** Visium HD Spatial Gene Expression Library, Human Breast Cancer (Fresh Frozen)

Supplementary external sections:

- **Lung cancer:** public 10x Genomics Visium HD lung cancer section
- **Prostate cancer:** public 10x Genomics Visium HD prostate cancer section
- **Ovarian cancer:** public 10x Genomics Visium HD ovarian cancer section

Users should download the original source data directly from 10x Genomics before rerunning preprocessing steps. This repository provides processed analysis outputs and manuscript-supporting reproducibility materials.

## Repository contents

A cleaned release should contain the following logical components:

```text
repo_root/
  README.md
  LICENSE
  requirements.txt

  scripts/
    bridge_target_construction/
    patch_index_standardization/
    region_aware_aggregation/
    within_section_validation/
    ridge_modeling/
    bayesian_sensitivity_modeling/
    external_applicability/
    figure_generation/

  supplementary_data/
    Supplementary_Data_1/
    Supplementary_Data_2/
    Supplementary_Data_3/
    Supplementary_Data_4/
    Supplementary_Data_5/
    Supplementary_Data_6/

  figures/
    Figure1_revised_region_aware_bridge_workflow.png
    Figure2_revised_within_section_validation.png
    Figure3_revised_ridge_exploratory_modeling.png
    Figure4_revised_bayesian_sensitivity_modeling.png
    Figure_S1_primary_crc_breast_partitions_heatmaps.png
    Figure_S2_primary_shuffle_partition_sensitivity.png
    Figure_S3_external_solid_tumor_heatmaps.png
    Figure_S4_external_shuffle_partition_sensitivity.png

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19801620.svg)](https://doi.org/10.5281/zenodo.19801620)
