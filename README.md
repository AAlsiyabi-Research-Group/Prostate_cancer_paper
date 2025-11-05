# Genetic Algorithm–Based Feature Selection (Parallel Implementation)

### Authors
**Syed Ahsan Shahid**, **Ahmed Al-Harrasi**, **Adil Al-Siyabi**  
Natural and Medical Sciences Research Center, University of Nizwa, Oman  
(2025)

---

### Paper
**"A Minimal Plasma Proteome-Based Biomarker Panel for Accurate Prostate Cancer Diagnosis"**  
*Syed Ahsan Shahid, Ahmed Al-Harrasi, Adil Al-Siyabi* (2025)

This repository provides the reproducible implementation of the **Genetic Algorithm (GA)**–based feature-selection framework developed for the above study.  
It enables large-scale, parallelized biomarker discovery in high-dimensional omics datasets.

---

## Overview

This script implements a **parallelized GA-based feature selection pipeline** using the `sklearn-genetic` library.  
It was originally applied to plasma proteomics data to identify a minimal, high-performing panel of proteins for prostate cancer classification.

### Key Features
- Parallelized GA runs across multiple processors  
- Configurable population size, crossover, and mutation rates  
- Supports multiple algorithms (Logistic Regression, SVM, etc.)  
- Aggregates top-performing feature subsets and frequency counts  
- Reusable for any **omics, transcriptomic, proteomic, or tabular** dataset  

---

## Dataset

This framework is compatible with the **Olink Explore 1536 pan-cancer plasma proteomics dataset** published by  
**Álvez et al. (2023), Nature Communications**,  
[*Next-generation pan-cancer blood proteome profiling using proximity extension assay*](https://www.nature.com/articles/s41467-023-39765-y).

> The dataset is publicly available and **not redistributed** here.  
> Please cite Álvez et al. (2023) when using their dataset.

---

## Requirements

Install dependencies:

```bash
pip install pandas numpy scikit-learn sklearn-genetic openpyxl
