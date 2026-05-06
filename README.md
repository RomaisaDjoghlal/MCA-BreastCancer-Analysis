# MCA-BreastCancer-Analysis

> Multivariate analysis of breast cancer clinical data using **Multiple Correspondence Analysis (MCA)**, applied to the Rotterdam Breast Cancer dataset.

**Academic TP Report** — ESI (École Supérieure d'Informatique), Option SIQ2  
Course: Analyse Multivariée des Données | Year: 2025/2026  
Authors: **DJOGHLAL Romaisa**, **CHOUIDER Ikram** | Supervisor: Mme BESSAH Naima

---

## Overview

This project applies Multiple Correspondence Analysis (MCA) to explore associations between demographic, clinical, and therapeutic characteristics of 2,982 breast cancer patients from the Rotterdam Breast Cancer dataset.

The main goals are:
- Identifying patient profiles based on age, tumor grade, and tumor size
- Understanding which characteristics are associated with a high risk of recurrence
- Revealing patterns in therapeutic strategies (chemotherapy vs. hormone therapy)

---

## Dataset

| Property | Value |
|----------|-------|
| Source | Rotterdam Breast Cancer dataset |
| Observations | 2,982 patients |
| Variables | 7 (1 quantitative + 6 qualitative) |

### Variables

| Variable | Description | Modalities |
|----------|-------------|------------|
| `age_cat` | Age category (discretized) | Q1, Q2, Q3, Q4 |
| `meno` | Menopausal status | 0 (pre), 1 (post) |
| `size` | Tumor size | ≤20mm, 20–50mm, >50mm |
| `grade` | Histological grade | 1, 2, 3 |
| `hormon` | Hormone therapy | 0 (no), 1 (yes) |
| `chemo` | Chemotherapy | 0 (no), 1 (yes) |
| `recur` | Recurrence | 0 (no), 1 (yes) |

---

## Methodology

### 1. Data Preparation
- The continuous `age` variable was **discretized into quartiles** (Q1–Q4) using `qcut`, each containing ~745 patients
- Missing values were handled by **median imputation** for numeric variables and **mode imputation** for categorical ones
- Output: `data_clean.csv` — 2,982 complete observations, 7 variables, no missing values

### 2. Multiple Correspondence Analysis (MCA)
MCA is an exploratory method that visualizes associations between qualitative variables in a reduced-dimension space. It transforms a complete disjunctive table into projections on factorial axes that maximize explained inertia.

---


