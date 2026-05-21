# LPBF 316L-Cu Corrosion Intelligence Dashboard

This repository is a **paper-based companion dashboard** for the draft study:

**Electrochemical and surface characterization of laser powder bed fusion copper-bearing 316L stainless steel in H2O2/NaCl simulated inflammatory electrolyte**

The project uses only the numerical values reported in the manuscript tables and text. It does **not** fabricate raw OCP, EIS, polarization, SEM, or EDS data.

## Purpose

The aim is to elevate the manuscript with a reproducible, transparent, ML-assisted electrochemical fingerprinting workflow.

Because the current manuscript contains only four summarized experimental states, this repository does **not** claim a general predictive ML model. Instead, it provides:

- feature extraction from the paper's electrochemical tables;
- corrosion severity indexing;
- PCA-based electrochemical fingerprint mapping;
- nearest-prototype classification based on the four reported paper states;
- Streamlit visualization for GitHub/demo use;
- manuscript-ready text for adding a data-driven section.

## Experimental states represented

| Sample | Alloy | Medium | Interpretation |
|---|---|---|---|
| S1 | AISI 316L | 0.9% NaCl | Stable passive reference |
| S2 | AISI 316L-Cu | 0.9% NaCl | Cu-modified weaker passivity |
| S3 | AISI 316L | 0.9% NaCl + H2O2 | Peroxide-weakened passive film |
| S4 | AISI 316L-Cu | 0.9% NaCl + H2O2 | Cu-assisted peroxide degradation |

## Key limitation

This is a **low-data, manuscript-derived workflow**. The current dataset is suitable for interpretable visualization and transparent severity ranking, not for strong generalized prediction. A stronger ML claim would require raw electrochemical curves, additional replicates, time-series EIS, SEM image quantification, or literature-expanded datasets.

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Repository structure

```text
.
├── app.py
├── requirements.txt
├── data/
│   └── processed/
│       ├── paper_extracted_corrosion_features.csv
│       └── severity_components.csv
├── src/
│   ├── feature_engineering.py
│   ├── ml_fingerprint.py
│   └── plotting.py
├── figures/
│   ├── corrosion_severity_index.png
│   ├── pca_fingerprint_map.png
│   └── feature_heatmap.png
└── paper_text/
    ├── manuscript_addition_ml_section.md
    ├── github_streamlit_description.md
    └── figure_captions.md
```

## Recommended manuscript wording

Use the phrase:

> ML-assisted electrochemical fingerprinting

Avoid the phrase:

> machine-learning prediction of corrosion behavior

until more raw data and replicates are added.
