# Datasets Description

This folder contains the eight benchmark datasets used in the experimental evaluation of the proposed framework. All datasets are publicly available and were processed without access to class labels.

## Data Sources

### 1. UCI Machine Learning Repository
The following datasets were obtained from the UCI Machine Learning Repository (Kelly et al., 2026):
- **heart.csv**: Cardiac diagnosis (hybrid features). 270 instances, 13 features.
- **ionosphere.csv**: Radar signal data (numerical). 351 instances, 34 features.
- **wpbc.csv**: Breast cancer prognosis (numerical). 198 instances, 34 features.
- **wdbc.csv**: Breast cancer diagnosis (numerical). 569 instances, 30 features.
- **segment.csv**: Image segmentation (numerical). 2310 instances, 19 features.
- **zoo.csv**: Animal taxonomy (categorical/binary). 101 instances, 16 features.

*URL:* https://archive.ics.uci.edu

### 2. Modern Biomedical Benchmarks
- **dataYM2018.csv**: Neuroblastoma Electronic Health Records. 169 instances, 12 features (hybrid).
  *Source:* Chicco, D. (2026). Neuroblastoma EHRs Open Data Repository.  
  *URL:* https://davidechicco.github.io/neuroblastoma_EHRs_data/

- **TCGA_Glioma.csv**: Glioma grading genomic/clinical data (hybrid). 839 instances, 23 features.
  *Source:* The Cancer Genome Atlas (TCGA) via curated biomedical repositories.

## Preprocessing Notes
All datasets were preprocessed as follows:
- **Categorical features:** One-hot encoded.
- **Missing values:** Median imputation (numerical), Mode imputation (categorical).
- **Scaling:** No scaling applied to preserve original distributional properties for entropy calculation.

## Automatic Download
You can download all datasets automatically by running:
```bash
python download_datasets.py