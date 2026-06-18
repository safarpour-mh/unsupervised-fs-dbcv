# Label-Agnostic Feature Selection in Biomedical Data: Integrating DBCV and Fuzzy Crowding Distance

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18989265.svg)](https://doi.org/10.5281/zenodo.18989265)

This repository contains the official **Python implementation** of the manuscript:  
**"Label-Agnostic Feature Selection in Biomedical Data: Integrating DBCV and Fuzzy Crowding Distance for Scalable Dimensionality Reduction"**  
*Author: Mohammad Hossein Safarpour*  
*Journal: PeerJ Computer Science (Under Revision)*

> **Note on Reproducibility:** In direct response to Reviewer 1's concern regarding proprietary software, the entire framework has been **rewritten from MATLAB to Python** using fully open-source libraries. The code is non-proprietary, platform-independent, and freely reproducible.

---

## 📌 Overview

This framework provides a fully unsupervised, multi-objective feature selection method for high-dimensional biomedical datasets where class labels are unavailable, noisy, or expensive. It jointly optimizes:

1. **Distributional Informativeness**: via Discretized Shannon Entropy (default $K=5$ bins).
2. **Structural Coherence**: via Density-Based Clustering Validation (DBCV) index (replacing the traditional Davies-Bouldin Index to handle concave/nested clusters).
3. **Subset Diversity**: via a novel Fuzzy Crowding-Distance ranking mechanism.

---

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/safarpour-mh/unsupervised-fs-dbcv.git
   cd unsupervised-fs-dbcv