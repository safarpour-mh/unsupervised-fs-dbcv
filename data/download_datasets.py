"""
Automated Dataset Downloader for the Unsupervised FS-DBCV Framework
===================================================================
This script automatically downloads the eight benchmark datasets used 
in the manuscript from their official public sources to ensure full 
reproducibility of the experiments.

Datasets:
- UCI Machine Learning Repository: heart, ionosphere, wpbc, wdbc, segment, zoo
- Modern Biomedical Benchmarks: dataYM2018 (Neuroblastoma), TCGA Glioma

Author: Mohammad Hossein Safarpour
Correspondence: m.safarpour@iau.ac.ir
GitHub: https://github.com/safarpour-mh/unsupervised-fs-dbcv
"""

import os
import requests

# Directory where this script is located
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Official URLs for the datasets
# Note: UCI datasets are downloaded in their original raw format.
DATASETS_URLS = {
    "heart.csv": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart.dat",
    "ionosphere.csv": "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data",
    "wpbc.csv": "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.data",
    "wdbc.csv": "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
    "segment.csv": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/segment/segment.dat",
    "zoo.csv": "https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data",
    "dataYM2018.csv": "https://raw.githubusercontent.com/davidechicco/neuroblastoma_EHRs_data/master/neuroblastoma_EHRs_data.csv"
}

def download_file(url, dest_path):
    """Download a file from a URL and save it to the destination path."""
    try:
        print(f"  -> Downloading from {url} ...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            f.write(response.content)
        print(f"     Successfully saved to: {dest_path}")
        return True
    except Exception as e:
        print(f"     [ERROR] Failed to download: {e}")
        return False

def main():
    print("=" * 70)
    print("Dataset Downloader for Unsupervised FS-DBCV Framework")
    print("=" * 70)
    
    downloaded_count = 0
    skipped_count = 0
    
    for filename, url in DATASETS_URLS.items():
        filepath = os.path.join(DATA_DIR, filename)
        
        if os.path.exists(filepath):
            print(f"[SKIP] {filename} already exists in the data directory.")
            skipped_count += 1
        else:
            print(f"[DOWNLOAD] {filename}")
            if download_file(url, filepath):
                downloaded_count += 1
    
    # Note on TCGA Glioma dataset
    tcga_path = os.path.join(DATA_DIR, "TCGA_Glioma.csv")
    if not os.path.exists(tcga_path):
        print("\n[INFO] TCGA_Glioma.csv is not included in the automated download.")
        print("       Please download it manually from the TCGA portal or curated repositories")
        print("       and place it in this 'data/' folder.")
    else:
        print(f"[SKIP] TCGA_Glioma.csv already exists.")
        skipped_count += 1

    print("\n" + "=" * 70)
    print(f"Download Summary: {downloaded_count} downloaded, {skipped_count} already present.")
    print("Please ensure all 8 CSV files are present before running the experiments.")
    print("=" * 70)

if __name__ == "__main__":
    main()