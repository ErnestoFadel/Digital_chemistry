"""
Plot assay size distributions (number of molecules per assay) for each
dataset as side-by-side log-scale histograms, with the median assay size
marked as a vertical dashed line.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# --- Config ---

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(RESULT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "Csv_files")

DATASETS = ["kinodata", "landrum", "omnivore", "omnivore_short"]

fig, axes = plt.subplots(1, len(DATASETS), figsize=(18, 5), sharey=True)

for ax, dataset in zip(axes, DATASETS):
    data = pd.read_csv(os.path.join(DATA_DIR, f"{dataset}_dataset.csv"))

    # Compute assay sizes and their median
    assay_sizes = data.groupby("Assay_id").size()
    median_size = assay_sizes.median()

    print(f"{dataset} median assay size: {median_size}")

    # Histogram with median marked as a vertical dashed line
    ax.hist(assay_sizes, bins=30, log=True, color="steelblue", alpha=0.7)
    ax.axvline(median_size, color="red", linestyle="--", linewidth=2)

    ax.set_title(f"{dataset} dataset assay size distribution")
    ax.set_xlabel("Assay size (# molecules)")
    ax.set_ylabel("Count (log scale)")

plt.tight_layout()
plt.savefig("Assay_Size_Histogram.png", dpi=300)