"""
Plot Fisher-z aggregated Pearson correlation heatmaps for the mean-prediction
baseline, comparing predictive mode (assay, scaffold, target) against split
method (butina, random, scaffold, target) across three datasets (kinodata,
landrum, omnivore), shown side by side with a shared color scale.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Config ---

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.dirname(SCRIPT_DIR)

PATHS = [
    os.path.join(RESULTS_DIR, "kinodata", "mean_results_kinodata.csv"),
    os.path.join(RESULTS_DIR, "landrum", "mean_results_landrum.csv"),
    os.path.join(RESULTS_DIR, "omnivore", "mean_results_omnivore.csv"),
]
NAMES = ["kinodata", "landrum", "omnivore"]

ROW_ORDER = ["assay", "scaffold", "target"]  # row order in plot (predictive mode)
COL_ORDER = ["butina", "random", "scaffold", "target"]  # column order (split method)


def fisher_mean(series):
    """Aggregate Pearson correlations via the Fisher z-transformation: convert
    each correlation to z, average, then convert back to correlation scale.
    This avoids the bias of directly averaging bounded correlation values.""" 
    z = np.arctanh(series)
    return np.tanh(z.mean())


fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for i, path in enumerate(PATHS):
    df = pd.read_csv(path)  # read data from specific dataset

    agg_df = (
        df.groupby(["predictive", "split_method"])["pearson"]
        .apply(fisher_mean)
        .reset_index()
    )
    heatmap_data = agg_df.pivot(
        index="predictive", columns="split_method", values="pearson"
    )

    # Reindex to a fixed row/column order; missing combinations show as empty
    heatmap_data = heatmap_data.reindex(index=ROW_ORDER, columns=COL_ORDER)

    # Only draw the color bar for the last (rightmost) dataset
    draw_cbar = i == 2 

    sns.heatmap(
        heatmap_data,
        ax=axes[i],
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,  # same color scale for all plots
        cbar=draw_cbar,
        cbar_kws={"shrink": 1.0} if draw_cbar else None,
    )

    # Formatting for each subplot
    axes[i].set_title(NAMES[i], fontsize=14, pad=10)
    axes[i].set_xlabel("Split", fontsize=12)

    if i == 0:  # y-labels only on the leftmost subplot
        axes[i].set_ylabel("Predictive", fontsize=12)
        # Rotate the y-labels to be vertical, matching the paper's figure
        axes[i].set_yticklabels(
            axes[i].get_yticklabels(), rotation=90, va="center"
        ) 
    else:
        axes[i].set_ylabel("")
        axes[i].tick_params(axis="y", length=0)

plt.tight_layout()
plt.savefig("Mean_Activity_Prediction_Performance_Pearson.png", bbox_inches="tight")