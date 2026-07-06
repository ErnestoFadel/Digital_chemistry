"""
Plot training and validation loss curves across five hyperparameter
configurations (learning rate sweeps, regularization, and batch composition).
Each configuration is compared against the baseline learning rate in its own
2x2 subplot panel to avoid overlapping scatter points, then saved as two
separate PNG files: LR_Train.png and LR_Val.png.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# --- Config ---

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.dirname(SCRIPT_DIR)
HYPERPARAM_DIR = os.path.join(RESULT_DIR, "hyperparameter optimization")

LR = load_and_smooth(os.path.join(HYPERPARAM_DIR, "metrics_omnivore_intraassay_split0_5.5e-5LR.csv"))
LRsqrt = load_and_smooth(os.path.join(HYPERPARAM_DIR, "metrics_omnivore_intraassay_split0_1.58e-4LR.csv"))
LR10x = load_and_smooth(os.path.join(HYPERPARAM_DIR, "metrics_5.5e-4LR.csv"))
LRsqrt_regularized = load_and_smooth(
    os.path.join(HYPERPARAM_DIR, "metrics_omnivore_intraassay_split0_1.58e-4LR_regularization.csv")
)
LR_50batch = load_and_smooth(os.path.join(HYPERPARAM_DIR, "metrics_50_batchsize.csv"))


def load_and_smooth(path, window=50):
    """Load a metrics CSV and add a rolling-mean smoothed training loss column.

    Parameters
    ----------
    path : str
        Path to the metrics CSV file.
    window : int
        Rolling window size for smoothing (default 50).

    Returns
    -------
    df : pd.DataFrame
        Original data with an added 'smooth' column.
    """
    df = pd.read_csv(path)
    df["smooth"] = df["Train loss"].rolling(window, min_periods=1).mean()
    return df


# Each entry: (dataframe, color, label)
series = {
    "LR": (LR, "C0", r"$\mathrm{LR} = 5\times10^{-5}$"),
    "LR10x": (LR10x, "C2", r"$\mathrm{LR} = 5\times10^{-5}\times10$"),
    "LRsqrt": (LRsqrt, "C1", r"$\mathrm{LR} = 5\times10^{-5}\times\sqrt{10}$"),
    "LRsqrt_reg": (
        LRsqrt_regularized,
        "C3",
        r"$\mathrm{LR} = 5\times10^{-5}\times\sqrt{10}$ Regularized",
    ),
    "LR_50batch": (LR_50batch, "C4", "50-Assay Batch"),
}

# Grid layout: each subplot gets its own pair so nothing overlaps
# (top-left): LR vs LR x10        (top-right): LR vs LRsqrt
# (bottom-left): LR vs LRsqrt_reg (bottom-right): LR vs 50-assay batch
subplot_groups = [
    ["LR", "LR10x"],
    ["LR", "LRsqrt"],
    ["LR", "LRsqrt_reg"],
    ["LR", "LR_50batch"],
]

# --- Training loss: 2x2 grid ---

fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)

for ax, group in zip(axes.flat, subplot_groups):
    for key in group:
        df, color, label = series[key]
        ax.scatter(df["epoch"], df["Train loss"], s=8, alpha=0.18, color=color)
        ax.plot(df["epoch"], df["smooth"], color=color, linewidth=2.2,
                 label=label)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="upper right")

for ax in axes[-1, :]:
    ax.set_xlabel("Epoch", fontsize=12)
for ax in axes[:, 0]:
    ax.set_ylabel("Training Loss", fontsize=12)

fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("LR_Train.png", dpi=300)


# --- Validation loss: single combined plot ---

plt.figure(figsize=(9, 6))

for df, color, label in series.values():
    val_df = df.dropna(subset=["Validation loss"])
    plt.plot(val_df["epoch"], val_df["Validation loss"], linewidth=2,
              color=color, label=label)

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("LR_Val.png", dpi=300)