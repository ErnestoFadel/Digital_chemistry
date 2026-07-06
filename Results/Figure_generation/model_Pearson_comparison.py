"""
Compare absolute prediction, inter-assay ranking, and intra-assay ranking
models via paired t-tests on Fisher z-transformed per-assay Pearson
correlations, and visualize results as a dot plot with paired-fold lines
and significance brackets, one panel per dataset.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

plt.rcParams["font.family"] = "Lato"

# --- Config ---

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.dirname(SCRIPT_DIR)

DATASETS = {
    "kinodata": {
        "direct": "kinodata/DTI_results_kinodata.csv",
        "inter_assay": "kinodata/kinodata_pairwise_inter_assay_summary.csv",
        "intra_assay": "kinodata/kinodata_pairwise_intra_assay_summary.csv",
    },
    "landrum": {
        "direct": "landrum/DTI_results_landrum.csv",
        "inter_assay": "landrum/landrum_pairwise_inter_assay_summary.csv",
        "intra_assay": "landrum/landrum_pairwise_intra_assay_summary.csv",
    },
    "omnivore": {
        "direct": "omnivore/DTI_results_omnivore.csv",
        "inter_assay": "omnivore/omnivore_pairwise_inter_assay_summary.csv",
        "intra_assay": "omnivore/omnivore_pairwise_intra_assay_summary.csv",
    },
    "omnivore_short": {
        "direct": "omnivore_short/DTI_results_omnivore_short.csv",
        "inter_assay": "omnivore_short/omnivore_short_pairwise_inter_assay_summary.csv",
        "intra_assay": "omnivore_short/omnivore_short_pairwise_intra_assay_summary.csv",
    },
}

MODELS = ["direct", "inter_assay", "intra_assay"]
LABELS = ["absolute\nprediction", "ranking\n(inter-assay)", "ranking\n(intra-assay)"]

# Comparisons used by both the stats and plotting functions
MODEL_PAIRS = [
    ("direct", "inter_assay"),
    ("direct", "intra_assay"),
    ("inter_assay", "intra_assay"),
]


def load(datasets):
    """Read per-fold rho_assay (and rho_global for the direct model only)
    into a single tidy DataFrame with columns: dataset, model, fold,
    rho_assay, rho_global."""
    rows = []
    for ds, files in datasets.items():
        for model, fname in files.items():
            df = pd.read_csv(os.path.join(RESULT_DIR, fname))
            df.columns = df.columns.str.strip()  # tolerate stray whitespace

            for i, (_, r) in enumerate(df.iterrows()):
                rows.append({
                    "dataset": ds,
                    "model": model,
                    "fold": i,
                    "rho_assay": float(str(r["rho_assay"]).lstrip("'")),
                    "rho_global": (
                        float(str(r["pearson"]).lstrip("'"))
                        if model == "direct"
                        else np.nan
                    ),
                })
    return pd.DataFrame(rows)


def stats(scores):
    """Run paired t-tests on Fisher z-transformed rho_assay values between
    each pair of models, per dataset. Returns a DataFrame of p-values."""
    rows = []
    for ds in scores["dataset"].unique():
        piv = scores[scores["dataset"] == ds].pivot(
            index="fold", columns="model", values="rho_assay"
        )
        for a, b in MODEL_PAIRS:
            za = np.arctanh(piv[a].values)
            zb = np.arctanh(piv[b].values)
            diff = za - zb
            print(
                f"{ds} {a} vs {b}: diffs={diff}, mean={diff.mean():.4f}, "
                f"std={diff.std(ddof=1):.6f}"
            )
            t, p = ttest_rel(za, zb)
            print(f"  t={t}, p={p}")
            rows.append({"dataset": ds, "comparison": f"{a} vs {b}", "p": p})
    return pd.DataFrame(rows)


def sig_marker(p):
    """Return a significance marker string for a given p-value."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def plot(scores, stats_df, out):
    """Draw one dot-plot panel per dataset: per-fold correlations connected
    by paired-fold lines, global Pearson for the direct model, and
    significance brackets for any model comparison with p < 0.05."""
    ds_names = list(DATASETS.keys())
    fig, axes = plt.subplots(
        1, len(ds_names), figsize=(4 * len(ds_names), 4.5), squeeze=False
    )
    axes = axes[0]
    x = np.arange(len(MODELS))

    for ax, ds in zip(axes, ds_names):
        d = scores[scores["dataset"] == ds]
        piv = d.pivot(
            index="fold", columns="model", values="rho_assay"
        ).reindex(columns=MODELS)

        # Paired-fold lines connecting each fold's scores across models
        for _, row in piv.iterrows():
            ax.plot(x, row.values, color="lightgray", linewidth=0.9, zorder=1)

        # Per-fold dots: within-assay correlation for each model
        for i, m in enumerate(MODELS):
            vals = d[d["model"] == m]["rho_assay"].values
            ax.scatter(
                [i] * len(vals),
                vals,
                color="#2E5C8A",
                s=55,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
                label="on assay groups" if (ax is axes[0] and i == 0) else None,
            )

        # Gray dots: global Pearson correlation, direct model only
        gvals = d[d["model"] == "direct"]["rho_global"].values
        ax.scatter(
            [0] * len(gvals),
            gvals,
            color="gray",
            s=55,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
            label="over entire test set" if ax is axes[0] else None,
        )

        # Significance brackets for each model pair with p < 0.05
        ds_stats = stats_df[stats_df["dataset"] == ds].set_index("comparison")
        ymax = max(d["rho_assay"].max(), d["rho_global"].max())
        ymin = min(d["rho_assay"].min(), d["rho_global"].min())
        span = ymax - ymin
        step = 0.08 * span
        for k, (a, b) in enumerate(MODEL_PAIRS):
            p = ds_stats.loc[f"{a} vs {b}", "p"]
            if p >= 0.05:  # skip non-significant comparisons
                continue
            y = ymax + (k + 1) * step
            xa, xb = MODELS.index(a), MODELS.index(b)
            ax.plot(
                [xa, xa, xb, xb],
                [y, y + step * 0.2, y + step * 0.2, y],
                color="black",
                linewidth=0.9,
            )
            ax.text(
                (xa + xb) / 2,
                y + step * 0.05,
                sig_marker(p),
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # Give the y-axis room for the brackets
        ax.set_ylim(ymin - 0.05 * span, ymax + (len(MODEL_PAIRS) + 1) * step)

        ax.set_title(ds, fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS, fontsize=12)
        if ax is axes[0]:
            ax.set_ylabel("Test correlation (Pearson)", fontsize=12)

    axes[0].legend(loc="center", frameon=False, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", transparent=True)
    print(f"Saved: {out}")


if __name__ == "__main__":
    scores = load(DATASETS)
    s = stats(scores)
    s.to_csv(os.path.join(RESULT_DIR, "model_comparison_stats.csv"), index=False)
    print(s.to_string(index=False))
    plot(scores, s, os.path.join(RESULT_DIR, "NN_model_results.png"))