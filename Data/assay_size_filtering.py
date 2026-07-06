"""
Create `omnivore_short` from `omnivore`, dropping all assays with fewer than
MIN_ASSAY_SIZE compounds.

IMPORTANT: the CSV and the protein .npy are aligned by row order. A single
boolean mask is built over rows and applied to BOTH, so they stay aligned.

Usage:
    python make_omnivore_short.py
"""

import os
import numpy as np
import pandas as pd

# --- Config ---

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "Csv_files")
FEATURE_DIR = os.path.join(
    PROJECT_ROOT, "Models", "protein_featurization", "Encoded_protein"
)

SRC_NAME = "omnivore"
DST_NAME = "omnivore_short"

MIN_ASSAY_SIZE = 3  # keep assays with at least this many compounds

# Column that identifies the assay. The training script renames "Assay_id" ->
# "assay_id", but the raw CSV uses "Assay_id". Both are checked for.
ASSAY_COL_CANDIDATES = ["Assay_id", "assay_id"]


def main():
    """Load the omnivore dataset and its protein features, filter out assays
    smaller than MIN_ASSAY_SIZE while keeping the CSV and .npy row-aligned,
    and write the filtered result as omnivore_short."""
    csv_path = os.path.join(DATA_DIR, f"{SRC_NAME}_dataset.csv")
    npy_path = os.path.join(FEATURE_DIR, f"{SRC_NAME}_protein_feature_vector.npy")

    df = pd.read_csv(csv_path)
    print(f"Loaded {csv_path}: {len(df)} rows")

    # Locate the assay-id column
    assay_col = next((c for c in ASSAY_COL_CANDIDATES if c in df.columns), None)
    if assay_col is None:
        raise KeyError(
            f"None of {ASSAY_COL_CANDIDATES} found in CSV columns: "
            f"{list(df.columns)}"
        )

    protein = np.load(npy_path, allow_pickle=True)
    protein_shape = (
        np.asarray(protein).shape if hasattr(protein, "shape") else len(protein)
    )
    print(f"Loaded {npy_path}: shape {protein_shape}")

    # Sanity check: CSV rows and protein rows must correspond 1:1
    if len(protein) != len(df):
        raise ValueError(
            f"Row mismatch: CSV has {len(df)} rows but protein array has "
            f"{len(protein)}. They must be aligned by row order; cannot "
            f"safely filter."
        )

    # --- Build the keep-mask: rows whose assay has >= MIN_ASSAY_SIZE compounds ---
    counts = df[assay_col].map(df[assay_col].value_counts())
    keep_mask = (counts >= MIN_ASSAY_SIZE).to_numpy()

    n_assays_before = df[assay_col].nunique()
    n_rows_before = len(df)

    df_short = df[keep_mask].reset_index(drop=True)
    protein_short = np.asarray(protein)[keep_mask]

    n_assays_after = df_short[assay_col].nunique()
    n_rows_after = len(df_short)

    # --- Report filtering summary ---
    print("\n=== Filtering summary ===")
    print(
        f"assays:  {n_assays_before:>8} -> {n_assays_after:>8} "
        f"({n_assays_before - n_assays_after} removed)"
    )
    print(
        f"rows:    {n_rows_before:>8} -> {n_rows_after:>8} "
        f"({n_rows_before - n_rows_after} removed, "
        f"{100 * (n_rows_before - n_rows_after) / n_rows_before:.1f}% of compounds)"
    )

    # Final alignment check
    assert len(df_short) == len(protein_short), "post-filter desync!"

    # --- Write outputs ---
    out_csv = os.path.join(DATA_DIR, f"{DST_NAME}_dataset.csv")
    out_npy = os.path.join(FEATURE_DIR, f"{DST_NAME}_protein_feature_vector.npy")

    df_short.to_csv(out_csv, index=False)
    np.save(out_npy, protein_short)

    print(f"\nWrote {out_csv}")
    print(f"Wrote {out_npy}")
    print('\nTo use: set dataset_name = "omnivore_short" in the training script.')


if __name__ == "__main__":
    main()