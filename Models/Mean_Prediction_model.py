"""
Mean-based baseline model for binding affinity prediction.
Predicts pIC50 values using assay-specific, target-specific, or scaffold-specific
training means. Evaluated across random, scaffold, target, and Butina splits using
Pearson correlation, RMSE, and Kendall's Tau.
""" 

from scipy.stats import pearsonr, kendalltau
from sklearn.metrics import root_mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import os
import pandas as pd
import time
from Data_split import (
    random_split_mean,
    scaffold_split_mean,
    butina_split_mean,
    target_split_mean,
)
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold


class MeanAssayPredictor(BaseEstimator, RegressorMixin):
    """
    A scikit-learn compatible regressor that predicts based on assay-specific means.

    For each assay in the test set:
    - If the assay was seen during training, predict its training mean.
    - If the assay is new, predict the overall training mean.

    Attributes
    ----------
    train_mean_ : dict
        Maps assay_id to mean pIC50 from training data.
    overall_mean_ : float
        Overall training mean, used for unseen assays.
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Fit the model by computing mean pIC50 values per assay.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, 1)
            Assay IDs for training samples.
        y : array-like of shape (n_samples,)
            Target pIC50 values.

        Returns
        -------
        self : fitted estimator
        """
        X = np.asarray(X).flatten()
        y = np.asarray(y).flatten()

        # Compute mean pIC50 for each assay
        self.train_mean_ = {}
        for unique_id in np.unique(X):
            mask = X == unique_id
            self.train_mean_[unique_id] = y[mask].mean()

        # Fallback mean for assays not seen during training
        self.overall_mean_ = y.mean()

        return self

    def predict(self, X):
        """
        Predict pIC50 values using assay-specific or overall mean.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, 1)
            Assay IDs for test samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted pIC50 values.
        """
        if not hasattr(self, "train_mean_"):
            raise RuntimeError("Model must be fitted before calling predict()")

        X = np.asarray(X).flatten()

        # Default to overall mean, then overwrite with known assay means
        y_pred = np.full(len(X), self.overall_mean_)
        for unique_id, mean_val in self.train_mean_.items():
            mask = X == unique_id
            y_pred[mask] = mean_val

        return y_pred


if __name__ == "__main__":

    start = time.time()

    # --- Load data ---

    dataset_name = "omnivore"

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "Csv_files")
    RESULT_DIR = os.path.join(PROJECT_ROOT, "Results")

    df = pd.read_csv(os.path.join(DATA_DIR, f"{dataset_name}_dataset.csv"))
    df = df.rename(columns={
        "Canonical_smile": "smiles",
        "pIC50": "pIC50",
        "Assay_id": "assay_id",
        "Target_id": "target",
    })
    print(" === Loading data completed === ")

    # --- Compute scaffolds and filter invalid SMILES ---

    df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
    invalid_rows = [i for i, mol in enumerate(df["mol"]) if mol is None] 

    df = df.drop(invalid_rows).reset_index(drop=True)
    print(f"Dropped {len(invalid_rows)} rows with invalid SMILES")

    for i, mol in enumerate(df["mol"]):
        try:
            df.at[i, "scaffold"] = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=False
            )
        except RuntimeError:
            # Treat molecule as its own scaffold if SMILES is malformed
            df.at[i, "scaffold"] = df.at[i, "smiles"]

    X = df[["assay_id", "smiles", "target", "scaffold", "mol"]].values
    y = df["pIC50"].values
    print(" === Scaffold computation completed === ")

    # --- Precompute all splits ---

    print(" === Starting data splitting === ")

    # Mapping predictive mode -> column index in X
    pred_to_idx = {"assay": 0, "target": 2, "scaffold": 3}

    # Mapping predictive mode -> valid split methods
    pred_to_splits = {
        "assay": ["random", "scaffold", "butina"],
        "target": ["random", "scaffold", "butina"],
        "scaffold": ["random", "target", "butina"],
    }

    all_splits = {
        "random": list(random_split_mean(X, y)),
        "scaffold": list(scaffold_split_mean(X, y)),
        "target": list(target_split_mean(X, y)),
        "butina": list(butina_split_mean(X, y)),
    }

    # --- Train and evaluate ---

    print(" === Starting model training and evaluation === ")

    results = []

    for pred in pred_to_idx:

        print(f" Predicting based on {pred} means")

        idx = pred_to_idx[pred]

        for split_name in pred_to_splits[pred]:
            for i, (X_train, y_train, X_test, y_test) in enumerate(
                all_splits[split_name]
            ):
                model = MeanAssayPredictor()

                X_tr = X_train[:, idx]
                X_ts = X_test[:, idx]

                model.fit(X_tr, y_train)
                y_pred = model.predict(X_ts)

                # --- Evaluation metrics ---

                pearson_global = pearsonr(y_pred, y_test)[0]
                rmse = root_mean_squared_error(y_pred, y_test)
                kendal = kendalltau(y_pred, y_test)[0]

                # --- Train/test overlap of grouping IDs ---

                overlap_percentage = (
                    len(set(X_tr) & set(X_ts)) / len(set(X_ts)) * 100
                ) 

                print(
                    f"Fold {i + 1}, {split_name}: Pearson = {pearson_global:.4f}, "
                    f"RMSE = {rmse:.4f}, Kendall's Tau = {kendal:.4f}, "
                    f"Overlap = {overlap_percentage:.2f}%"
                ) 

                results.append({
                    "dataset": dataset_name,
                    "predictive": pred,
                    "split_method": split_name,
                    "fold": i + 1,
                    "pearson": pearson_global,
                    "kendall_tau": kendal,
                    "rmse": rmse,
                    "overlap": overlap_percentage,
                })

    # --- Save results ---

    pd.DataFrame(results).to_csv(
        os.path.join(RESULT_DIR, f"{dataset_name}/{dataset_name}_mean_model_results.csv"), index=False
    )

    end = time.time()
    print(f"Total execution time: {(end - start) / 60:.2f} minutes")