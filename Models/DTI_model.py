"""
Baseline Drug-Target Interaction (DTI) model for binding affinity prediction.
Encodes ligand ECFP6 fingerprints and ESM-2 protein embeddings through separate
MLPs, concatenates them, and minimizes MSE loss on pIC50 values. Outputs per-assay
Fisher-z weighted Pearson correlation and RMSE on the test set.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, TQDMProgressBar, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from scipy.stats import pearsonr
from sklearn.metrics import root_mean_squared_error
import numpy as np
import os
import pandas as pd
import time
from Data_split import assay_split
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem

# General purpose seed, used throughout
SEED = 42

# Dataset to train and test model on
dataset_name = "omnivore_short"

# ====== Define base directory ======
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "Csv_files")
FEATURE_DIR = os.path.join(SCRIPT_DIR, "protein_featurization", "Encoded_protein")
RESULT_DIR = os.path.join(PROJECT_ROOT, "Results")

def get_accel():
    """Return the best available hardware accelerator."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class DTI_data(Dataset):
    """PyTorch Dataset holding ligand fingerprints, protein embeddings, and pIC50
    targets. Returns tensors for a single compound by index."""

    def __init__(self, ligand, protein, pIC50):
        self.ligand = ligand
        self.protein = protein
        self.pIC50 = pIC50

    def __len__(self):
        return self.ligand.shape[0]

    def __getitem__(self, idx):
        """Return (ligand, protein, pIC50) tensors for compound idx."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        ligand_ = torch.as_tensor(self.ligand[idx].astype(np.float32))
        protein_ = torch.as_tensor(
            self.protein[idx].astype(np.float32).reshape(-1)
        )  # Reshape to ensure shape (1280,)
        pIC50_ = torch.as_tensor(self.pIC50[idx].astype(np.float32))
        return ligand_, protein_, pIC50_


class DTI(LightningModule):
    """PyTorch Lightning module for binding affinity prediction.
    Encodes ligand (ECFP6) and protein (ESM-2) features through separate MLPs,
    concatenates them, and passes through an interaction MLP to predict pIC50.
    Trained by minimizing MSE loss."""

    def __init__(
        self,
        ligand_sz,
        protein_sz,
        train_data,
        valid_data,
        test_data,
        batch_size=1024,
        lr=5e-5,
    ):
        super().__init__()
        self.lr = lr
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.batch_size = batch_size

        # Ligand encoder - 3 layers
        self.ligand_encoder = nn.Sequential(
            nn.Linear(ligand_sz, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
        )

        # Protein encoder - 2 layers
        self.protein_encoder = nn.Sequential(
            nn.Linear(protein_sz, 512), nn.SiLU(), nn.Linear(512, 512)
        )

        # Interaction MLP: BatchNorm -> Linear -> SiLU -> Linear -> SiLU -> Dropout -> output
        self.interaction = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(512, 1),
        )

    def training_step(self, batch, batch_idx):
        x_ligand, x_protein, y = batch
        z = self(x_ligand, x_protein)
        loss = F.mse_loss(z, y)
        self.log("Train loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_ligand, x_protein, y = batch
        z = self(x_ligand, x_protein)
        loss = F.mse_loss(z, y)
        self.log("Validation loss", loss)

    def test_step(self, batch, batch_idx):
        x_ligand, x_protein, y = batch
        z = self(x_ligand, x_protein)
        loss = F.mse_loss(z, y)
        self.log("Test loss", loss)

    def predict_step(self, batch, batch_idx):
        x_ligand, x_protein, y = batch
        return self(x_ligand, x_protein)

    def configure_optimizers(self):
        """AdamW optimizer with ReduceLROnPlateau scheduler (factor=0.5, patience=50),
        monitored on validation loss."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=50
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "Validation loss"},
        }

    def forward(self, x_ligand, x_protein):
        """Encode ligand and protein, concatenate, and return scalar pIC50 predictions."""
        h_ligand = self.ligand_encoder(x_ligand)
        h_protein = self.protein_encoder(x_protein)
        combined = torch.cat([h_ligand, h_protein], dim=1)
        return self.interaction(combined).flatten()

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=False,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=False,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=False,
            persistent_workers=True,
        )


if __name__ == "__main__":

    start = time.time()

    # --- Load data and prepare splits ---

    df = pd.read_csv(os.path.join(DATA_DIR, f"{dataset_name}_dataset.csv"))
    df = df.rename(
        columns={"Canonical_smile": "smiles", "pIC50": "pIC50", "Assay_id": "assay_id"}
    )

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)

    df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)

    valid_mask = df["mol"].notnull()
    df = df[valid_mask].reset_index(drop=True)
    print(f"Dropped {(~valid_mask).sum()} rows with invalid SMILES")

    ligand_fingerprints = []
    for mol in df["mol"]:
        fp = morgan_gen.GetFingerprintAsNumPy(mol)
        ligand_fingerprints.append(fp)

    ligand_fingerprints = np.array(ligand_fingerprints)  # shape: (N, 2048)
    print(" === Ligand fingerprints computed === ")

    protein_features = np.load(
        os.path.join(FEATURE_DIR, f"{dataset_name}_protein_feature_vector.npy"),
        allow_pickle=True,
    )  # shape: (N, D)

    # Filter protein features to match valid SMILES rows
    protein_features = np.array(protein_features)[valid_mask.values]
    print(" === Protein features loaded === ")

    df = df[["smiles", "assay_id", "pIC50"]].reset_index(drop=True)
    df["ligand"] = list(ligand_fingerprints)  # store as objects for split indexing
    df["protein"] = list(protein_features)

    X = df[["assay_id", "ligand", "protein"]]
    y = df["pIC50"].values
    print(" === X and y array prepared === ")

    results = []
    exceptions = []

    for i, (train_split, val_split, test_split) in enumerate(
        assay_split(X, y, test_size=0.2, val_size=0.1, random_state=SEED)
    ):
        # --- Build datasets ---
        train_set = DTI_data(
            train_split["ligand"], train_split["protein"], train_split["pIC50"]
        )
        val_set = DTI_data(
            val_split["ligand"], val_split["protein"], val_split["pIC50"]
        )
        test_set = DTI_data(
            test_split["ligand"], test_split["protein"], test_split["pIC50"]
        )

        y_test = test_split["pIC50"]
        test_assay_ids = test_split["assay_ids"]

        # --- Initialize model ---
        dti_model = DTI(
            ligand_sz=2048,
            protein_sz=train_set.protein.shape[1],
            train_data=train_set,
            valid_data=val_set,
            test_data=test_set,
            batch_size=1024,
            lr=5e-5,
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="Validation loss", save_top_k=1, mode="min"
        )

        trainer = Trainer(
            max_epochs=500,
            logger=CSVLogger(save_dir = os.path.join(SCRIPT_DIR, "logs"), name=f"DTI log {dataset_name}split number {i+10}"),
            accelerator=get_accel(),
            devices=1,
            strategy="auto",
            callbacks=[
                checkpoint_callback,
                EarlyStopping(monitor="Validation loss", patience=100),
                TQDMProgressBar(refresh_rate=20),
            ],
            precision="32-true",
        )

        # --- Train and load best checkpoint ---
        trainer.fit(dti_model)

        best_model = DTI.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            ligand_sz=2048,
            protein_sz=train_set.protein.shape[1],
            train_data=train_set,
            valid_data=val_set,
            test_data=test_set,
        )

        # --- Predict on test set ---
        predictions = trainer.predict(
            best_model, dataloaders=best_model.test_dataloader()
        )
        y_pred = torch.cat(predictions).detach().cpu().numpy()

        # --- Per-assay Fisher-z weighted Pearson correlation ---
        weighted_sum = 0
        weight_total = 0
        for assay_id in np.unique(test_assay_ids):
            mask = test_assay_ids == assay_id
            n = mask.sum()
            if n > 3 and np.std(y_pred[mask]) > 1e-8 and np.std(y_test[mask]) > 1e-8:
                r, _ = pearsonr(y_test[mask], y_pred[mask])

                if np.isnan(r) or r > 0.9999 or r < -0.9999:
                    print(
                        f"Assay {assay_id} has n={n} and r={r:.3f} with std(y_pred)={np.std(y_pred[mask])} and std(y_test)={np.std(y_test[mask]):.3f}"
                    )
                    print(
                        "y_pred values:", y_pred[mask], "y_test values:", y_test[mask]
                    )
                    exceptions.append({
                        "dataset": dataset_name,
                        "fold": i + 1,
                        "assay_id": assay_id,
                        "n_samples": n,
                        "pearson_r": r,
                        "testing_values": y_test[mask],
                        "predicted_values": y_pred[mask],
                    })

                if r > 0.9999 or r < -0.9999 or np.isnan(r):
                    continue

                weighted_sum += (n - 3) * np.arctanh(r)
                weight_total += n - 3

        rho_assay = np.tanh(weighted_sum / weight_total) if weight_total > 0 else np.nan

        # --- Global metrics ---
        pearson, _ = pearsonr(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)

        print(
            f"split number {i}: train size = {len(train_split['pIC50'])} | "
            f"val size = {len(val_split['pIC50'])} | test size = {len(test_split['pIC50'])}"
        )
        print(
            f"split number {i}: rho_global={pearson:.3f} | rho_assay={rho_assay:.3f} | RMSE={rmse:.3f}"
        )

        results.append({
            "dataset": dataset_name,
            "fold": i + 1,
            "pearson": pearson,
            "rho_assay": rho_assay,
            "rmse": rmse,
        })

    # --- Save results ---
    pd.DataFrame(results).to_csv(
        os.path.join(RESULT_DIR, f"{dataset_name}/{dataset_name}_DTI_model_results.csv"), index=False
    )

    pd.DataFrame(exceptions).to_csv(
        os.path.join(RESULT_DIR, f"{dataset_name}_DTI_model_exceptions.csv"), index=False
    )

    end = time.time()
    print(f"Total execution time: {end - start:.2f} seconds")