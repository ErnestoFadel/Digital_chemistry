"""
Pairwise binding affinity model using ECFP6 ligand fingerprints and ESM-2 protein
embeddings. Supports intra-assay training (real assay groups) and inter-assay
training (shuffled fake groups with matched size distribution). Outputs per-assay
and global Pearson correlations on the test set.
"""

import numpy as np
import pandas as pd
import os
import time
import gc
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from collections import defaultdict
from Data_split import assay_split

torch.backends.cudnn.benchmark = True

# Random seed used throughout
SEED = 42

# Training mode, can be "inter_assay" or "intra_assay"
training_mode = "inter_assay"

# Dataset to train and test model on
dataset_name = "kinodata"

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


class Pairwise_data(Dataset):
    """PyTorch Dataset holding ligand fingerprints, protein embeddings, pIC50
    targets, and assay IDs. Returns tensors for a single compound by index."""

    def __init__(self, ligand, protein, pIC50, assay_ids):
        self.ligand = ligand
        self.protein = protein
        self.pIC50 = pIC50
        self.assay_ids = assay_ids

    def __len__(self):
        return self.ligand.shape[0]

    def __getitem__(self, idx):
        """Return (ligand, protein, assay_id, pIC50, index) tensors for compound idx."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        ligand_ = torch.as_tensor(self.ligand[idx].astype(np.float32))
        protein_ = torch.as_tensor(self.protein[idx].astype(np.float32).reshape(-1))  # Reshape to ensure shape (1280,)
        pIC50_ = torch.as_tensor(self.pIC50[idx].astype(np.float32))
        assay_ids_ = torch.as_tensor(self.assay_ids[idx])
        return ligand_, protein_, assay_ids_, pIC50_, torch.tensor(idx)


class AssayBatchSampler:
    """Batch sampler that keeps each assay's compounds together in the same batch.
    Batches are capped by both compound count and assay count. Oversized assays
    are chunked across multiple batches. Shuffles assay order each epoch.""" 

    def __init__(self, assay_ids, max_batch_size, max_assays_per_batch=100,
                 shuffle=True, seed=SEED, drop_last=False):
        self.max_batch_size = max_batch_size  # cap on compounds per batch
        self.max_assays_per_batch = max_assays_per_batch  # cap on # assays (sets) per batch
        self.batch_size = max_batch_size  # Lightning compatibility
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        self.assay_to_indices = defaultdict(list)
        for idx, assay_id in enumerate(assay_ids):
            self.assay_to_indices[assay_id].append(idx)

        self.assays = list(self.assay_to_indices.keys())
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _get_assay_order(self):
        assays = self.assays.copy()
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(assays)
        return assays

    def __iter__(self):
        assays = self._get_assay_order()

        batch = []
        batch_size = 0       # compounds accumulated
        assay_count = 0      # assays (sets) accumulated

        for assay in assays:
            indices = self.assay_to_indices[assay]
            assay_size = len(indices)

            # --- handle oversized assay (bigger than the compound cap) ---
            if assay_size > self.max_batch_size:
                # flush whatever is pending first, so chunks don't mix with a partial batch
                if batch:
                    yield batch
                    batch = []
                    batch_size = 0
                    assay_count = 0
                for i in range(0, assay_size, self.max_batch_size):
                    yield indices[i:i + self.max_batch_size]
                continue

            # --- start a new batch if adding this assay would break EITHER limit ---
            if (batch_size + assay_size > self.max_batch_size
                    or assay_count + 1 > self.max_assays_per_batch):
                if batch:  # only yield non-empty
                    yield batch
                batch = []
                batch_size = 0
                assay_count = 0

            batch.extend(indices)
            batch_size += assay_size
            assay_count += 1

        # final partial batch
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        assays = self._get_assay_order()

        count = 0
        batch_size = 0
        assay_count = 0

        for assay in assays:
            size = len(self.assay_to_indices[assay])

            # --- oversized assay ---
            if size > self.max_batch_size:
                if batch_size > 0:      # flush pending partial
                    count += 1
                    batch_size = 0
                    assay_count = 0
                count += (size + self.max_batch_size - 1) // self.max_batch_size
                continue

            # --- new batch on either limit ---
            if (batch_size + size > self.max_batch_size
                    or assay_count + 1 > self.max_assays_per_batch):
                if batch_size > 0:
                    count += 1
                batch_size = 0
                assay_count = 0

            batch_size += size
            assay_count += 1

        if batch_size > 0 and not self.drop_last:
            count += 1

        return count


class FakeAssayBatchSampler:
    """Inter-assay training sampler: groups molecules into fake 'assays' whose sizes
    matches the real assay-size sequence (same multiset, same count), but whose
    membership is arbitrary. Isolates the effect of intra-assay training (real groups)
    vs ranking on arbitrary groups (this sampler), holding batch geometry identical.
    Sizes are fixed to the real sequence; compound membership is re-rolled each epoch.
    Within any epoch every compound appears exactly once (no duplicates, no omissions)."""

    def __init__(self, n_samples, assay_size_distribution, max_batch_size,
                 max_assays_per_batch=100, shuffle=True, seed=SEED, drop_last=False):
        self.n_samples = n_samples
        self.assay_sizes = np.asarray(assay_size_distribution, dtype=np.int64)
        self.max_batch_size = max_batch_size
        self.max_assays_per_batch = max_assays_per_batch
        self.batch_size = max_batch_size       # Lightning compatibility
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        # The real sizes must tile the data exactly, otherwise carving would drop or
        # duplicate compounds. Fail loudly rather than silently corrupt the epoch.
        total = int(self.assay_sizes.sum())
        assert total == n_samples, (
            f"assay sizes sum to {total} but n_samples={n_samples}; "
            f"fake groups would not match the data exactly"
        )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _build_fake_groups(self):
        """Assign every index to a fake group; group sizes are sampled from the real
        assay size distribution. Each epoch produces different groupings."""
        rng = random.Random(self.seed + self.epoch)

        indices = list(range(self.n_samples))
        if self.shuffle:
            rng.shuffle(indices)

        groups = []
        pos = 0
        for size in self.assay_sizes:
            size = int(size)
            groups.append(indices[pos:pos + size])
            pos += size
        # pos == n_samples here by the assertion above; no leftover to handle.
        return groups

    def __iter__(self):
        groups = self._build_fake_groups()

        batch = []
        batch_size = 0       # compounds accumulated
        group_count = 0      # fake groups (sets) accumulated

        for group in groups:
            group_size = len(group)

            # --- handle oversized group (a drawn size exceeding the compound cap) ---
            if group_size > self.max_batch_size:
                if batch:                       # flush pending first
                    yield batch
                    batch = []
                    batch_size = 0
                    group_count = 0
                for i in range(0, group_size, self.max_batch_size):
                    yield group[i:i + self.max_batch_size]
                continue

            # --- start a new batch if adding this group breaks either limit ---
            if (batch_size + group_size > self.max_batch_size
                    or group_count + 1 > self.max_assays_per_batch):
                if batch:
                    yield batch
                batch = []
                batch_size = 0
                group_count = 0

            batch.extend(group)
            batch_size += group_size
            group_count += 1

        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        # Mirror __iter__'s packing.
        count = 0
        batch_size = 0
        group_count = 0

        for size in self.assay_sizes:
            size = int(size)

            if size > self.max_batch_size:
                if batch_size > 0:
                    count += 1
                    batch_size = 0
                    group_count = 0
                count += (size + self.max_batch_size - 1) // self.max_batch_size
                continue

            if (batch_size + size > self.max_batch_size
                    or group_count + 1 > self.max_assays_per_batch):
                if batch_size > 0:
                    count += 1
                batch_size = 0
                group_count = 0

            batch_size += size
            group_count += 1

        if batch_size > 0 and not self.drop_last:
            count += 1

        return count


def pearson_correlation(y_pred, y_true):
    """Pearson r as cosine similarity of mean-centered vectors.
    Centering ensures offset-invariance; +1e-8 guards against division by zero
    for near-constant vectors (caught earlier by the std guard in ranking_loss)."""
    y_pred = y_pred - y_pred.mean()
    y_true = y_true - y_true.mean()
    r = (y_pred * y_true).sum() / (
        torch.sqrt((y_pred ** 2).sum()) * torch.sqrt((y_true ** 2).sum()) + 1e-8
    )
    return r


def ranking_loss(y_pred, y_true, assay_ids):
    """Compute 1 - mean per-assay Pearson correlation.
    Assays with fewer than 2 compounds or near-zero variance in predictions or
    targets are excluded. Returns a graph-connected loss of 1.0 if no assay
    is usable, so backpropagation does not crash."""
    correlations = []
    for assay_id in assay_ids.unique():
        mask = assay_ids == assay_id

        # Guard 1: Pearson is undefined for <2 points (0/0 -> NaN). Skip singletons.
        # Guard 2: Pearson is also degenerate if either vector has zero variance
        # (all preds identical, or all targets identical) -> meaningless ~0 value
        # propped up only by the epsilon. Skip those assays too.
        if mask.sum() > 1:
            yp, yt = y_pred[mask], y_true[mask]
            if yp.std() > 1e-8 and yt.std() > 1e-8:
                correlations.append(pearson_correlation(yp, yt))

    # Fail-safe: if NO assay in this batch was usable (all singletons or all
    # zero-variance), return a loss of 1.0 that is still connected to the graph.
    # (y_pred.sum() * 0.0) is exactly zero numerically but carries gradient history,
    # so .backward() has a valid (zero-contribution) path and training won't crash.
    # A bare torch.tensor(1.0) would have no grad_fn and break the optimizer step.
    if len(correlations) == 0:
        return (y_pred.sum() * 0.0) + 1.0

    # Loss is low when correlations are high (good intra-assay ranking).
    return 1 - torch.stack(correlations).mean()


class Pairwise(LightningModule):
    """PyTorch Lightning module for pairwise binding affinity ranking.
    Encodes ligand (ECFP6) and protein (ESM-2) features separately, concatenates
    and projects them, then applies a Set Transformer with assay-masked attention
    so each compound only attends to compounds in the same assay. Trained to
    maximise within-assay Pearson correlation via ranking_loss."""

    def __init__(self, ligand_sz, protein_sz, train_data, valid_data, test_data,
                 train_assay_ids, val_assay_ids, test_assay_ids, batch_size=2048, lr=1.12e-4,
                 training_mode=training_mode, max_assays_per_batch=100):
        super().__init__()
        self.training_mode = training_mode
        self.lr = lr
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.max_assays_per_batch = max_assays_per_batch
        self.train_assay_ids = train_assay_ids
        self.val_assay_ids = val_assay_ids
        self.test_assay_ids = test_assay_ids

        # Ligand encoder - 3 layers
        self.ligand_encoder = nn.Sequential(
            nn.Linear(ligand_sz, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512)
        )

        # Protein encoder - 2 layers
        self.protein_encoder = nn.Sequential(
            nn.Linear(protein_sz, 512),
            nn.SiLU(),
            nn.Linear(512, 512)
        )

        # Feature concatenation and reduction
        self.projection = nn.Linear(1024, 512)

        # Set transformer with assay-masked self-attention
        self.set_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=512, batch_first=True),
            num_layers=4
        )

        # Prediction head
        self.prediction = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(512, 1)
        )

    def training_step(self, batch, batch_idx):
        x_ligand, x_protein, assay_ids, y, _ = batch
        if self.training_mode == "inter_assay":
            # Each batch IS one fake "assay". Override the assay_ids so the model
            # treats the whole batch as one group for attention and loss.
            assay_ids = torch.zeros_like(assay_ids)
        z = self(x_ligand, x_protein, assay_ids)
        loss = ranking_loss(z, y, assay_ids)
        self.log("Train loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_ligand, x_protein, assay_ids, y, _ = batch
        z = self(x_ligand, x_protein, assay_ids)
        loss = ranking_loss(z, y, assay_ids)
        self.log("Validation loss", loss)

    def test_step(self, batch, batch_idx):
        x_ligand, x_protein, assay_ids, y, _ = batch
        z = self(x_ligand, x_protein, assay_ids)
        loss = ranking_loss(z, y, assay_ids)
        self.log("Test loss", loss)

    def predict_step(self, batch, batch_idx):
        x_ligand, x_protein, assay_ids, y, indices = batch
        predictions = self(x_ligand, x_protein, assay_ids)
        return predictions, indices

    def configure_optimizers(self):
        """AdamW optimizer with ReduceLROnPlateau scheduler (factor=0.5, patience=25),
        monitored on validation loss."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=25
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "Validation loss"}}

    def forward(self, x_ligand, x_protein, assay_ids):
        """Encode ligand and protein, project concatenation to 512-d, apply
        assay-masked Set Transformer, and return scalar predictions."""
        h_ligand = self.ligand_encoder(x_ligand)
        h_protein = self.protein_encoder(x_protein)
        combined = torch.cat([h_ligand, h_protein], dim=1)  # (B, 1024)
        combined = self.projection(combined)  # (B, 512)

        # Treat batch as one sequence: (1, B, 512)
        seq = combined.unsqueeze(0)

        # Mask: True = block attention. Block compound i from attending to j if different assays.
        attn_mask = (assay_ids.unsqueeze(1) != assay_ids.unsqueeze(0))  # (B, B)

        output = self.set_transformer(seq, mask=attn_mask)  # (1, B, 512)
        output = output.squeeze(0)  # (B, 512)

        return self.prediction(output).flatten()

    def on_train_epoch_start(self):
        # See note in train_dataloader: custom batch_samplers aren't auto-advanced,
        # so we push the epoch in manually to get per-epoch reshuffling.
        if hasattr(self, "_train_sampler") and hasattr(self._train_sampler, "set_epoch"):
            self._train_sampler.set_epoch(self.current_epoch)

    def train_dataloader(self):
        if self.training_mode == "intra_assay":
            sampler = AssayBatchSampler(
                self.train_data.assay_ids, self.batch_size,
                max_assays_per_batch=self.max_assays_per_batch,
                shuffle=True, drop_last=False)
        elif self.training_mode == "inter_assay":
            _, real_sizes = np.unique(self.train_data.assay_ids, return_counts=True)
            sampler = FakeAssayBatchSampler(
                n_samples=len(self.train_data),
                assay_size_distribution=real_sizes,
                max_batch_size=self.batch_size,
                max_assays_per_batch=self.max_assays_per_batch,
                shuffle=True, drop_last=False)
        else:
            raise ValueError(f"Unknown training_mode: {self.training_mode}")

        self._train_sampler = sampler   # keep a handle for set_epoch
        return DataLoader(self.train_data,
                          batch_sampler=sampler,
                          num_workers=6, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.valid_data,
                          batch_sampler=AssayBatchSampler(self.valid_data.assay_ids, self.batch_size,
                                                          shuffle=False, drop_last=False),
                          num_workers=6, pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_data,
                          batch_sampler=AssayBatchSampler(self.test_data.assay_ids, self.batch_size,
                                                          shuffle=False, drop_last=False),
                          num_workers=6, pin_memory=True, persistent_workers=True)


if __name__ == "__main__":

    start = time.time()

    # --- Load Data ---

    df = pd.read_csv(os.path.join(DATA_DIR, f"{dataset_name}_dataset.csv"))
    df = df.rename(columns={
        "Canonical_smile": "smiles",
        "pIC50": "pIC50",
        "Assay_id": "assay_id"
    })

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)

    ligand_fingerprints = []
    valid_mask = []
    for smi in df["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            valid_mask.append(False)
            continue
        valid_mask.append(True)
        ligand_fingerprints.append(morgan_gen.GetFingerprintAsNumPy(mol))

    ligand_fingerprints = np.array(ligand_fingerprints, dtype=np.uint8)
    valid_mask = np.array(valid_mask)
    df = df[valid_mask].reset_index(drop=True)
    print(f"Dropped {(~valid_mask).sum()} rows with invalid SMILES")
    print(" === Ligand fingerprints computed === ")
    gc.collect()

    protein_features = np.load(os.path.join(FEATURE_DIR, f"{dataset_name}_protein_feature_vector.npy"),
                               allow_pickle=True)

    if dataset_name == "omnivore":
        # Halve memory footprint for the largest dataset; cast back to float32 at training time.
        protein_features = np.array(protein_features, dtype=np.float16)[valid_mask]
    else:
        protein_features = np.array(protein_features)[valid_mask]

    print(f" === Protein features loaded ({protein_features.dtype}) === ")

    df = df[["smiles", "assay_id", "pIC50"]].reset_index(drop=True)
    df["ligand"] = list(ligand_fingerprints)  # store as objects for split indexing
    df["protein"] = list(protein_features)

    # Free the standalone arrays — df now owns this data
    del ligand_fingerprints, protein_features
    gc.collect()

    X = df[["assay_id", "ligand", "protein"]]
    y = df["pIC50"].values

    print(" === X and y array prepared === ")

    summary_rows = []

    for i, (train_split, val_split, test_split) in enumerate(
            assay_split(X, y, test_size=0.2, val_size=0.1, random_state=SEED)):

        # --- Encode assay IDs (string to integers) ---

        d = {}

        split_names = ["train", "val", "test"]
        splits = [train_split, val_split, test_split]

        for name, subset in zip(split_names, splits):
            d[f"ligand_{name}"] = subset["ligand"]
            d[f"protein_{name}"] = subset["protein"]
            d[f"y_{name}"] = subset["pIC50"]
            d[f"assay_{name}"] = subset["assay_ids"]

        le = LabelEncoder()
        all_assay = np.concatenate([d[f"assay_{name}"] for name in split_names])
        le.fit(all_assay)

        for name, subset in zip(split_names, splits):
            d[f"assay_{name}"] = le.transform(d[f"assay_{name}"]).astype(np.int64)

        print(f"train compounds: {len(d['y_train'])}, "
              f"train assays: {len(np.unique(d['assay_train']))}, "
              f"val: {len(d['y_val'])}, test: {len(d['y_test'])}")

        # --- Extract arrays after encoding ---

        ligand_train = d["ligand_train"]
        ligand_val = d["ligand_val"]
        ligand_test = d["ligand_test"]

        protein_train = d["protein_train"]
        protein_val = d["protein_val"]
        protein_test = d["protein_test"]

        y_test = d["y_test"]
        test_assay_ids = d["assay_test"]

        # --- Build Dataset ---

        pairwise_train = Pairwise_data(ligand_train, protein_train, d["y_train"], d["assay_train"])
        pairwise_val = Pairwise_data(ligand_val, protein_val, d["y_val"], d["assay_val"])
        pairwise_test = Pairwise_data(ligand_test, protein_test, d["y_test"], d["assay_test"])

        # --- Initiate model ---

        pairwise_model = Pairwise(
            ligand_sz=ligand_train.shape[1],
            protein_sz=1280,
            train_data=pairwise_train,
            valid_data=pairwise_val,
            test_data=pairwise_test,
            train_assay_ids=d["assay_train"],
            val_assay_ids=d["assay_val"],
            test_assay_ids=d["assay_test"],
            training_mode=training_mode,
            max_assays_per_batch=100
        )

        # ---- Train ----

        checkpoint_callback = ModelCheckpoint(
            monitor="Validation loss",
            save_top_k=1,
            mode="min"
        )
        
        trainer = Trainer(
            max_epochs=500,
            logger=CSVLogger(save_dir = os.path.join(SCRIPT_DIR, "logs"), name=f"Pairwise_{dataset_name}_split_{i}_{training_mode}"),
            accelerator=get_accel(),
            callbacks=[
                checkpoint_callback,
                EarlyStopping(monitor="Validation loss", patience=50, check_on_train_epoch_end=False),
                TQDMProgressBar(refresh_rate=100)
            ],
            #replace_sampler_ddp=False,     # use only in older version of pytorch
        )

        trainer.fit(pairwise_model)
        best_model = Pairwise.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            ligand_sz=2048,
            protein_sz=pairwise_train.protein.shape[1],
            train_data=pairwise_train,
            valid_data=pairwise_val,
            test_data=pairwise_test,
            train_assay_ids=d["assay_train"],
            val_assay_ids=d["assay_val"],
            test_assay_ids=d["assay_test"],
            training_mode=training_mode
        )

        # ---- Evaluate (manual loop, bypasses problematic Lightning predict path) ----

        best_model.eval()
        best_model.cuda()

        predictions = []

        with torch.no_grad():
            for batch in best_model.test_dataloader():
                x_ligand, x_protein, assay_ids, y, indices = batch
                x_ligand = x_ligand.cuda()
                x_protein = x_protein.cuda()
                assay_ids = assay_ids.cuda()
                preds = best_model(x_ligand, x_protein, assay_ids)
                predictions.append((preds.cpu(), indices))

        # Reconstruct predictions in original test-set order
        y_pred_dict = {}
        for preds, indices in predictions:
            preds = preds.numpy()
            indices = indices.numpy()
            for pred, idx in zip(preds, indices):
                y_pred_dict[idx] = pred

        y_pred = np.array([y_pred_dict[i] for i in range(len(y_test))])

        # ---- Per-assay aggregated correlation (Fisher-z weighted) ----

        weighted_sum = 0.0
        weight_total = 0
        for assay_id in np.unique(test_assay_ids):
            mask = test_assay_ids == assay_id
            n = mask.sum()
            if n > 3 and np.std(y_pred[mask]) > 1e-8 and np.std(y_test[mask]) > 1e-8:
                r, _ = pearsonr(y_test[mask], y_pred[mask])

                if np.isnan(r) or r > 0.9999 or r < -0.9999:
                    print(
                        f"Assay {assay_id} has n={n} and r={r:.3f} with std(y_pred)={np.std(y_pred[mask])} and std(y_test)={np.std(y_test[mask])}")
                    print("y_pred values:", y_pred[mask], "y_test values:", y_test[mask])

                if r > 0.9999 or r < -0.9999 or np.isnan(r):
                    continue

                weighted_sum += (n - 3) * np.arctanh(r)
                weight_total += (n - 3)

        rho_total = np.tanh(weighted_sum / weight_total) if weight_total > 0 else np.nan

        # ---- Global metrics ----

        pearson, _ = pearsonr(y_test, y_pred)

        print(f"for split number {i + 1} | rho_global={pearson:.3f} | rho_assay={rho_total:.3f}")

        # --- Save per-compound predictions ---
        pred_df = pd.DataFrame({
            "y_true": y_test,
            "y_pred": y_pred,
            "assay_id": test_assay_ids,
        })
        pred_df.to_csv(os.path.join(RESULT_DIR,f"{dataset_name}/{dataset_name}_assay_split{i}_pairwise_{training_mode}_predictions.csv"), index=False)

        # --- Save summary metrics ---
        summary_rows.append({
            "dataset": dataset_name,
            "split": i,
            "rho_global": pearson,
            "rho_assay_aggregated": rho_total,
        })

        # --- Mean baseline (predicts training assay mean per test assay) ---
        y_train = d["y_train"]
        assay_train_arr = d["assay_train"]
        train_mean = y_train.mean()
        mean_per_assay = {a: y_train[assay_train_arr == a].mean()
                          for a in np.unique(assay_train_arr)}
        y_pred_meanbaseline = np.array([mean_per_assay.get(a, train_mean)
                                        for a in test_assay_ids])

        mean_df = pd.DataFrame({
            "y_true": y_test,
            "y_pred": y_pred_meanbaseline,
            "assay_id": test_assay_ids,
        })
        mean_df.to_csv(os.path.join(RESULT_DIR,f"{dataset_name}//{dataset_name}_assay_split{i}_meanbaseline_{training_mode}.csv"), index=False)

        # Free split-specific data before next iteration (needed to prevent omnivore
        # from crashing due to memory shortage)
        del pairwise_train, pairwise_val, pairwise_test
        del pairwise_model, best_model, predictions, y_pred_dict, y_pred
        del d, train_split, val_split, test_split
        gc.collect()
        torch.cuda.empty_cache()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(RESULT_DIR,f"{dataset_name}/{dataset_name}_pairwise_{training_mode}_summary.csv"), index=False)

    end = time.time()
    print(f"Time taken {(end - start) / 60:.2f} minutes")