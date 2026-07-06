"""
Compute ESM-2 protein embeddings for a dataset's unique protein sequences,
mean-pooled over residues using the attention mask, then expand back to the
full dataset row order and save as a single .npy feature array.
""" 

import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmModel
from tqdm import tqdm

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True

# --- Config ---

# Dataset to encode
DATASET_NAME = "omnivore"

# Define base directory (where the script lives)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(MODEL_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "Csv_files")  
FEATURE_DIR = os.path.join(SCRIPT_DIR, "Encoded_protein")

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
EMBEDDING_DIM = 1280
BATCH_SIZE = 4  # FP32 + ESM-650M on 8 GB


def main():
    """Load the dataset, deduplicate protein sequences, embed each unique
    sequence with ESM-2, mean-pool over residues, then expand the per-
    sequence embeddings back to the full dataset and save to disk."""
    # --- Load data ---
    data = pd.read_csv(os.path.join(DATA_DIR, f"{DATASET_NAME}_dataset.csv"))

    # --- Deduplicate ---
    unique_seqs = data["Protein_sequence"].drop_duplicates().tolist()
    print(f"Total rows: {len(data)}, unique sequences: {len(unique_seqs)}")

    # Sort by length for efficient dynamic padding
    unique_seqs_sorted = sorted(unique_seqs, key=len)

    # --- Model ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME).to(device).eval()

    # --- Embed unique sequences ---
    seq_to_emb = {}
    with torch.inference_mode():
        for i in tqdm(range(0, len(unique_seqs_sorted), BATCH_SIZE)):
            batch_seqs = unique_seqs_sorted[i:i + BATCH_SIZE]
            enc = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)

            out = model(input_ids=ids, attention_mask=mask).last_hidden_state
            m = mask.unsqueeze(-1).float()
            pooled = ((out * m).sum(1) / m.sum(1).clamp(min=1e-9)).cpu().numpy()

            for seq, emb in zip(batch_seqs, pooled):
                seq_to_emb[seq] = emb

    # --- Expand to full dataset ---
    full_emb = np.stack([seq_to_emb[s] for s in data["Protein_sequence"]])

    # --- Save embeddings to the feature directory ---
    out_path = os.path.join(
        FEATURE_DIR, f"{DATASET_NAME}_protein_feature_vector.npy"
    )
    np.save(out_path, full_emb)
    print(f"Saved: shape={full_emb.shape}")


if __name__ == "__main__":
    main()