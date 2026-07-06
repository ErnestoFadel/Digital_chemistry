"""
Generate representative molecule images per split type for a poster.
Produces transparent-background PNGs grouped by split method (random,
scaffold, Butina, assay), each showing molecules from the largest
group/cluster/scaffold/assay found in the dataset.
"""

import os
from collections import defaultdict
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity 

# --- Config ---

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "Csv_files")
INPUT_CSV = os.path.join(DATA_DIR, "kinodata_dataset.csv")
OUTPUT_DIR = SCRIPT_DIR

N_MOLS = 6  # how many molecules to draw per cluster/scaffold/assay
SUBSAMPLE_FOR_BUTINA = (
    5000  # Butina is slow to run. Using smaller group for visualization purposes.
)
IMG_SIZE = (400, 400)
RANDOM_SEED = 42


def draw_transparent(mol, path, size=IMG_SIZE):
    """Render a molecule to a transparent-background PNG at the given path."""  
    drawer = Draw.rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    drawer.drawOptions().clearBackground = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    with open(path, "wb") as f:
        f.write(drawer.GetDrawingText())


def save_n_from_indices(indices, df, out_subdir, label, n=N_MOLS, rng=None):
    """Randomly sample up to n molecules from indices and save each as a
    transparent PNG in out_subdir."""
    os.makedirs(out_subdir, exist_ok=True)
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    picks = rng.choice(indices, size=min(n, len(indices)), replace=False)
    for slot, idx in enumerate(picks):
        mol = Chem.MolFromSmiles(df.iloc[idx]["smiles"])
        if mol is None:
            continue
        draw_transparent(mol, os.path.join(out_subdir, f"mol{slot}.png"))
    print(f"  [{label}] saved {len(picks)} mols to {out_subdir}/")


def export_random():
    """Export N_MOLS molecules chosen uniformly at random from the dataset."""
    picks = rng.choice(len(df), size=N_MOLS, replace=False)
    save_n_from_indices(
        picks, df, os.path.join(OUTPUT_DIR, "random"), "random", rng=rng
    )


def export_scaffold():
    """Export N_MOLS molecules from the largest Murcko scaffold group."""
    scaffolds = defaultdict(list)
    for i, smi in enumerate(df["smiles"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=False
            )
        except Exception:
            continue
        if scaf:
            scaffolds[scaf].append(i)
    biggest = max(scaffolds.items(), key=lambda kv: len(kv[1]))
    print(f"  scaffold: '{biggest[0]}' with {len(biggest[1])} compounds")
    save_n_from_indices(
        biggest[1], df, os.path.join(OUTPUT_DIR, "scaffold"), "scaffold", rng=rng
    )


def export_butina():
    """Export N_MOLS molecules from the largest Butina cluster, computed on
    a subsample of the dataset for speed (Butina clustering is expensive)."""
    sub_idx = rng.choice(
        len(df), size=min(SUBSAMPLE_FOR_BUTINA, len(df)), replace=False
    )
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)

    fps, fps_to_df = [], []
    for idx in sub_idx:
        mol = Chem.MolFromSmiles(df.iloc[idx]["smiles"])
        if mol is None:
            continue
        fps.append(gen.GetFingerprint(mol))
        fps_to_df.append(idx)

    n = len(fps)
    sim_thresh = 0.65
    nbr_lists = [[] for _ in range(n)]
    for i in range(1, n):
        sims = BulkTanimotoSimilarity(fps[i], fps[:i])
        for k, s in enumerate(sims):
            if s >= sim_thresh:
                nbr_lists[i].append(k)
                nbr_lists[k].append(i)

    # Greedy cluster assignment: highest-degree nodes become centroids first
    order = sorted(range(n), key=lambda k: len(nbr_lists[k]), reverse=True)
    clusters, seen = [], [False] * n
    for idx in order:
        if seen[idx]:
            continue
        cluster = [idx]
        seen[idx] = True
        for nbr in nbr_lists[idx]:
            if not seen[nbr]:
                cluster.append(nbr)
                seen[nbr] = True
        clusters.append(cluster)

    biggest = max(clusters, key=len)
    real_indices = [fps_to_df[i] for i in biggest]
    print(f"  butina: largest cluster has {len(biggest)} compounds")
    save_n_from_indices(
        real_indices, df, os.path.join(OUTPUT_DIR, "butina"), "butina", rng=rng
    )


def export_assay():
    """Export N_MOLS molecules from the largest assay in the dataset."""
    by_assay = df.groupby("assay_id").indices
    biggest_id, biggest_idx = max(by_assay.items(), key=lambda kv: len(kv[1]))
    print(f"  assay: id={biggest_id} with {len(biggest_idx)} compounds")
    save_n_from_indices(
        biggest_idx, df, os.path.join(OUTPUT_DIR, "assay"), "assay", rng=rng
    )


# --- Load ---

df = pd.read_csv(INPUT_CSV)
df = df.rename(columns={"Canonical_smile": "smiles", "Assay_id": "assay_id"})
df = df.dropna(subset=["smiles"]).reset_index(drop=True)
print(f"Loaded {len(df)} rows")

os.makedirs(OUTPUT_DIR, exist_ok=True)
rng = np.random.default_rng(RANDOM_SEED)


# --- Run ---

if __name__ == "__main__":
    print("Random:")
    export_random()
    print("Scaffold:")
    export_scaffold()
    print("Butina:")
    export_butina()
    print("Assay:")
    export_assay()
    print(f"\nDone -> {OUTPUT_DIR}/")