"""
Dataset splitting utilities for cross-validation and train/val/test splits.
Provides random, scaffold, target, Butina (chemical similarity), and assay-based
splits. All CV splits use 10 folds by default; assay_split yields 5 train/val/test
folds using a rotating assay-group scheme.
"""

from collections import defaultdict
from multiprocessing import Pool
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import BulkTanimotoSimilarity
import datamol as dm 

# Seed used for functions requiring randomness
SEED = 42


def random_split_mean(X, y, random_state=SEED, n_split=10):
    """Split data into n_split folds by randomly shuffling sample indices.

    Yields
    ------
    X_train, y_train, X_test, y_test for each fold.
    """
    np.random.seed(random_state)
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // n_split

    for i in range(n_split):
        test_idx = indices[i * fold_size:(i + 1) * fold_size]
        train_idx = np.setdiff1d(indices, test_idx)
        yield X[train_idx], y[train_idx], X[test_idx], y[test_idx] 


def scaffold_split_mean(X, y, random_state=SEED, n_split=10):
    """Split data into n_split folds by Murcko scaffold, keeping scaffolds intact
    within each fold. Folds are balanced by compound count (greedy smallest-fold).

    Yields
    ------
    X_train, y_train, X_test, y_test for each fold.
    """

    # Group sample indices by scaffold
    scaffold_dict = defaultdict(list)
    for idx, scaffold in enumerate(X[:, 3]):
        scaffold_dict[scaffold].append(idx)

    scaffold_groups = sorted(scaffold_dict.values(), key=len, reverse=True)

    rng = np.random.default_rng(random_state)
    rng.shuffle(scaffold_groups)

    # Assign scaffold groups to folds (greedy: always extend the smallest fold)
    folds = [[] for _ in range(n_split)]
    for group in scaffold_groups:
        smallest_fold = min(range(n_split), key=lambda i: len(folds[i]))
        folds[smallest_fold].extend(group)

    for i in range(n_split):
        test_idx = np.array(folds[i])
        train_idx = np.array(
            [idx for j, fold in enumerate(folds) if j != i for idx in fold]
        )
        yield X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def target_split_mean(X, y, random_state=SEED, n_split=10):
    """Split data into n_split folds by protein target, keeping targets intact
    within each fold. Folds are balanced by compound count (greedy smallest-fold).

    Yields
    ------
    X_train, y_train, X_test, y_test for each fold.
    """

    # Group sample indices by target
    target_dict = defaultdict(list)
    for idx, target in enumerate(X[:, 2]):
        target_dict[target].append(idx)

    target_groups = sorted(target_dict.values(), key=len, reverse=True)

    rng = np.random.default_rng(random_state)
    rng.shuffle(target_groups)

    # Assign target groups to folds (greedy: always extend the smallest fold)
    folds = [[] for _ in range(n_split)]
    for group in target_groups:
        smallest_fold = min(range(n_split), key=lambda i: len(folds[i]))
        folds[smallest_fold].extend(group)

    for i in range(n_split):
        test_idx = np.array(folds[i])
        train_idx = np.array(
            [idx for j, fold in enumerate(folds) if j != i for idx in fold]
        )
        yield X[train_idx], y[train_idx], X[test_idx], y[test_idx]


# --- Butina clustering helpers ---

_FPS = None


def _init_worker(fps):
    """Initialise the global fingerprint list for multiprocessing workers."""
    global _FPS
    _FPS = fps


def _neighbors_of(args):
    """Return (i, neighbours) where neighbours are indices with Tanimoto >= sim_thresh."""
    i, sim_thresh = args
    if i == 0:
        return 0, []
    sims = BulkTanimotoSimilarity(_FPS[i], _FPS[:i])
    return i, [j for j, s in enumerate(sims) if s >= sim_thresh]


def butina_cluster_fps(fps, dist_thresh=0.35, n_workers=8, chunksize=32):
    """Cluster fingerprints using the Butina algorithm with parallel Tanimoto
    computation. Returns a tuple of clusters, each cluster being a tuple of indices.

    Parameters
    ----------
    fps : list of RDKit fingerprints
    dist_thresh : float
        Maximum Tanimoto distance to consider two molecules neighbours (default 0.35).
    n_workers : int
        Number of parallel worker processes.
    chunksize : int
        Task chunksize for imap_unordered.
    """
    n = len(fps)
    sim_thresh = 1.0 - dist_thresh
    nbr_lists = [[] for _ in range(n)]

    with Pool(n_workers, initializer=_init_worker, initargs=(fps,)) as pool:
        tasks = [(i, sim_thresh) for i in range(n)]
        for i, nbrs in tqdm(
            pool.imap_unordered(_neighbors_of, tasks, chunksize=chunksize),
            total=n,
            desc="Neighbors",
            smoothing=0.05,
            mininterval=5.0,
        ):
            for j in nbrs:
                nbr_lists[i].append(j)
                nbr_lists[j].append(i)

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
        clusters.append(tuple(cluster))
    return tuple(clusters)


def butina_split_mean(X, y, n_split=10, random_state=SEED):
    """Split data into n_split folds by Butina chemical similarity clustering,
    keeping clusters intact within each fold. Folds are balanced by compound count.

    Yields
    ------
    X_train, y_train, X_test, y_test for each fold.
    """
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    n = len(y)

    fps_bv = [morgan_gen.GetFingerprint(Chem.MolFromSmiles(X[i, 1])) for i in range(n)] 

    clusters = sorted(butina_cluster_fps(fps_bv), key=len, reverse=True)

    rng = np.random.default_rng(random_state)
    rng.shuffle(clusters)

    # Assign clusters to folds (greedy: always extend the smallest fold)
    folds = [[] for _ in range(n_split)]
    for group in clusters:
        smallest_fold = min(range(n_split), key=lambda i: len(folds[i]))
        folds[smallest_fold].extend(group)

    for i in range(n_split):
        test_idx = np.array(folds[i])
        train_idx = np.array(
            [idx for j, fold in enumerate(folds) if j != i for idx in fold]
        )
        yield X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def assay_split(X, y, test_size=0.2, val_size=0.1, random_state=SEED, n_split=5):
    """Split data into n_split train/val/test folds by assay group. Assay groups are
    shuffled once and then rotated by (n_assays // n_split) positions each fold,
    so each fold gets a different block of assays as its test set. All compounds
    from a given assay always appear in the same partition (train, val, or test).

    Yields
    ------
    train_set, val_set, test_set : dicts with keys ligand, protein, pIC50, assay_ids.
    """
    assay_groups = X.groupby("assay_id").indices
    assay_groups = sorted(assay_groups.values(), key=len, reverse=True)

    train_cutoff = (1 - val_size - test_size) * len(y)
    val_cutoff = (1 - test_size) * len(y)

    rng = np.random.default_rng(random_state)
    rng.shuffle(assay_groups)

    block = len(assay_groups) // n_split  # rotate by this many assays per fold

    ligand = X["ligand"].to_numpy()
    protein = X["protein"].to_numpy()
    assay = X["assay_id"].to_numpy()

    for i in range(n_split):
        # Rotate assay groups so each fold has a different test slice
        rotated = assay_groups[i * block:] + assay_groups[:i * block]

        train_idx, val_idx, test_idx = [], [], []
        for group in rotated:
            if len(train_idx) < train_cutoff:
                train_idx.extend(group)
            elif len(train_idx) + len(val_idx) < val_cutoff:
                val_idx.extend(group)
            else:
                test_idx.extend(group)

        yield (
            {
                "ligand": np.stack(ligand[train_idx]),
                "protein": np.stack(protein[train_idx]),
                "pIC50": y[train_idx],
                "assay_ids": assay[train_idx],
            },
            {
                "ligand": np.stack(ligand[val_idx]),
                "protein": np.stack(protein[val_idx]),
                "pIC50": y[val_idx],
                "assay_ids": assay[val_idx],
            },
            {
                "ligand": np.stack(ligand[test_idx]),
                "protein": np.stack(protein[test_idx]),
                "pIC50": y[test_idx],
                "assay_ids": assay[test_idx],
            },
        )