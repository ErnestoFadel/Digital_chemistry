================================================================================
BINDING AFFINITY RANKING VS. ABSOLUTE PREDICTION
================================================================================

This project investigates binding affinity prediction between small
molecules and protein kinase targets, comparing two framings: predicting
absolute pIC50 values (regression) versus ranking compounds within an
assay (correlation-based ranking). The pairwise ranking model is evaluated
under two training modes, intra-assay (real assay groups) and inter-assay
(randomly shuffled groups of matching sizes), to isolate the effect of
assay-aware grouping. A mean-prediction baseline and several data-splitting
strategies (random, scaffold, target, Butina, assay) are also implemented
for comparison.


--------------------------------------------------------------------------------
PROJECT STRUCTURE
--------------------------------------------------------------------------------

project/
|-- Data/
|   |-- Csv_files/
|   |   |-- human_kinases_and_chembl_targets.chembl_33.csv
|   |   |-- kinodata_dataset.csv
|   |   |-- landrum_dataset.csv
|   |   |-- omnivore_dataset.csv
|   |   `-- omnivore_short_dataset.csv
|   |-- assay_size_filtering.py      (builds omnivore_short)
|   |-- Data_curation.ipynb
|   `-- Data_from_sql.ipynb
|
|-- Models/
|   |-- logs/                        (training logs, CSVLogger output)
|   |-- protein_featurization/
|   |   |-- Encoded_protein/         (precomputed ESM-2 embeddings, .npy)
|   |   `-- Protein_encoder.py       (computes ESM-2 embeddings)
|   |-- Data_split.py                (random, scaffold, target, Butina, assay)
|   |-- DTI_model.py                 (baseline absolute-prediction model)
|   |-- Mean_Prediction_model.py     (mean-per-group baseline)
|   |-- Pairwise_model.py            (Set Transformer ranking model)
|   `-- pairwise_requirements.txt
|
`-- Results/
    |-- Figure_generation/           (plotting scripts for all figures)
    |-- hyperparameter optimization/ (training metrics from LR/batch sweeps)
    |-- kinodata/
    |-- landrum/
    |-- omnivore/                    (same substructure for all datasets)
    |   |-- Pairwise_predictions/    
    |   |-- DTI_results_omnivore.csv
    |   |-- mean_results_omnivore.csv
    |   |-- omnivore_pairwise_inter_assay_summary.csv
    |   `-- omnivore_pairwise_intra_assay_summary.csv
    `-- omnivore_short/

In order to obtain the full structure clone the Github repository and run 'get_data.py'. 
It will download the used dataset and encoded proteins from hugging face.

--------------------------------------------------------------------------------
MODELS
--------------------------------------------------------------------------------

DTI Model (DTI_model.py)
    A standard regression baseline that predicts absolute pIC50 values.
    Ligand ECFP fingerprints and ESM-2 protein embeddings are encoded by
    separate MLPs, concatenated, and passed through an interaction MLP
    with BatchNorm. Trained with MSE loss.

Pairwise Model (Pairwise_model.py)
    A Set Transformer model that ranks compounds within assays. Ligand
    and protein features are encoded separately, concatenated, projected
    to 512 dimensions, and passed through a 4-layer Transformer Encoder
    with assay-masked attention so molecules can only attend to others
    from the same assay. Trained with a ranking loss of 1 - rho,
    where rho is the mean per-assay Pearson correlation.

    Has two training modes, set via `training_mode` at the top of
    the script:
      - "intra_assay": real assay groups, assay-masked attention
      - "inter_assay": randomly shuffled groups matching the real
        assay-size distribution, isolating the effect of assay identity
        from batch composition

Mean-Prediction Baseline (Mean_Prediction_model.py)
    Predicts the training mean pIC50 per assay, target, or scaffold,
    falling back to the global training mean for unseen groups.
    Evaluated across all four split strategies for each predictive mode.


--------------------------------------------------------------------------------
DATA
--------------------------------------------------------------------------------

Each dataset CSV (kinodata_dataset.csv, landrum_dataset.csv,
omnivore_dataset.csv, omnivore_short_dataset.csv) should contain:

    Canonical_smile    SMILES string for the ligand
    Protein_sequence   protein sequence, used for ESM-2 embeddings
    pIC50              binding affinity target value
    Assay_id           assay identifier
    Target_id          protein target identifier (used for target split)

omnivore_short is derived from omnivore by Data/assay_size_filtering.py,
which drops all assays with fewer than 3 compounds while keeping the CSV
and protein .npy array row-aligned.

Protein embeddings are precomputed ESM-2 vectors
(facebook/esm2_t33_650M_UR50D, mean-pooled, 1280-dim) generated by
Models/protein_featurization/Protein_encoder.py and stored as .npy files
in Models/protein_featurization/Encoded_protein/.


--------------------------------------------------------------------------------
SPLITS
--------------------------------------------------------------------------------

Five splitting strategies are implemented in Models/Data_split.py:

    Random
        Shuffled 10-fold cross-validation.

    Scaffold
        Murcko scaffold-based; no scaffold appears in both train and test.

    Target
        All compounds sharing a protein target go to the same partition,
        testing generalization to unseen proteins.

    Butina
        Chemical similarity clustering (Tanimoto, distance 0.35).

    Assay
        Rotating assay-group scheme yielding 5 train/val/test folds,
        where all compounds from an assay always appear in the same
        partition. Used by the DTI and pairwise models.


--------------------------------------------------------------------------------
SETUP
--------------------------------------------------------------------------------

Hardware note
    The pairwise model was developed for an NVIDIA GTX 980 Ti (Maxwell
    architecture, compute capability 5.2). PyTorch 2.6+ dropped support
    for this GPU. The last compatible version is PyTorch 2.5.0.

    For a modern GPU, any recent PyTorch version should work.

Installation

    1. Install PyTorch manually first to ensure the correct CUDA build:

       # For GTX 980 Ti (compute capability 5.2)
       pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
           --index-url https://download.pytorch.org/whl/cu124

       # For modern GPUs
       pip install torch torchvision torchaudio \
           --index-url https://download.pytorch.org/whl/cu124

    2. Install remaining dependencies:

       pip install -r Models/pairwise_requirements.txt

    3. Install extra packages needed for protein embedding:

       pip install transformers tqdm

pairwise_requirements.txt
    torch==2.5.0
    pytorch-lightning==2.4.0
    numpy==1.24.4
    pandas==2.0.3
    scipy==1.11.4
    scikit-learn==1.3.2
    rdkit==2023.9.4


--------------------------------------------------------------------------------
RUNNING THE PIPELINE
--------------------------------------------------------------------------------

1. Generate protein embeddings (only needed once per dataset):

       cd Models/protein_featurization
       python Protein_encoder.py

2. (Optional) Build omnivore_short from omnivore, dropping assays with
   fewer than 3 compounds:

       cd Data
       python assay_size_filtering.py

3. Train a model. Set `dataset_name` and (for the pairwise model)
   `training_mode` at the top of the script before running:

       cd Models
       python DTI_model.py
       python Pairwise_model.py
       python Mean_Prediction_model.py

   Results are saved as CSVs in the corresponding Results/<dataset_name>/
   directory. Training logs are saved under Models/logs/.

4. Generate figures:

       cd Results/Figure_generation
       python Assay_Size_Histogram.py
       python LR_Plots.py
       python model_Pearson_comparison.py
       python plots_mean_activity_predictions.py


--------------------------------------------------------------------------------
EVALUATION
--------------------------------------------------------------------------------

Models are evaluated using:

    - Global Pearson correlation (rho_global)
        Computed across all test compounds.

    - Fisher-z weighted per-assay Pearson correlation (rho_assay)
        Aggregated across assays, weighted by (n_i - 3) to account for
        sample size.

    - RMSE
        Used for the DTI and mean baseline models.

    - Kendall's Tau
        Used for the mean baseline.

Assays with fewer than 4 compounds, or near-zero variance in predictions
or targets, are excluded from per-assay metrics, since Pearson
correlation is undefined or degenerate in those cases. Comparisons
between models use paired t-tests on Fisher z-transformed correlations
across cross-validation folds.