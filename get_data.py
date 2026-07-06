"""
Downloads heavy files from the HF dataset repo and places them
into the local project structure:
  Data/Csv_files/
  Models/protein_featurization/Encoded_protein/
"""

from pathlib import Path
from huggingface_hub import snapshot_download

HF_REPO_ID = "ernfad/Digital_chemistry_dataset"  
REPO_ROOT = Path(__file__).resolve().parent.parent  # project/ root

# Download everything from the HF repo into a local cache-controlled folder
downloaded_path = snapshot_download(
    repo_id=HF_REPO_ID,
    repo_type="dataset",
    allow_patterns=["Csv_files/*", "Encoded_protein/*"],
)

downloaded_path = Path(downloaded_path)

# Map HF folder -> local project folder
mapping = {
    "Csv_files": REPO_ROOT / "Data" / "Csv_files",
    "Encoded_protein": REPO_ROOT / "Models" / "protein_featurization" / "Encoded_protein",
}

import shutil

for hf_folder, local_target in mapping.items():
    src = downloaded_path / hf_folder
    if not src.exists():
        print(f"Warning: {src} not found in HF repo, skipping.")
        continue
    local_target.mkdir(parents=True, exist_ok=True)
    for file in src.iterdir():
        shutil.copy2(file, local_target / file.name)
    print(f"Copied {hf_folder} -> {local_target}")

print("Done. Heavy files are in place.")