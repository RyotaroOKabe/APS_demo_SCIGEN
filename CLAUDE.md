# SCIGEN — Claude Code Project Guide

## Project Overview

SCIGEN (Structural Constraint Integration in GENerative model) is a diffusion-based generative model for crystal structure discovery with geometric constraints (kagome, honeycomb, triangular, etc.). Published in Nature Materials 2025.

**Current goal:** Convert this research codebase into a Google Colab-friendly tutorial that demonstrates material generation using pretrained models.

## Before Making Major Changes

1. **Read `README.md` first** — it is the authoritative source for project intent, dependencies, and workflows.
2. **Inspect the repository structure** before proposing changes. Understand what exists.
3. **Do not rewrite the core codebase.** Add Colab-specific files (notebooks, helper scripts) alongside the existing code.

## Repository Structure

```
scigen/              # Core package: diffusion model, data modules, utilities
  pl_modules/        # PyTorch Lightning model definitions (diffusion_w_type.py is key)
  pl_data/           # Dataset and datamodule classes
  common/            # Shared utilities, constants (utils.py loads .env at import time!)
  run.py             # Training entry point
script/              # Generation, evaluation, CIF conversion scripts
  generation.py      # Main generation script (called by gen_mul.py)
  save_cif.py        # Convert outputs to CIF files
  eval_screen.py     # Pre-screening filters
  sc_utils.py        # Structural constraint definitions
  gen_utils.py       # SampleDataset class (loads ./data/kde_bond.pkl at module level!)
conf/                # Hydra YAML configs (model, data, training, optim)
gnn_eval/            # GNN-based evaluation models (stability, magnetism)
data/                # Dataset files (mp_20 train/val/test CSVs, kde_bond.pkl)
gen_mul.py           # Top-level generation launcher
config_scigen_template.py  # Config template for generation
notebooks/           # Colab tutorial notebooks
  tutorial_colab.ipynb
scripts/             # Colab helper scripts
  download_models.py # Download pretrained models from Figshare API
requirements-colab.txt  # Minimal Colab dependencies
```

## Key Workflows

### Generation (primary focus for Colab)
1. Download pretrained model via `python scripts/download_models.py` (uses Figshare API)
2. Model goes into `models/mp_20/` (contains `*.ckpt`, `hparams.yaml`, `lattice_scaler.pt`, `prop_scaler.pt`)
3. Set env vars: PROJECT_ROOT, HYDRA_JOBS, WANDB_DIR
4. Run `python gen_mul.py` or directly `python script/generation.py --model_path ... --sc kag ...`
5. Convert to CIF: `python script/save_cif.py --label <out_name>`

### Training (secondary — avoid for tutorial unless needed)
- `python scigen/run.py data=mp_20 model=diffusion_w_type expname=<name>`

## Colab Migration Priorities

1. **Inference/generation first.** The tutorial should demonstrate generating materials with pretrained weights.
2. **Use pretrained models.** Download from Figshare — do not retrain.
3. **Minimize dependencies.** The full dependency list is heavy. For Colab, identify the minimum set needed for generation only.
4. **Environment setup:** Python 3.9+, PyTorch with CUDA, torch-geometric, pymatgen, hydra-core, einops, pyxtal, smact.
5. **`.env` setup:** Must set PROJECT_ROOT, HYDRA_JOBS, WANDB_DIR for the code to work. In Colab, these should point to `/content/SCIGEN` or similar.
6. **Structural constraints** are a key feature — the tutorial should let users pick different lattice types (kag, hon, tri, etc.).

## Coding Guidelines

- **Do not delete or restructure existing source files.** Add new files for Colab support.
- **Prefer adding a `notebooks/` directory** for Colab notebooks.
- **Prefer adding a `colab_requirements.txt`** with minimal deps for generation.
- **Test imports and model loading** before claiming a notebook works. At minimum verify: torch, torch_geometric, pymatgen, hydra import correctly; model checkpoint loads; one small generation batch runs.
- **Keep batch sizes small for Colab** (e.g., batch_size=2, num_batches=1) to fit in free-tier GPU memory.
- **Document each notebook cell** with markdown explaining what it does — this is a tutorial.

## External Assets

- **Pretrained model weights:** https://figshare.com/ndownloader/articles/27778134/versions/1
  - Must be downloaded at runtime in Colab (not checked into git)
  - Extract to `models/mp_20/` relative to project root
- **Dataset files:** Already in `data/mp_20/` (train.csv, val.csv, test.csv) and `data/kde_bond.pkl`

## Critical Import-Time Side Effects

These are the main gotchas when importing SCIGEN modules (especially in notebooks):

1. **`scigen/common/utils.py`** — executes `load_envs()` and `os.chdir(PROJECT_ROOT)` at import time. `PROJECT_ROOT` env var MUST be set before any `import scigen.*`.
2. **`script/gen_utils.py`** — opens `./data/kde_bond.pkl` at module level. CWD must be PROJECT_ROOT before importing.
3. **`script/sc_utils.py`** — imports `MAX_ATOMIC_NUM` from `scigen.pl_modules.diffusion_w_type`, which triggers the full scigen import chain.
4. **`script/eval_utils.py`** — uses `from hydra.experimental import compose` which was removed in hydra >=1.2. The Colab notebook avoids this by using a custom model loader.

## Colab Compatibility Solutions (implemented in notebooks/tutorial_colab.ipynb)

- **pytorch_lightning version**: Notebook uses manual `state_dict` loading via `torch.load()` + `load_state_dict()` instead of PL's `load_from_checkpoint()`. Works with any PL version.
- **hydra version**: Notebook uses `from hydra import compose` (not `hydra.experimental`), compatible with hydra >=1.2.
- **torch.load weights_only**: Notebook passes `weights_only=False` explicitly for PyTorch >=2.6 compatibility.
- **Figshare download**: Uses Figshare REST API (`api.figshare.com/v2/articles/27778134`) instead of the `ndownloader` bulk URL which often fails with redirects.
- **Google Drive caching**: Optional, not mandatory. Models default to ephemeral Colab storage.

## Known Constraints

- The full import chain from `gen_utils` pulls in pyxtal, p_tqdm, pathos, sklearn, pymatgen, torch_scatter — all must be installed even for inference-only
- `hydra.experimental.compose` in `script/eval_utils.py` is incompatible with hydra >=1.2 (the notebook bypasses this entirely)
- Hydra config resolution depends on correct working directory and env vars
- `torch-scatter` must be installed from the PyG wheel index matching the exact torch+CUDA version
