# Colab Migration / Tutorialization

Use this skill when converting a research codebase into a Google Colab tutorial, or when creating/updating Colab notebooks for this project.

## Workflow

### Phase 1: Understand the Codebase
1. Read `README.md` thoroughly — it defines project intent, dependencies, and workflows
2. Read `CLAUDE.md` for project-specific guidance and priorities
3. Inspect repository structure: `ls -la`, then explore `scigen/`, `script/`, `conf/`, `data/`
4. Identify the **minimum inference/generation path** — trace from `gen_mul.py` → `script/generation.py` → model loading → output

### Phase 2: Identify Requirements
1. Map the dependency chain for generation only (not training)
2. Check which dependencies are Colab-compatible (Colab provides PyTorch, CUDA by default)
3. Identify external assets that must be downloaded (pretrained models from Figshare)
4. Note any `.env` or config files that need Colab-specific values

### Phase 3: Create Colab Artifacts
1. Create `notebooks/` directory if it doesn't exist
2. Write notebook(s) with clear markdown documentation in each cell
3. Create `colab_requirements.txt` with minimal dependencies for generation
4. Include cells for:
   - Environment setup and dependency installation
   - Pretrained model download and extraction
   - `.env` / config file setup pointing to Colab paths
   - A minimal generation run (small batch: batch_size=2, num_batches=1)
   - Output inspection and optional CIF conversion
   - Visualization if feasible

### Phase 4: Verify
1. Check that all imports resolve (at minimum: torch, torch_geometric, pymatgen, hydra)
2. Verify model checkpoint loading code path works
3. Confirm generation produces output with a minimal batch
4. Ensure the notebook is self-contained — a user clicking "Run All" should get results

## Key Decisions

- **Add, don't rewrite.** Put Colab files alongside the existing codebase.
- **Pretrained only.** Do not include training in the tutorial unless explicitly requested.
- **Small defaults.** Use small batch sizes and single runs for Colab free-tier compatibility.
- **Structural constraints are the showcase.** Let users pick lattice types (kag, hon, tri, etc.).
- **Document everything.** This is a tutorial — every cell needs a markdown explanation.

## Pretrained Model Download

```python
import os, subprocess
if not os.path.exists('models/mp_20'):
    subprocess.run(['wget', '-q', 'https://figshare.com/ndownloader/articles/27778134/versions/1', '-O', 'models.zip'])
    subprocess.run(['unzip', '-q', 'models.zip', '-d', '.'])
    os.rename('mp_20', 'models/mp_20')  # adjust path as needed after inspecting zip contents
```

## Colab Environment Setup Pattern

```python
import os
os.environ['PROJECT_ROOT'] = '/content/SCIGEN'
os.environ['HYDRA_JOBS'] = '/content/SCIGEN/hydra'
os.environ['WANDB_DIR'] = '/content/SCIGEN/wandb'
```

## Common Issues to Watch For
- `pytorch_lightning==1.3.8` is very old — may conflict with Colab's default PyTorch
- `torch-geometric` requires matching torch+CUDA versions for installation
- Hydra working directory assumptions may break in notebook context
- `python-dotenv` must be installed and `.env` must exist for config loading
