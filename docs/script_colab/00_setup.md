# Colab Demo Script: Notebook 00 — Setup

**Estimated time: 5 min** | **Cells: 17** (7 markdown, 10 code) | **Long-running: 2 cells**

---

## Cell 0 — Markdown: Title & welcome

"Welcome to Notebook 0: Setup. This notebook gets your Colab environment ready for the rest of the tutorial. We have five things to do here: first, verify that you have GPU access; second, clone the SCIGEN repository from GitHub; third, install all the dependencies; fourth, download the pretrained model weights; and fifth, verify that everything works. You should run this notebook first, then move on to the other notebooks in order."

Add: "Please make sure you have Colab open with a GPU runtime. If you haven't already, go to Runtime > Change runtime type > T4 GPU."

---

## Cell 1 — Markdown [0.1]: 0.1 Check GPU

"Section 0.1: Check GPU availability. Before running this cell, go to Runtime, then Change runtime type, and make sure T4 GPU is selected."

---

## Cell 2 — Code [0.1]: GPU check

**[Run this cell.]**

"You should see something like 'GPU available: Tesla T4'. If you see the warning about no GPU, go to Runtime > Change runtime type, select T4 GPU, and re-run."

*Wait a few seconds for participants.* "Everyone good? Great."

---

## Cell 3 — Markdown [0.2]: 0.2 Clone the repository

"Section 0.2: Clone the SCIGEN repository. We're going to download the full SCIGEN codebase from GitHub. This repository contains: the diffusion model code in the scigen directory, generation scripts in the script directory, Hydra configuration files in conf, dataset files in data including the MP-20 training and test splits, and these Colab notebooks in the notebooks directory. Cloning takes about 10 seconds, and if the repo is already present it gets skipped."

---

## Cell 4 — Code [0.2]: Git clone

**[Run this cell.]**

"This takes about 10 seconds. It clones the full SCIGEN codebase from GitHub — model code, generation scripts, configuration files, dataset, and these notebooks."

*Wait for output: 'Repository ready.'*

---

## Cell 5 — Markdown [0.3]: 0.3 Install dependencies

"Section 0.3: Install dependencies. This installs the PyTorch Geometric extensions, matched to the Colab PyTorch and CUDA versions, plus all the other libraries SCIGEN needs. This step takes about 2 to 5 minutes, so you can read ahead while it runs."

---

## Cell 6 — Code [0.3]: Detect PyTorch/CUDA versions

**[Run this cell.]**

"This auto-detects your PyTorch and CUDA versions so we install the right PyTorch Geometric wheels. You should see something like 'PyTorch 2.x.x, CUDA 12.x' and a wheel URL."

---

## Cell 7 — Code [0.3]: Install all dependencies (TAKES ~2-5 min)

**[Run this cell.]**

"This is the longest step — about 2 to 5 minutes. It installs PyTorch Geometric extensions from prebuilt wheels and all the other libraries: pymatgen, hydra, CHGNet, pyxtal, smact, einops, and more."

### While waiting — fill time with overview talk:

"While this runs, let me give you the big picture of what we'll do today."

"This tutorial has 6 notebooks. In Notebook 01, we'll learn how crystal structures are represented — the (L, X, A) format that our generative model works with. In Notebook 02, we train a real diffusion model on MNIST handwritten digits to build intuition. In Notebook 03, we see how that same diffusion framework extends to crystal structures — with the key innovation of wrapped normal noise for periodic coordinates. Notebook 04 is the capstone: we use SCIGEN to generate new materials with targeted lattice geometries — kagome, honeycomb, triangular. And in Notebook 05, we evaluate those structures with CHGNet, a machine-learning interatomic potential that predicts energies and stability in seconds instead of hours."

"The running example throughout is the kagome lattice — corner-sharing triangles that create magnetic frustration and flat electronic bands. These materials are extremely rare in nature but scientifically valuable. SCIGEN lets us generate them computationally."

"The paper behind this tutorial was published in Nature Materials in 2025. The full pipeline generated 10 million candidate structures and screened them down to about 25,000 DFT-validated materials."

*Check progress.* "Still installing? Let me continue..."

"One more thing about the setup: SCIGEN has an important quirk. When you import the scigen Python package, it immediately reads an environment variable called PROJECT_ROOT and changes the working directory. So we have to set those variables before importing anything. That's what the next cell does."

*Check progress.* "Should be done soon — look for 'All dependencies installed.'"

---

## Cell 8 — Markdown [0.4]: 0.4 Configure environment variables

"Section 0.4: Configure environment variables. SCIGEN requires several environment variables to be set before you import any of its modules, because the import chain calls os.chdir of PROJECT_ROOT at load time. There are three variables: PROJECT_ROOT, which is the root directory of the codebase and where all relative paths resolve from; HYDRA_JOBS, where Hydra stores its configuration outputs, needed for model loading; and WANDB_DIR, which is the logging directory required by the training code even during inference. We also add the project and script directories to the Python path so that imports work correctly in the notebook."

Add: "The key thing is: these must be set before any `import scigen`. The import chain calls `os.chdir(PROJECT_ROOT)` at load time, so if this variable isn't set, everything breaks."

---

## Cell 9 — Code [0.4]: Set environment variables

**[Run this cell.]**

"You should see 'Working directory: /content/APS_demo_SCIGEN' and 'Environment configured.'"

---

## Cell 10 — Markdown [0.5]: 0.5 Download pretrained model

"Section 0.5: Download pretrained model weights. This downloads the SCIGEN model checkpoint, about 250 megabytes, from Figshare. We need this for Notebook 04 where we actually generate materials."

---

## Cell 11 — Code [0.5]: Download model weights (TAKES ~2-5 min)

**[Run this cell.]**

"Now we download the pretrained SCIGEN model — about 250 MB from Figshare. This uses the Figshare REST API, which is more reliable than the bulk download URL."

### While waiting — fill time with model details:

"These are the same weights used in the Nature Materials paper. The model was trained on about 45,000 crystal structures from the Materials Project — all with 20 or fewer atoms per unit cell. Training took about 1,000 epochs."

"The model architecture is a CSPNet — a graph neural network designed for crystal structure prediction. It has about 2 million parameters. During generation, it simultaneously denoises three things: atom positions, the lattice matrix, and atom types."

"The checkpoint file contains the model weights plus some scalers that normalize lattice parameters and properties. We'll see exactly how model loading works in Notebook 04."

*Check for output.* "You should see 'Model ready:' followed by the checkpoint filename."

---

## Cell 12 — Markdown [0.5]: Troubleshooting

"A quick note on troubleshooting: if the download fails, just try re-running the cell. If it fails repeatedly, you can manually download the files from Figshare and upload them to the models/mp_20 directory using the Colab file browser on the left."

"If anyone had the download fail, just re-run the cell. Figshare can be slow but usually works on the second try."

---

## Cell 13 — Markdown [0.5]: Verification code (displayed as markdown)

*[Skip — this cell shows code as markdown text for reference. Move to next cell.]*

---

## Cell 14 — Code [0.5]: Verify installation

**[Run this cell.]**

"This prints version numbers for all the key packages. You should see torch, torch_geometric, pymatgen, hydra, and CUDA available: True."

*Wait for output.* "Everyone see 'Setup complete! You are ready for the tutorial'? Great."

---

## Cell 15 — Markdown [0.5]: References

"The references section lists the papers for SCIGEN, PyTorch, PyTorch Geometric, and Hydra. Feel free to check those out later."

---

## Cell 16 — Markdown [0.5]: What's next?

"Setup is complete! Let's move on to Notebook 01: Crystal Structures, where we'll learn how to work with crystal structures in Python."

"Please open Notebook 01 now."
