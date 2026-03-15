# Colab Demo Script: Notebook 04 — SCIGEN Generation (Capstone)

**Estimated time: 20 min** | **Cells: 39** | **Long-running: 4 cells**

---

## Cell 0 — Markdown: Title and recap of previous notebooks

"We are now at Notebook 4: SCIGEN — Constrained Crystal Generation. This is the capstone notebook."

"Let me quickly recap where we've been. In Notebook 01, we learned how to represent crystals with the triplet L, X, A — lattice, fractional coordinates, atom types — and we saw that kagome structures are rare in existing databases. In Notebook 02, we trained a denoising diffusion model on MNIST to understand the core DDPM algorithm. In Notebook 03, we applied diffusion to crystal structures with DiffCSP, handling wrapped normal distributions for periodic coordinates."

> 📖 *Same text as notebook — read from screen*

"Now we bring it all together. We are going to generate brand-new crystal structures with a targeted geometric constraint using SCIGEN's inpainting approach. This is the main result of the Nature Materials 2025 paper."

---

## Cell 1 — Code: Setup (clone, install, download models) ⏳ TAKES LONG (~5-10 min first run)

**[Run this cell.]**

If NB00 was already run: "This should be fast — just verifying everything is in place. It checks for the repo, packages, and model weights before downloading anything."

If fresh session:

### 💬 While waiting

"While the environment sets up, let me explain what makes SCIGEN different from the vanilla DiffCSP we ran in Notebook 03.

DiffCSP generates any crystal from noise — unconditional generation. It's powerful, but you have no control over what comes out. You might get a cubic oxide or a hexagonal nitride — whatever the learned distribution favors.

SCIGEN adds one key innovation: inpainting. You specify a geometric pattern — say, the kagome lattice with three Mn atoms at specific fractional positions — and the model generates the rest of the crystal around it. The constrained atom positions are revealed at the final denoising step, replacing whatever the model predicted at those sites. During training, the model learned to anticipate these constraints, so the complementary atoms it generates are chemically reasonable.

The published pipeline started with 13 different lattice types — from simple triangular with 1 known atom to the great rhombotrihexagonal with 12 known atoms. For each type, they generated millions of candidates, screened them with bond length and composition filters, then evaluated with CHGNet and finally DFT. The result: 24,743 DFT-validated new materials, many of which had never been proposed before.

Today we'll run a miniature version of this pipeline in real time."

---

## Cell 2 — Markdown [4.1]: 4.1 What is SCIGEN?

"Section 4.1: What is SCIGEN? SCIGEN is a diffusion-based generative model for crystal structures. The key innovation is that you can specify a geometric constraint — kagome, honeycomb, triangular, and so on — and the model generates a complete crystal around it. It uses an inpainting approach with four steps. Step one: generate a full structure from noise. Step two: the constrained positions are inpainted at the end. Step three: the model has learned to anticipate these constraints during training. Step four: the lattice geometry itself is also constrained to match the target symmetry."

"The published pipeline screened 10 million candidates down to 24,743 DFT-validated new materials."

> 📖 *Same text as notebook — read from screen*

"The four-step inpainting process is worth repeating. Step one: define the constraint — which atoms sit where. Step two: start from pure noise for everything else. Step three: denoise for 1000 steps. Step four: at the very end, replace the model's prediction at the constrained sites with the true positions. The model has already learned to expect this — it generates atoms that make chemical sense around the kagome or honeycomb sublattice."

---

## Cell 3 — Markdown [4.1.1]: Pipeline figure and lattice types image

"Here we have two figures. The top figure shows the full SCIGEN pipeline from the paper — from constraint definition through generation, screening, and DFT validation. The bottom image shows all 13 available lattice types. Each one defines a different 2D tiling pattern with a specific number of known atoms. The simplest is triangular with 1 known atom. The most complex is great rhombotrihexagonal with 12."

> 📖 *Same text as notebook — read from screen*

---

## Cell 4 — Markdown [4.2]: 4.2 Environment setup

"Section 4.2 is environment setup. If you already ran Notebook 00, this is already done. Otherwise the next cell handles it."

> 📖 *Same text as notebook — read from screen*

---

## Cell 5 — Code [4.2]: Set environment variables and sys.path

**[Run this cell.]**

"Sets PROJECT_ROOT, HYDRA_JOBS, WANDB_DIR, and adds the project directories to the Python path. You should see 'Working directory:' followed by the project path."

---

## Cell 6 — Markdown [4.3]: 4.3 Load the pretrained model

"Section 4.3: Load the pretrained model. We use Hydra config plus manual state-dict loading for compatibility across different versions of PyTorch Lightning."

> 📖 *Same text as notebook — read from screen*

---

## Cell 7 — Code [4.3]: Define load_model_for_inference() helper

**[Run this cell.]**

"This defines a helper function that handles all the complexity of loading the model: clearing Hydra state, composing the config from hparams.yaml, instantiating the architecture, loading the checkpoint state_dict, and attaching the data scalers. No output yet — we call it in the next cell."

---

## Cell 8 — Code [4.3]: Load model and attach sample_scigen method

**[Run this cell.]**

"This calls the helper function and attaches the generation sampling method. You should see: 'Loading checkpoint: ...' followed by 'Model loaded: ~2,000,000 parameters'. The device should show cuda."

---

## Cell 9 — Markdown [4.4]: 4.4 Choose your structural constraint (THE KEY CELL)

"Section 4.4: Choose your structural constraint. This is the interactive heart of the notebook. Here is a table with all 13 available lattice types."

"Let me read through them. 'kag' is kagome — 3 known atoms. 'hon' is honeycomb — 2 known atoms. 'tri' is triangular — 1 known atom. 'sqr' is square — 1 known atom. 'lieb' is the Lieb lattice — 3 known atoms. 'elt' is elongated triangular — 2. 'sns' is snub square — 4. 'tsq' is truncated square — 4. 'srt' is small rhombotrihexagonal — 6. 'snh' is snub hexagonal — 6. 'trh' is trihexagonal — 6. 'grt' is great rhombotrihexagonal — 12 known atoms. And finally 'van' is vanilla, meaning no constraint at all — just 1 atom."

> 📖 *Same text as notebook — read from screen*

"In the next code cell, you will set SC_TYPE — the lattice geometry — and ELEMENT — which element sits at the constrained sites. We'll start with kagome Mn, but I encourage you to come back and try honeycomb Fe, triangular Co, or Lieb Ni after we finish the first pass."

---

## Cell 10 — Code [4.4]: Set generation parameters (user-editable)

**[Run this cell.]**

"The defaults are SC_TYPE equals 'kag' for kagome, ELEMENT equals 'Mn', BATCH_SIZE equals 4. You can change these and re-run. The atom count range is automatically set based on the constraint type — kagome allows 1 to 12 atoms per cell.

You should see a summary: lattice type kagome, element Mn, 4 structures, 1 to 12 atoms per cell."

---

## Cell 11 — Code [4.4]: Build dataset and run diffusion sampling ⏳ TAKES LONG (~20-30 sec)

**[Run this cell.]**

"This builds the SampleDataset with the structural constraints and runs the diffusion sampling. Four structures, 1000 denoising steps each."

### 💬 While waiting (~20-30 sec)

"What's happening inside: the SampleDataset creates the initial noise and the constraint template. For kagome, that means three Mn atoms at fractional positions (0.5, 0, z), (0, 0.5, z), and (0.5, 0.5, z) in a hexagonal cell with gamma equals 120 degrees. The remaining atom slots start as random noise.

The model runs the same predictor-corrector reverse diffusion we saw in Notebook 03 — predicting noise with the CSPNet, updating the lattice, running Langevin dynamics on the coordinates with periodic wrapping. The only difference is that at each step, the constrained positions are revealed and override the model's predictions.

After 1000 steps, the model outputs the final structure: fractional coordinates, lattice matrix, and atom types for all atoms — both the constrained kagome sites and the model-generated complementary atoms."

*Check for output.* "You should see 'Generation complete in about 20-30 seconds, roughly 5-8 seconds per structure.'"

---

## Cell 12 — Markdown [4.5]: 4.5 Visualizing the diffusion trajectory

"Section 4.5: Visualizing the diffusion trajectory. The trajectory captures every denoising step. We will render key frames so you can see the crystal forming, with constrained atoms highlighted."

> 📖 *Same text as notebook — read from screen*

---

## Cell 13 — Code [4.5]: Trajectory filmstrip (8 key frames)

**[Run this cell.]** Takes a few seconds.

"Eight frames from the trajectory of the first structure. The highlighted atoms — shown in red if PyVista is available — are the constrained kagome sites. Watch how the rest of the crystal forms around them: the complementary atoms gradually find their positions while the kagome sublattice remains fixed.

If PyVista isn't available, you'll see matplotlib 3D scatter plots instead, with red circles marking the constrained sites."

---

## Cell 14 — Code [4.5]: Animated GIF of the trajectory

**[Run this cell.]** Takes 10-20 seconds.

"A twenty-frame animation of the full generation process. The red-highlighted atoms are the kagome constraint. Everything else was generated by the model. Watch how the non-kagome atoms crystallize around the fixed kagome sites — first roughly clustering, then settling into precise positions."

---

## Cell 15 — Markdown [4.6]: 4.6 Inspect generated structures

"Section 4.6: Inspect generated structures. We convert the raw model output to pymatgen Structure objects and look at composition, lattice parameters, and how many atoms are known versus total."

> 📖 *Same text as notebook — read from screen*

---

## Cell 16 — Code [4.6]: Print structure table (compositions, lattice params)

**[Run this cell.]**

"A table with one row per generated structure. The columns show: structure index, total atoms, known atoms (the kagome constraint sites), composition, and lattice parameters a, b, c, alpha, beta, gamma.

Look at the compositions — the model chose real elements that commonly form compounds with Mn. Look at the lattice parameters — for kagome, you should see a approximately equal to b and gamma near 120 degrees, reflecting the hexagonal symmetry of the kagome constraint."

---

## Cell 17 — Code [4.6]: Convert to pymatgen Structure objects

**[Run this cell.]**

"This converts the raw tensors to pymatgen Structure objects. For each structure, you see the reduced formula and unit cell volume. These Structure objects are what we use for all downstream analysis: symmetry, XRD, bond distances, and eventually CHGNet evaluation in Notebook 05."

---

## Cell 18 — Markdown [4.6.1]: Lattice parameter distributions

"Now we look at lattice parameter distributions. For kagome, the hexagonal constraint means a should approximately equal b and gamma should be near 120 degrees. The physical range for lattice parameters is typically 2 to 15 angstroms. If we had more structures, we'd see tight peaks at those values."

> 📖 *Same text as notebook — read from screen*

---

## Cell 19 — Code [4.6.1]: Lattice parameter plots

**[Run this cell.]**

"With only 4 structures, we see strip plots rather than histograms — each point is one structure. Check that a and b are similar and gamma is near 120. The c parameter is unconstrained and varies more freely."

---

## Cell 20 — Code [4.6.1]: PyVista interactive viewer and grid view

**[Run this cell.]**

"If PyVista is available, you get an interactive 3D viewer for the first structure — you can rotate, zoom, and adjust atom sizes. Below that, a grid showing all generated structures side by side. If PyVista isn't available, a fallback message appears."

---

## Cell 21 — Markdown [4.7]: 4.7 Space group analysis

"Section 4.7: Space group analysis. We determine the symmetry of each generated structure and compare against known materials. For kagome compounds, we typically see hexagonal or trigonal space groups — groups 143 to 194 — reflecting the six-fold symmetry of the kagome lattice."

> 📖 *Same text as notebook — read from screen*

---

## Cell 22 — Code [4.7]: SpacegroupAnalyzer on all structures

**[Run this cell.]**

"A table showing each structure's space group symbol, number, crystal system, and point group. For kagome Mn structures, you'll often see P-3, P6mm, or other hexagonal/trigonal groups. Some structures may have lower symmetry — the model isn't forced to produce the highest possible symmetry; it generates whatever is most probable given the training data."

---

## Cell 23 — Code [4.7]: Space group distribution plots

**[Run this cell.]**

"A histogram of space group numbers and a pie chart of crystal systems. With only 4 structures, the distributions aren't very informative — but if you increase BATCH_SIZE to 20 or more, you'll see meaningful patterns emerge."

---

## Cell 24 — Markdown [4.8]: 4.8 Simulated X-ray diffraction

"Section 4.8: Simulated X-ray diffraction. We compute a powder XRD fingerprint using Cu K-alpha radiation. This is a key validation step — if you were to actually synthesize one of these materials, you would compare your measured diffraction pattern against this simulation to confirm you made the right crystal phase."

> 📖 *Same text as notebook — read from screen*

---

## Cell 25 — Code [4.8]: XRD patterns for up to 4 structures

**[Run this cell.]**

"Simulated powder XRD patterns with Cu K-alpha radiation. Each panel shows intensity versus 2-theta angle, with the strongest peaks labeled by their Miller indices. If you were to synthesize one of these materials, you'd compare your measured pattern against this simulation to confirm the crystal phase. The peak positions are determined by the lattice spacing via Bragg's law; the intensities depend on the atom types and positions."

---

## Cell 26 — Markdown [4.9]: 4.9 Comparing vs published SCIGEN structures

"Section 4.9: Comparing against published SCIGEN structures. This is a three-way comparison: the MP-20 training data that the model learned from, the 24,743 DFT-validated structures from the SCIGEN paper, and our freshly generated structures. If all three overlap in property space, it means the model is generating realistic materials."

> 📖 *Same text as notebook — read from screen*

---

## Cell 27 — Code [4.9]: Download SCIGEN DFT-relaxed structures ⏳ TAKES LONG (~1-3 min)

**[Run this cell.]**

"Downloading 28 megabytes of DFT-relaxed structures from Figshare — about 24,700 materials across all constraint types."

### 💬 While waiting (~1-3 min)

"These are the crown jewels of the SCIGEN paper. Starting from 10 million generated candidates, they applied a four-stage screening pipeline: first, composition and bond-distance filters that are essentially free computationally. Second, GNN-based stability predictions. Third, CHGNet energy and force evaluations — the same tool we'll use in Notebook 05. Fourth and finally, full DFT calculations with VASP to confirm stability.

The result: 24,743 structures that survived all four stages. That's about 0.25 percent of the initial 10 million — the vast majority of generated candidates are not stable or physically reasonable. But the ones that survive are genuine new materials candidates, many of which have never been proposed before.

The breakdown by constraint type is interesting. Kagome structures are one of the largest categories, followed by honeycomb and triangular. Some of the complex tilings — like snub hexagonal or great rhombotrihexagonal — have fewer surviving candidates, because those constraints are harder to satisfy with real chemistry."

*Check output.* "You should see the total count and a breakdown by constraint type."

---

## Cell 28 — Code [4.9]: Parse sampled SCIGEN structures + training data ⏳ TAKES LONG (~5-10 min)

**[Run this cell.]**

"This parses 500 random SCIGEN DFT structures — reading VASP CONTCAR files — plus 500 training structures from the CSV. For each, it computes nearest-neighbor distances and element frequencies."

### 💬 While waiting (~5-10 min)

"Let me talk about what makes a generated structure 'good' versus 'bad.'

A physically reasonable structure has nearest-neighbor distances between about 1.5 and 3.5 angstroms. Below 1 angstrom, atoms overlap — the Pauli exclusion principle says that can't happen. Above 4 angstroms, atoms aren't really bonded. So the NN distance distribution is our first sanity check.

Lattice parameters should match the range seen in training — typically 3 to 15 angstroms for a, b, c. Very small cells suggest the model collapsed; very large cells suggest it didn't converge.

Element choices should reflect reasonable chemistry. Mn kagome compounds often contain O, S, or halides as the complementary atoms, because these form stable ternary and quaternary compounds with manganese. If the model generates kagome Mn with noble gases or very heavy actinides, something went wrong.

The three-way comparison we're about to see is the gold standard validation. The freshly generated structures — our 4 kagome Mn crystals — sit alongside 500 published SCIGEN structures and 500 training structures. If they all overlap in property space, our generation run produced realistic materials.

One thing to watch for in the constraint type breakdown: the relative proportions of different lattice types in the published dataset reflect both the generation success rate and the screening survival rate. Kagome and honeycomb tend to have high survival rates because their hexagonal symmetry is common in real crystals. The complex tilings with 6 or 12 known atoms have lower survival because it's harder to find compatible chemistry for so many constrained sites.

The t-SNE plot we'll see after this groups structures by composition similarity. Clusters in t-SNE space correspond to groups of materials with similar element ratios — for example, all the Mn-O binary compounds might cluster together, while Mn-S ternary compounds form a different cluster."

---

## Cell 29 — Code [4.9]: 6-subplot distribution comparison (3-way)

**[Run this cell.]** Runs quickly.

"Six panels. Blue is MP-20 training data. Green is SCIGEN DFT-validated. Coral is our freshly generated structures — though with only 4 points, they won't dominate the plot.

Top row: atoms per cell, lattice parameter a, lattice parameter c. Bottom row: NN bond distances, element frequency bar chart, and constraint type breakdown for the SCIGEN dataset.

The key observation: the green SCIGEN distribution overlaps well with the blue training distribution for most properties, confirming that the screened materials are physically realistic. Look at the bond distance panel — the green and blue histograms should largely overlap, with most distances in the 1.5 to 3.5 angstrom range."

---

## Cell 30 — Code [4.9]: t-SNE of composition space (3-way)

**[Run this cell.]** Takes a few seconds.

"Three datasets projected into 2D composition space. Blue dots: training. Green dots: SCIGEN DFT-validated. Coral stars: our freshly generated structures. The SCIGEN points should overlap with training while also filling some gaps — exploring new compositions not present in the training data. Our 4 coral stars should sit somewhere within or near the green and blue clouds."

---

## Cell 31 — Markdown [4.10]: 4.10 Export as CIF files

"Section 4.10: Export as CIF. We save the generated structures as CIF files so they can be loaded into visualization tools like VESTA or Materials Studio, or used for further computational analysis."

> 📖 *Same text as notebook — read from screen*

---

## Cell 32 — Code [4.10]: Save CIF files

**[Run this cell.]**

"Each structure is saved as a CIF file named with the constraint type and formula. You should see 4 saved files. These can be loaded into VESTA, Materials Studio, or any crystallography software for further analysis."

---

## Cell 33 — Code [4.10]: Pipeline flowchart

**[Run this cell.]**

"A visual summary of the full SCIGEN discovery pipeline: choose constraint, generate with SCIGEN, pre-screen by composition and bonds, MLIP screen with CHGNet, DFT validation, and experimental synthesis. We've done the first three steps in this notebook. Notebook 05 covers the MLIP screening."

---

## Cell 34 — Code [4.10]: Download CIF zip (Colab only)

**[Run this cell.]**

"This packages the CIF files into a zip and triggers a download in Colab. If you're not in Colab, it just prints the file path. You can also find the files in the Colab file browser under generated_cifs/."

---

## Cell 35 — Markdown [4.11]: 4.11 Optional CHGNet evaluation

"Section 4.11: Optional CHGNet evaluation. You could run a quick energy check with CHGNet right here, but the full evaluation — relaxation, phonon calculations, convex hull analysis — is all in Notebook 05. Let's skip this for now and move on."

> 📖 *Same text as notebook — read from screen*

---

## Cell 36 — Markdown [4.12]: 4.12 Try it yourself — exercises

"Section 4.12: Try it yourself. There are three exercises here. First: change SC_TYPE to 'hon' for honeycomb and ELEMENT to 'Fe', then compare the compositions and lattice parameters you get. Second: try a different value for step_lr — the Langevin dynamics step size — and see how it affects the output. Third: check for unphysical bond distances in your generated structures."

"If we have time at the end, come back and try these. For now, let's move to evaluation."

> 📖 *Same text as notebook — read from screen*

---

## Cell 37 — Markdown [4.12]: References

"References for SCIGEN, CHGNet, SMACT, pymatgen, and PyVista are in the notebook."

---

## Cell 38 — Markdown [4.12]: Summary and what's next

"Let me wrap up with the summary. Over these four notebooks, we followed four steps. Step one in Notebook 01: represent crystal structures with the L, X, A triplet. Step two in Notebook 02: learn the denoising diffusion framework on MNIST. Step three in Notebook 03: understand how diffusion applies to periodic crystal structures. Step four here in Notebook 04: generate new materials with targeted geometric constraints using SCIGEN."

"You can see the full pipeline diagram in the notebook."

> 📖 *Same text as notebook — read from screen*

"We've now completed the generation side. We have 4 kagome Mn crystal structures that the model created from scratch, guided only by the constraint that 3 atoms form a kagome pattern. In the final notebook, we'll ask: are these structures actually stable? Do they have interesting magnetic properties? That's where CHGNet comes in. Let's open Notebook 05."
