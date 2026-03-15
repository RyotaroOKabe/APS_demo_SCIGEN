# Colab Demo Script: Notebook 03 — Diffusion for Materials

**Estimated time: 15 min** | **Cells: 30** | **Long-running: 3 cells**

---

## Cell 0 — Markdown: Title, learning objectives, prerequisites

"Welcome to Notebook 3: DiffCSP — Crystal Structure Prediction via Diffusion. In this notebook we're going to learn how DiffCSP generates crystals from scratch, how noise is applied to fractional coordinates using periodic wrapping and to lattice matrices using standard Gaussian noise, how the reverse denoising process reconstructs a crystal from that noise, and we'll visualize the full diffusion trajectory in both directions. DiffCSP is the foundation that SCIGEN builds upon — it was published by Jiao and colleagues at NeurIPS 2023."

> 📖 *Same text as notebook — read from screen*

Add: "This is really the heart of the tutorial. DiffCSP is the engine that powers SCIGEN. Once you understand how diffusion works on crystals — the periodic wrapping, the lattice noise, the reverse denoising — everything in Notebook 04 about constrained generation will click into place. Let's dive in."

---

## Cell 1 — Code: Setup (clone, install, download models)

**[Run this cell.]** ⏳ TAKES LONG (~5-10 min first run)

If Notebook 00 was already run in this session, say: "This should be fast — just verifying that the repo, packages, and model weights are all in place. It checks whether each component exists before downloading, so it skips anything already done."

If this is a fresh session, say: "This cell handles the full environment setup — cloning the repository, installing PyTorch Geometric extensions matched to our CUDA version, installing all Python dependencies, and downloading pretrained model weights from Figshare. It takes about five to ten minutes the first time."

### While waiting (fresh session only)

"While that runs, let me set the stage for what we're about to do. DiffCSP — Crystal Structure Prediction by Joint Equivariant Diffusion — was published at NeurIPS 2023 by Jiao and colleagues. It was one of the first models to successfully apply denoising diffusion to the full crystal structure: not just atom positions, but also the lattice matrix and atom types, all generated simultaneously.

The fundamental challenge with crystals is periodicity. In an image, pixel values are unbounded reals — standard Gaussian noise works fine. But fractional coordinates live on a torus: x=0 and x=1 are the same point. If you naively add Gaussian noise, you break the periodicity and generate atoms outside the unit cell. DiffCSP solved this with the wrapped normal distribution, which we'll visualize in a few cells.

SCIGEN took DiffCSP's architecture and added an inpainting mechanism for structural constraints — kagome lattices, honeycomb nets, and so on. But the underlying diffusion machinery is identical, which is why we study it here first."

---

## Cell 2 — Markdown [3.1]: 3.1 DiffCSP foundation

"Section 3.1: DiffCSP foundation. This table shows the three components of a crystal that the model must learn to generate: atom positions, lattice, and atom types. For atom positions, the representation is fractional coordinates, and the noise model is a wrapped normal distribution — that's the key innovation. For the lattice, it's a 3-by-3 matrix with standard Gaussian noise. And for atom types, it's a one-hot encoding, again with Gaussian noise. The critical point is that fractional coordinates are periodic, so noise must wrap around the unit cell boundaries. SCIGEN extends DiffCSP with structural constraints via an inpainting mechanism."

> 📖 *Same text as notebook — read from screen*

Point at the table and add: "Take a moment to look at this table. Three components, three different noise models. The atom positions use wrapped normal noise — that's the key innovation. The lattice matrix and atom types use standard Gaussian noise, which is simpler. SCIGEN inherits all of this and adds structural constraints on top via inpainting."

---

## Cell 3 — Markdown [3.2]: 3.2 Load pretrained model

"Section 3.2: Load pretrained model. We're going to load the same SCIGEN model checkpoint and use it here. The important thing to know is that this model can run both with and without structural constraints — it's the same architecture either way."

> 📖 *Same text as notebook — read from screen*

Add: "We're using the same SCIGEN checkpoint we downloaded in setup. It shares DiffCSP's architecture — a CSPNet graph neural network — and can operate in both unconstrained and constrained modes. For this notebook, we run it in vanilla mode to understand the base diffusion process."

---

## Cell 4 — Code [3.2]: Load model via Hydra and state_dict

**[Run this cell.]** Should complete in a few seconds.

"This cell sets environment variables, initializes Hydra with the model's saved hyperparameters, instantiates the model architecture, then loads the pretrained weights from the checkpoint file. Notice we use torch.load with weights_only=False — that's needed for PyTorch 2.6 compatibility. We also load the lattice and property scalers, which normalize inputs during generation.

You should see three lines of output: the parameter count — around two million — the device, which should be cuda, and the number of diffusion timesteps, which is one thousand."

---

## Cell 5 — Markdown [3.3]: 3.3 Forward process math

"Section 3.3: the forward process. There are two types of noise happening simultaneously. For coordinate noise, we use a wrapped normal distribution. That means we sample Gaussian noise and then take mod one. Here's the concrete example: an atom sitting at position x equals 0.95 gets noise of 0.1. Instead of ending up at 1.05, which would be outside the unit cell, it wraps around to 0.05. For lattice noise, it's simpler — standard Gaussian noise scaled by alpha-bar from the noise schedule."

> 📖 *Same text as notebook — read from screen*

Add: "The wrapped normal equation is the crucial one. That WN means we sample Gaussian noise and then take mod one. The concrete example here is very intuitive: an atom at x=0.95 gets noise of 0.1, and instead of flying to 1.05 — which is outside the unit cell — it wraps to 0.05. It just reappears on the other side, exactly like periodic boundary conditions in a simulation.

For the lattice, it's simpler: standard Gaussian noise scaled by the cumulative product of alphas. As t increases, alpha-bar shrinks toward zero and the signal vanishes."

---

## Cell 6 — Code [3.3]: Load test structure, extract noise schedules, plot

**[Run this cell.]** Runs quickly (< 5 seconds).

"We load a real crystal from the MP-20 test set and extract the noise schedules from the model. Two plots appear side by side.

On the left: the lattice noise schedule — alpha-bar versus timestep. It starts near one, meaning almost no noise, and drops toward zero. The dashed line at 0.5 marks the crossover point where noise and signal are equal.

On the right: the coordinate noise schedule — sigma versus timestep. This controls how much the fractional coordinates spread before wrapping. By t=999, sigma is large enough that coordinates are essentially uniform on zero-to-one — pure noise."

---

## Cell 7 — Markdown [3.3.3]: Visualizing noisy crystal structures

"Now let's visualize what the forward process actually looks like. We'll take the crystal we just loaded and add noise at several different timesteps, then render each one so you can watch the structure dissolve."

> 📖 *Same text as notebook — read from screen*

Add: "This next cell is where the forward process becomes visual. We'll take the crystal we just loaded and corrupt it at six different timesteps."

---

## Cell 8 — Code [3.3.3]: Forward process visualization (6 timesteps, 3D scatter)

**[Run this cell.]** Runs in a few seconds.

"Six panels, one for each timestep. At t=0, you see the clean crystal — atoms neatly arranged, lattice intact. At t=100, slight jitter. By t=300, things are noticeably blurred. At t=500, halfway through the schedule, the structure is barely recognizable. By t=800 and t=999, it's a random cloud of points in a distorted box.

Notice how the unit cell itself distorts — that's the lattice noise at work. And the atoms don't fly off to infinity; they wrap around, so they always stay inside the cell boundaries. That's the wrapped normal in action.

The text at the bottom reminds us: coordinates wrap around zero-to-one. Atoms that drift past a cell face reappear on the other side."

---

## Cell 9 — Markdown [3.4]: 3.4 PyVista rendering header

"Section 3.4: PyVista rendering. We can get publication-quality 3D renderings of our noisy structures using PyVista, with proper unit cell edges and element-colored spheres."

> 📖 *Same text as notebook — read from screen*

Add: "PyVista gives us publication-quality 3D renders with proper unit cell edges and element-colored spheres. If PyVista isn't available in your environment, the notebook gracefully falls back to the matplotlib plots you already saw."

---

## Cell 10 — Code [3.4]: PyVista filmstrip of 5 forward frames

**[Run this cell.]** Takes a few seconds for rendering.

"This produces a horizontal filmstrip — five frames of the forward process rendered with PyVista. You can see the unit cell box, color-coded atoms, and the progressive corruption. If PyVista isn't installed, you'll see a message saying to refer to the matplotlib plots above — that's fine, the physics is the same."

---

## Cell 11 — Markdown [3.4.1]: Animated forward process

"Next we'll create an animated version of the forward process — a GIF that shows the crystal dissolving into noise frame by frame, rather than just a few snapshots."

> 📖 *Same text as notebook — read from screen*

Add: "The next cell creates a GIF animation so you can see the dissolution happen frame by frame, rather than jumping between a few snapshots."

---

## Cell 12 — Code [3.4.1]: 20-frame GIF of forward diffusion

**[Run this cell.]** Takes 10-20 seconds to render all frames.

"This renders twenty frames and stitches them into an animated GIF. Watch the crystal melt: first the atoms jitter, then they spread apart, the unit cell warps, and finally you're left with what looks like random scatter.

This is the forward process that the model must learn to reverse. Given pure noise like the last frame, the trained network has to figure out where all the atoms should go and what shape the lattice should be. That's a remarkable feat of learning."

---

## Cell 13 — Markdown [3.5]: 3.5 Reverse process (predictor-corrector)

"Section 3.5: the reverse process. This is where the model generates crystals. It uses a predictor-corrector scheme. The predictor applies a reverse SDE step to the lattice — scaling by one over root alpha, subtracting predicted noise, and adding fresh stochasticity. The corrector applies a Langevin dynamics step specifically for the fractional coordinates, guided by the score function, with a mod-one wrapping at the end to keep everything periodic. We need both steps because the predictor alone accumulates errors in periodic space — the corrector refines positions and dramatically improves crystal quality."

> 📖 *Same text as notebook — read from screen*

Add: "Two equations here — the predictor and the corrector. The predictor is a standard DDPM reverse step applied to the lattice: scale by one over root alpha, subtract the predicted noise, add fresh stochasticity. The corrector is a Langevin dynamics step on the fractional coordinates, guided by the score function — the gradient of the log-probability. That mod-one at the end is critical: every update wraps coordinates back into the unit cell.

Why two steps? The predictor alone accumulates errors in periodic space. The corrector refines positions using the score, which dramatically improves the quality of generated crystals. It's the same predictor-corrector idea used in score-based generative models for images, adapted for periodic coordinates."

---

## Cell 14 — Code [3.5]: Generate 4 structures (vanilla, no constraints)

**[Run this cell.]** ⏳ TAKES LONG (~20-30 sec)

"This cell generates four crystal structures from scratch — starting from pure noise and running the full one-thousand-step reverse diffusion. We're in vanilla mode: no structural constraints, just free generation."

### While waiting

"What's happening right now inside the GPU: the model starts with random atom positions wrapped uniformly on zero-to-one, a random 3x3 lattice matrix, and random atom type vectors. Then for each of the one thousand timesteps, it runs two steps.

First, the predictor: the CSPNet graph neural network takes the current noisy structure, processes it through message-passing layers that respect the periodic symmetry, and predicts the noise that was added. We subtract that predicted noise and add a smaller amount of fresh noise — stepping backward along the diffusion schedule.

Second, the corrector: a Langevin dynamics update on the fractional coordinates using the predicted score. This nudges atoms toward more probable positions, respecting the wrapped geometry.

After all one thousand steps, the model outputs fractional coordinates, a lattice matrix, and atom type logits. The atom types are converted to elements by taking the argmax.

You should see output telling you it completed in about twenty to thirty seconds, with a per-structure time of five to eight seconds. That's the power of diffusion — it's slower than a single neural network forward pass, but it produces much higher quality samples."

---

## Cell 15 — Markdown [3.6]: 3.6 Visualizing reverse trajectory

"Section 3.6: Visualizing the reverse trajectory. The model saved the full trajectory — positions, lattice, and atom types at every timestep. Let's render eight key frames to see the noise-to-crystal transition."

> 📖 *Same text as notebook — read from screen*

Add: "The model saved the full trajectory — positions, lattice, and types at every step. Let's render eight key frames to see the noise-to-crystal transition."

---

## Cell 16 — Code [3.6]: 8-frame reverse trajectory rendering

**[Run this cell.]** Takes a few seconds.

"Eight frames arranged in a grid. Reading left to right, top to bottom: the first frame is pure noise — random positions, distorted lattice. As we step through the reverse process, you see structure emerge. Atoms start clustering. The lattice regularizes into something with proper angles. By the last frame at t=0.00, you have a crystal.

If PyVista is available, you'll see proper 3D renderings with unit cell edges. Otherwise, matplotlib scatter plots — either way, the transformation from noise to crystal is clearly visible."

---

## Cell 17 — Markdown [3.6.1]: Animated reverse process

"Now let's see the reverse process as an animation — noise transforming into a crystal, frame by frame."

> 📖 *Same text as notebook — read from screen*

Add: "This is the reverse of what we saw earlier — now the animation goes from noise to crystal."

---

## Cell 18 — Code [3.6.1]: 20-frame reverse GIF and before/after comparison

**[Run this cell.]** Takes 10-20 seconds.

"The animation plays the full reverse process: starting from the random cloud at t=1.0, atoms gradually find their positions, the lattice locks into shape, and by t=0.0 you have a generated crystal. Below the GIF, there's a side-by-side comparison: the starting noise on the left, the final crystal on the right.

The cell also prints the formula of the generated structure — how many atoms and which elements the model chose. Remember, nothing was specified in advance: the model decided the composition, the lattice, and all atom positions entirely from the learned distribution."

---

## Cell 19 — Markdown [3.7]: 3.7 Comparing generated vs training distributions

"Section 3.7: Comparing generated versus training distributions. Pretty visualizations are great, but now we need to ask the quantitative question — do generated structures actually look like real materials? We'll compare distributions of lattice parameters, bond distances, atom counts, and element frequencies between our generated structures and the training data."

> 📖 *Same text as notebook — read from screen*

Add: "This is the quantitative check. Pretty pictures are nice, but what we really want to know is: does the model produce crystals that are statistically indistinguishable from real materials? The next two cells generate a larger batch and compare six different property distributions."

---

## Cell 20 — Code [3.7]: Generate 50 structures and parse 300 training structures

**[Run this cell.]** ⏳ TAKES LONG (~5-10 min)

"This is the longest cell in the notebook. It generates fifty structures from scratch — the same vanilla generation we just did, but fifty at a time instead of four. Then it parses three hundred structures from the MP-20 training set to build the reference distributions. For each structure, it computes lattice parameters, nearest-neighbor distances, and element frequencies."

### While waiting

"Let me talk about why this distribution comparison matters and what the physical quantities mean.

First, nearest-neighbor distances. In real crystals, the shortest bond between two atoms is typically between 1.5 and 3.5 angstroms. Below about 1 angstrom, atoms overlap — that's physically impossible. Above 4 angstroms, you're looking at non-bonded distances. So a well-trained generative model should produce nearest-neighbor distances concentrated in that 1.5 to 3.5 range, matching the training data.

Second, lattice parameters. The MP-20 dataset contains structures with up to 20 atoms per unit cell, so lattice parameters a, b, c typically range from about 3 to 15 angstroms. The angles alpha, beta, gamma cluster around 60, 90, and 120 degrees, corresponding to common crystal systems — cubic, hexagonal, monoclinic, and so on. If the generated distribution matches, the model has learned realistic cell geometries.

Third, element frequencies. The Materials Project is dominated by certain elements — oxygen, transition metals like iron and manganese, alkali metals. A good generative model should reproduce these frequencies approximately, not hallucinate exotic elements.

Fourth, the number of atoms per unit cell. The training data has a specific distribution — most structures have between 4 and 16 atoms. If the model consistently generates structures with 1 or 2 atoms, or always maxes out at 20, something is wrong.

Now, about SMACT screening — which we'll see in a few cells. SMACT stands for Semiconducting Materials by Analogy and Chemical Theory. It's a fast, rule-based filter that checks two things: does the composition have both electropositive and electronegative elements — you need that contrast to form a compound — and can the elements be assigned integer oxidation states that sum to zero? If a generated composition fails SMACT, it's almost certainly not a real material.

The pass rate on SMACT gives us a quick measure of chemical validity. For a well-trained model, we'd expect 60 to 80 percent of generated compositions to pass, because some novel combinations might be valid but not recognized by SMACT's conservative rules.

Looking ahead to Notebook 04: when we add structural constraints — say, a kagome lattice — we're not just asking the model to generate any crystal. We're fixing the positions of certain atoms into a kagome pattern and letting the model fill in everything else. The distribution comparison becomes even more interesting there, because we want to see whether the constrained structures are still physically realistic or whether the constraints push them into unphysical territory.

This is the key advantage of the SCIGEN approach: it combines the diversity of unconditional generation with the precision of structural templates. The diffusion model handles the chemistry — picking elements, setting bond lengths, determining the lattice — while the inpainting mechanism enforces the geometry."

---

## Cell 21 — Code [3.7]: 6-subplot distribution comparison

**[Run this cell.]** Runs quickly (< 5 seconds).

"Six panels comparing generated (coral) versus training (blue) distributions.

Top left: atoms per unit cell. You want these histograms to overlap — the model should generate a range of structure sizes, not just one.

Top center: lattice parameter a. Top right: lattice parameter c. These show whether the model generates realistic cell dimensions. Look for good overlap with the blue training distribution.

Bottom left: nearest-neighbor distances. The green band marks the typical physical range of 1.5 to 3.5 angstroms. Most generated distances should fall in that band.

Bottom center: element frequency bar chart, top fifteen elements. Blue bars are training, coral bars are generated. Perfect overlap isn't expected with only fifty samples, but the relative ordering should be similar.

Bottom right: lattice angles alpha, beta, gamma. The peaks at 60, 90, and 120 degrees correspond to hexagonal, cubic, and other common crystal systems.

The bottom line is printed below: what fraction of generated structures have physical nearest-neighbor distances between 1 and 4 angstroms. A good model should be above 80 percent."

---

## Cell 22 — Markdown [3.7.1]: t-SNE composition space

"Now let's look at this from a different angle — literally. We'll embed each structure as a composition vector and project to 2D using t-SNE. If the generated structures overlap with the training data in this embedding, it means they're producing chemically realistic compositions."

> 📖 *Same text as notebook — read from screen*

Add: "t-SNE gives us a two-dimensional map of composition space. Each structure is represented as a vector of element fractions, and t-SNE projects these high-dimensional vectors down to 2D while preserving local neighborhood structure."

---

## Cell 23 — Code [3.7.1]: t-SNE embedding and scatter plot

**[Run this cell.]** Takes a few seconds.

"The plot shows blue dots for training structures and coral stars for generated structures. What you want to see: the stars mixed in with the dots, meaning generated compositions are chemically similar to training data. Stars that sit in isolation represent novel compositions not present in the training set — these could be interesting discovery candidates, or they could be unphysical. That's where the SMACT filter comes in."

---

## Cell 24 — Markdown [3.8]: 3.8 SMACT composition screening

"Section 3.8: SMACT screening. SMACT checks for electronegativity balance and charge neutrality — it's a fast first-pass filter to identify which generated compositions are chemically plausible."

> 📖 *Same text as notebook — read from screen*

Add: "SMACT is essentially a lookup table of known oxidation states. It checks: can these elements, in this ratio, form a charge-neutral compound? It's not a stability prediction — a composition can pass SMACT and still be thermodynamically unstable — but it's a fast sanity check that catches obviously nonsensical combinations."

---

## Cell 25 — Code [3.8]: SMACT filter on generated structures

**[Run this cell.]** Runs quickly (< 5 seconds).

"A table showing each generated structure, its formula, and whether it passed the SMACT charge-neutrality test. PASS means at least one valid oxidation state assignment exists. FAIL means no combination of known oxidation states can balance the charges.

At the bottom: the overall pass rate. For vanilla DiffCSP generation, you typically see 60 to 80 percent passing — the model has learned to generate mostly charge-balanced compositions, even though it was never explicitly trained on charge balance. That's an emergent property of learning from real crystal data."

---

## Cell 26 — Markdown [3.9]: 3.9 Training overview

"Section 3.9: Training overview. We're not training today, but here's how it works. You sample a random timestep t, add the appropriate noise to the crystal, then the model predicts that noise. The loss is a weighted sum of three terms: lattice noise MSE, coordinate score MSE, and type noise MSE. The important subtlety is that the coordinate loss uses the score of the wrapped normal distribution, not the raw noise — this accounts for the periodic boundary conditions. The model was trained for roughly one thousand epochs."

> 📖 *Same text as notebook — read from screen*

Add: "We're not training today — that would take hours on a full GPU node. But it's important to understand the training objective. The key subtlety is that the coordinate loss uses the score of the wrapped normal, not just the raw noise residual. This accounts for the periodicity: the gradient of the log-probability of the wrapped normal has contributions from all periodic images, not just the nearest one. Getting this right is essential for generating valid fractional coordinates."

---

## Cell 27 — Markdown [3.10]: 3.10 DiffCSP vs SCIGEN comparison table

"Section 3.10: DiffCSP versus SCIGEN. This table summarizes the key differences. DiffCSP is unconditional — all atoms are free, the lattice is unconstrained, and it's designed for general crystal generation. SCIGEN adds constraints: known atoms are inpainted into fixed positions, the lattice can be constrained, and the goal shifts to targeted discovery of materials with specific geometric motifs."

> 📖 *Same text as notebook — read from screen*

Add: "This table is the bridge to the next notebook. Everything in the left column — vanilla DiffCSP — is what we just did. Everything in the right column — SCIGEN with constraints — is what we'll do next. The architecture is the same. The noise model is the same. The only difference is that SCIGEN fixes certain atoms into a geometric pattern and lets the model fill in the rest. That simple change enables targeted discovery of materials with exotic lattice geometries — kagome magnets, honeycomb topological materials, and so on."

---

## Cell 28 — Markdown [3.10]: References

"For references, the key papers are: the DiffCSP paper by Jiao et al. at NeurIPS 2023 for the full mathematical derivation of wrapped normal diffusion on crystals; the SCIGEN paper in Nature Materials 2025 for the constrained generation extension; the original DDPM paper by Ho, Jain, and Abbeel that all of this builds on; and pymatgen, which we use for structure manipulation throughout."

> 📖 *Same text as notebook — read from screen*

Add: "These are the key papers if you want to go deeper. The DiffCSP paper has the full mathematical derivation of the wrapped normal diffusion. The SCIGEN paper shows how inpainting extends this to constrained generation. And the original DDPM paper by Ho, Jain, and Abbeel is the foundation that all of this builds on."

---

## Cell 29 — Markdown [3.10]: What's next?

"What's next? In Notebook 04, we move to SCIGEN Generation — where we add structural constraints and generate targeted materials with specific lattice geometries."

> 📖 *Same text as notebook — read from screen*

Add: "Notebook 04 is the capstone of this tutorial series. We'll take everything we've learned about diffusion and add structural constraints — kagome, honeycomb, triangular lattices. You'll see the model generate materials that have specific geometric patterns baked in, and then in Notebook 05, we'll evaluate those generated structures with machine-learning interatomic potentials. Let's move on."
