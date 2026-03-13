# APS Tutorial Outline: T4 — Generative AI for Physics: From Models to Materials

**Date:** Sunday, March 15, 2026, 8:30 a.m. – 12:30 p.m.
**Location:** Colorado Convention Center, Room 109/111
**Duration:** 4 hours (with breaks)
**Audience:** Graduate students, post-docs, and researchers in physics, materials science, and computational science

---

## Design Principles

1. **Physics-first, not ML-first.** Start from materials problems, not from neural network architectures. The audience are physicists — lead with "why does this matter for discovering materials?" not "here is how a U-Net works."
2. **Hands-on over slides.** Every conceptual section should be paired with something the audience can run. Even simple code (load a structure, visualize it) builds confidence.
3. **Scaffolded complexity.** Each section should build on the previous one. No section should require knowledge that hasn't been introduced yet.
4. **Fail gracefully.** Prefer robust, pre-tested notebooks over ambitious live coding. Have fallback outputs ready for any cell that depends on network access or GPU.
5. **Respect the clock.** 4 hours sounds long but goes fast. Budget time generously for setup issues — Colab installs, GPU allocation, WiFi problems.

---

## Recommended Overall Flow

```
8:30 – 8:45   [0] Welcome & Setup
8:45 – 9:15   [1] Crystal Structures: Representation & Tools
9:15 – 9:45   [2] Accessing Materials Data (Materials Project)
9:45 – 10:00  ── Break ──
10:00 – 10:40 [3] Generative AI Concepts for Physicists
10:40 – 11:10 [4] Evaluating Generated Materials (MLIPs)
11:10 – 11:20 ── Break ──
11:20 – 12:10 [5] SCIGEN: Constrained Crystal Generation (Capstone)
12:10 – 12:30 [6] Wrap-up, Landscape & Resources
```

---

## Section-by-Section Breakdown

### [0] Welcome & Setup (15 min)

**Purpose:** Get everyone on the same page technically. Minimize setup surprises later.

**Why it belongs:** If 50 people spend 10 minutes each debugging Colab setup during the SCIGEN demo, the session is ruined. Front-load the friction.

**Level:** Basic

**Format:** Live demo + follow-along

**Content:**
- Open Google Colab, verify GPU runtime (Runtime > Change runtime type > T4)
- Clone the tutorial repo, run the dependency install cell
- While installs run (~5-10 min for PyG), give a brief intro:
  - Who are the instructors, what will we cover today
  - What you'll walk away with: ability to load/inspect/generate crystal structures programmatically
  - Quick poll: "Who has used pymatgen? Who has used a generative model? Who has trained a neural network?"
- Verify install succeeded before moving on

**Key decision:** Start the install FIRST, talk WHILE it runs. Don't make people wait in silence.

**Notebook:** `notebooks/00_setup.ipynb` (new — minimal: clone, install, verify imports)

---

### [1] Crystal Structures: Representation & Tools (30 min)

**Purpose:** Ensure everyone can load, inspect, visualize, and save a crystal structure in Python. This is the vocabulary for everything that follows.

**Why it belongs:** You cannot appreciate generated materials if you don't know what a crystal structure looks like computationally. Many physicists work with structures conceptually but have never manipulated one in code.

**Level:** Basic → Intermediate

**Format:** Live demo + follow-along notebook

**Content:**
- What is a crystal structure? (lattice + basis, fractional vs. Cartesian coordinates, space groups)
  - Keep this to 2-3 minutes — the audience knows the physics, just ground the terminology
- **pymatgen** basics:
  - Create a `Structure` from scratch (lattice parameters + species + coordinates)
  - Load from CIF file
  - Inspect: `structure.composition`, `structure.lattice`, `structure.frac_coords`
  - Visualize (simple matplotlib 3D scatter, or use `crystal_toolkit` if available)
  - Save to CIF
- **Why pymatgen over ASE?** Brief mention — pymatgen is the standard for materials informatics, ASE is more for atomistic simulations. We use pymatgen here because SCIGEN outputs pymatgen structures.
- Quick exercise: "Load this CIF, print the formula, how many atoms per unit cell?"

**What to cut:**
- Do NOT teach ASE alongside pymatgen. Pick one. pymatgen is the right choice for this workflow.
- Do NOT go deep into symmetry/space groups — mention them, don't implement them.
- Do NOT cover POSCAR/VASP formats — CIF is sufficient for this tutorial.

**Notebook:** `notebooks/01_crystal_structures.ipynb` (new)

---

### [2] Accessing Materials Data — Materials Project (30 min)

**Purpose:** Show how to query real materials databases. This grounds the tutorial in real data and motivates "why would we want to generate NEW structures?"

**Why it belongs:** The audience needs to understand the data landscape before they can appreciate generative models. Also, MP API skills are immediately useful — attendees will use this next week in their own research.

**Level:** Basic → Intermediate

**Format:** Live demo + follow-along notebook

**Content:**
- What is the Materials Project? (brief — most attendees will have heard of it)
- Getting an API key (attendees should pre-register, but show the signup flow)
- **mp-api** basics:
  - Query by formula: `mpr.materials.search(formula="MnO2")`
  - Query by elements: all structures containing Mn
  - Query by properties: band gap, formation energy, stability (energy above hull)
  - Download a structure, inspect it with pymatgen (connects to Section 1)
- Key concept: **the training data for generative models comes from databases like MP**
  - Show that SCIGEN's training set (MP_20) is a subset of Materials Project
  - "There are ~150,000 known stable inorganic materials. Can we generate new ones?"
- Quick exercise: "Find all kagome-lattice materials containing Mn in MP. How many are there?"
  - This directly motivates SCIGEN: "What if we could generate MORE of these?"

**What to cut:**
- Do NOT cover the full MP API (dozens of endpoints). Just `materials.search` and `materials.get_structure_by_material_id`.
- Do NOT discuss MP's internal DFT workflow — it's interesting but off-topic.
- Do NOT require attendees to have an API key for every cell. Have pre-downloaded example data as fallback.

**Notebook:** `notebooks/02_materials_project.ipynb` (new)

---

### [3] Generative AI Concepts for Physicists (40 min)

**Purpose:** Build intuition for what generative models do, how they work, and why they matter for materials science. This is the conceptual core of the tutorial.

**Why it belongs:** The session title is "Generative AI for Physics." Attendees came to understand these ideas. But they're physicists — explain with physics analogies, not ML jargon.

**Level:** Intermediate (conceptual)

**Format:** Conceptual with minimal code (mostly diagrams/slides + short demos)

**Content:**

**Part A: What is a generative model? (10 min)**
- Discriminative vs. generative: "classifying images of cats" vs. "creating new cat images"
- For materials: "predicting if a structure is stable" vs. "generating a new stable structure"
- The key idea: learning a probability distribution over data
  - Physics analogy: a generative model learns the "partition function" of the materials space
- Brief taxonomy (keep high-level, ~1 slide each):
  - **VAE:** Encode to latent space, decode back. Think of it as a compressed representation of crystal space.
  - **GAN:** Generator vs. discriminator game. Powerful but hard to train, mode collapse issues.
  - **Diffusion models:** Start with noise, gradually denoise. Like running a Langevin dynamics simulation backwards.
  - **Flow-based:** Invertible transformations. Elegant but expensive.
- **Key message:** "Diffusion models have become the dominant approach for crystal generation because they naturally handle the periodicity and symmetry constraints of crystal structures."

**Part B: Diffusion models — the physics intuition (15 min)**
- Forward process: gradually add noise to a crystal structure (displace atoms, randomize types, distort lattice)
  - Analogy: heating a crystal until it melts into a disordered liquid
- Reverse process: learn to denoise step by step
  - Analogy: "controlled crystallization from a melt" — the model learns the crystallization dynamics
- What the neural network learns: the score function ∇log p(x), i.e., "which direction to move atoms to make the structure more crystal-like"
- Optional mini-demo: show a 2D diffusion process on a simple distribution (e.g., Gaussian mixture)
  - This can be a pre-computed animation, not live training

**Part C: Conditioning and constraints (10 min)**
- Unconditional generation: "generate any crystal"
- Conditional generation: "generate a crystal with these properties"
  - Property-guided: target band gap, formation energy
  - Composition-guided: fix the elements
  - **Structure-guided: fix the lattice geometry** → this is what SCIGEN does
- Classifier-free guidance (brief mention — "a technique to strengthen the conditioning signal")
- **Key message:** "The power of generative models for materials isn't just making new structures — it's making structures with DESIRED properties."

**Part D: The landscape of crystal generative models (5 min)**
- Brief survey (1 slide, table format):
  - CDVAE (Xie et al., 2022) — VAE-based
  - DiffCSP (Jiao et al., 2023) — diffusion for crystal structure prediction
  - SCIGEN (Okabe et al., 2025) — diffusion with structural constraints
  - UniMat, MatterGen, etc. — brief mentions
- Position SCIGEN: "What makes SCIGEN unique is that it allows you to specify geometric constraints — like 'I want a kagome lattice' — and generate materials that satisfy those constraints."

**What to cut:**
- Do NOT teach the math of diffusion models in detail (no ELBO derivations, no score matching proofs)
- Do NOT train any model live — training takes hours/days, not minutes
- Do NOT go deep into GAN or VAE architectures — mention them, don't implement them
- Do NOT cover image/text generation — stay focused on crystal structures

**What to emphasize for APS audience:**
- Physics analogies (Langevin dynamics, partition functions, crystallization)
- The symmetry challenges unique to crystals (periodicity, permutation invariance, space groups)
- Why materials generation is harder than image generation (discrete atoms, periodic boundaries, physical validity)

**Notebook:** `notebooks/03_generative_concepts.ipynb` (new — mostly markdown/images with 1-2 simple demos)

---

### [4] Evaluating Generated Materials — MLIPs (30 min)

**Purpose:** Once you generate a structure, how do you know if it's any good? Introduce MLIPs as fast surrogates for DFT, used to screen generated candidates.

**Why it belongs:** Generation without evaluation is useless. This section bridges "we made some structures" to "which ones are actually promising." It also introduces tools (CHGNet/MACE) that attendees can use in their own research.

**Level:** Intermediate

**Format:** Live demo + follow-along notebook

**Content:**
- The evaluation problem: "We generated 1000 structures. Which ones are stable? Which ones have interesting properties?"
- Traditional approach: DFT relaxation (expensive — days per structure)
- Modern approach: Machine Learning Interatomic Potentials (MLIPs) — evaluate in seconds
- **CHGNet** demo (recommended — good Colab support, universal potential):
  - Load a structure (from Section 1 or MP)
  - Predict energy, forces, stress
  - Run a structure relaxation
  - "Is this structure stable?" → check energy above hull
- **MACE-MP-0** as alternative mention (don't demo both — pick one)
- Connect to SCIGEN: "SCIGEN includes GNN-based screening models that play a similar role"
  - Show the gnn_eval/ directory exists, explain its purpose
  - "After generating candidates with SCIGEN, we screen them for stability and magnetic properties"
- Quick exercise: "Relax this generated structure with CHGNet. Did the energy go down? Did the structure change much?"

**What to cut:**
- Do NOT train an MLIP from scratch — way too slow for a tutorial
- Do NOT compare 5 different MLIPs — just use CHGNet (or MACE), acknowledge others exist
- Do NOT discuss the details of equivariant GNNs / E(3)-invariance unless asked
- Do NOT include MatGL/M3GNet as separate demos — mention them in a comparison table at most

**What to emphasize for APS audience:**
- The physics: what is a potential energy surface, what does "relaxation" mean
- The speedup: DFT = hours per structure, MLIP = seconds
- The limitation: MLIPs are trained on DFT data, so they inherit DFT's accuracy (and errors)

**Notebook:** `notebooks/04_mlip_evaluation.ipynb` (new)

---

### [5] SCIGEN: Constrained Crystal Generation — Capstone (50 min)

**Purpose:** The main event. Show how SCIGEN generates crystal structures with targeted geometric constraints. This is your contribution and the most unique part of the tutorial.

**Why it belongs:** This is what the attendees came to see. Everything before this was building toward it.

**Level:** Intermediate → Advanced

**Format:** Live demo + follow-along notebook (existing `tutorial_colab.ipynb`, refined)

**Content:**

**Part A: What is SCIGEN? (10 min, mostly slides/markdown)**
- The paper: Nature Materials 2025
- The problem: "I want to discover new kagome-lattice magnets. How do I generate crystal structures that have a kagome sublattice?"
- The approach: diffusion model with structural constraints
  - Fix some atomic positions to form the desired lattice pattern
  - Let the model fill in the rest (additional atoms, lattice parameters, atom types)
- Available constraints: kagome, honeycomb, triangular, square, etc. (show the table)
- Why this matters: kagome magnets are a hot topic in condensed matter physics (frustrated magnetism, topological states, flat bands)

**Part B: Hands-on generation (25 min)**
- Load the pretrained model (should already be downloaded from setup)
- Configure: pick a lattice type (kagome), element (Mn), batch size (4)
- Run generation — watch the diffusion process
- Inspect results: compositions, lattice parameters
- Visualize: 3D scatter plots of generated unit cells
- Export as CIF files
- Try a different lattice type (honeycomb) or element (Fe) — show the flexibility

**Part C: Understanding the outputs (10 min)**
- What do the generated structures look like? Are they reasonable?
- How to evaluate: use pymatgen to check distances, coordination, composition validity
- Connect to Section 4: "You could now relax these with CHGNet and check stability"
- Show the screening pipeline (conceptual — `eval_screen.py` exists but don't run it live)
- Mention the published results: "From 10 million generated structures → 1 million after screening → 24,743 DFT-relaxed"

**Part D: Try it yourself (5 min buffer)**
- Let attendees experiment: change SC_TYPE, ELEMENT, BATCH_SIZE
- Encourage questions

**Notebook:** `notebooks/05_scigen_generation.ipynb` (refactored from existing `tutorial_colab.ipynb`)

---

### [6] Wrap-up: The Landscape & Resources (20 min)

**Purpose:** Place everything in context. Give attendees a roadmap for continuing after the tutorial.

**Why it belongs:** Attendees need to know where to go next. A tutorial without a "what now?" ending feels incomplete.

**Level:** All levels

**Format:** Slides/discussion (no code)

**Content:**
- Summary: what we covered today (crystal structures → data access → generative AI → evaluation → SCIGEN)
- The broader landscape of generative AI for materials:
  - Crystal structure generation (CDVAE, DiffCSP, SCIGEN, MatterGen, UniMat)
  - Molecule generation (different problem — no periodicity)
  - Quantum device optimization (mentioned in session description)
  - Protein/biomolecule generation (analogous ideas in a different domain)
- Open challenges:
  - Synthesizability: "The model says this structure is stable, but can I actually make it in a lab?"
  - Novelty vs. validity trade-off
  - Benchmarking: how do we compare generative models fairly?
  - Multi-objective optimization: stable AND magnetic AND synthesizable
- Resources and links:
  - SCIGEN repo and paper
  - Materials Project
  - Pymatgen documentation
  - CHGNet / MACE repos
  - Key review papers
- Q&A

**Notebook:** None (slides only)

---

## What to Cut or De-emphasize

| Topic | Recommendation | Reason |
|-------|---------------|--------|
| **ASE** | Cut entirely | Pymatgen is sufficient; adding ASE doubles the learning surface for no gain in this context |
| **DiffCSP / CDVAE demos** | Mention only (1 slide) | Demoing multiple generative models would be confusing and setup-heavy. SCIGEN is the capstone. |
| **Training a model** | Cut entirely | Too slow (hours), too complex, and not the learning objective. Use pretrained models only. |
| **Benchmarking** | Brief mention in wrap-up | Important topic but too meta for a hands-on tutorial. Mention it exists, point to papers. |
| **Multiple MLIP comparisons** | Cut — pick one (CHGNet) | Comparing 3+ MLIPs is a separate tutorial. Show one well, mention others. |
| **Detailed math (ELBO, score matching)** | Cut | The audience is physicists, not ML researchers. They want intuition, not derivations. |
| **Space group analysis** | De-emphasize | Mention that pymatgen can do symmetry analysis, but don't make it a section. |
| **VASP/QE/DFT tools** | Cut | Off-topic for a generative AI tutorial. |

---

## What Would Make This Tutorial Especially Strong for APS

### 1. Physics-First Framing
Every section should start with a physics question, not a code snippet:
- "What defines a kagome lattice?" → then show how to represent it in code
- "Why are frustrated magnets interesting?" → then show how SCIGEN generates kagome materials
- "How do we know if a generated structure is stable?" → then introduce MLIPs

### 2. Real-World Impact Stories
Briefly mention real discoveries or applications:
- "SCIGEN generated candidates that were validated by DFT — these are real materials predictions"
- "MLIPs like CHGNet are now used in high-throughput screening pipelines at national labs"
- "The Materials Project database has been cited >10,000 times"

### 3. Hands-On Confidence Building
Many attendees may be intimidated by Python. Ensure:
- Every notebook has a "Run this cell and you should see this output" checkpoint
- Cells that might fail (network, GPU) have fallback outputs pre-computed
- Error messages are human-readable ("If you see this error, try X")

### 4. Take-Home Value
Attendees should leave with:
- A working Colab notebook they can reuse
- Knowledge of pymatgen and MP API (useful regardless of generative AI)
- Understanding of what generative models can (and can't) do for materials
- Awareness of SCIGEN and how to use it for their own research

### 5. Kagome/Frustrated Magnets as a Running Example
Use kagome lattice materials as a thread throughout the tutorial:
- Section 1: "Here is a kagome lattice structure in pymatgen"
- Section 2: "How many kagome materials are in the Materials Project?"
- Section 3: "How would a diffusion model generate a new kagome material?"
- Section 4: "How do we check if a generated kagome material is stable?"
- Section 5: "Let's generate kagome materials with SCIGEN"

This creates narrative coherence and makes the tutorial feel like a story, not a disconnected set of demos.

---

## How SCIGEN Should Be Introduced and Framed

### Positioning
SCIGEN should be presented as the **culmination of everything learned in the tutorial**:
- We learned how to represent crystal structures (Section 1) → SCIGEN generates them
- We learned where materials data comes from (Section 2) → SCIGEN was trained on this data
- We learned what diffusion models are (Section 3) → SCIGEN is a diffusion model
- We learned how to evaluate structures (Section 4) → SCIGEN's outputs can be screened this way

### Key Messages
1. **Novel capability:** SCIGEN is the first generative model that lets you specify geometric constraints on the output. This is a new capability, published in Nature Materials.
2. **Practical relevance:** Kagome magnets, honeycomb lattices, and other frustrated geometries are active areas of condensed matter research. SCIGEN directly targets these.
3. **Accessible:** "You can generate new materials candidates in your Colab notebook right now, with no training required."
4. **Part of a pipeline:** Generation is one step. Screening, DFT validation, and experimental synthesis come after. Show the full pipeline conceptually.

### What NOT to Do
- Don't apologize for SCIGEN's limitations — present them honestly as open challenges
- Don't spend time on the model architecture details (CSPNet, GNN decoder) — attendees care about inputs and outputs
- Don't compare SCIGEN to other models in a competitive way — position it as one tool in a growing toolkit

---

## Notebook Summary

| # | Notebook | Status | Duration | Level |
|---|----------|--------|----------|-------|
| 0 | `notebooks/00_setup.ipynb` | **New — to create** | 15 min | Basic |
| 1 | `notebooks/01_crystal_structures.ipynb` | **New — to create** | 30 min | Basic–Int |
| 2 | `notebooks/02_materials_project.ipynb` | **New — to create** | 30 min | Basic–Int |
| 3 | `notebooks/03_generative_concepts.ipynb` | **New — to create** | 40 min | Intermediate |
| 4 | `notebooks/04_mlip_evaluation.ipynb` | **New — to create** | 30 min | Intermediate |
| 5 | `notebooks/05_scigen_generation.ipynb` | **Refactor from existing** | 50 min | Int–Advanced |
| 6 | (Slides only — no notebook) | N/A | 20 min | All |

**Total notebook creation effort:** 5 new notebooks + 1 refactor

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| WiFi fails during tutorial | Pre-download all dependencies, models, and data into the notebook. Have a backup plan with pre-computed outputs. |
| Colab runs out of GPU | Show CPU fallback (slower but functional). Keep batch sizes tiny (2-4 structures). |
| PyG install takes 10+ min | Start install in Section 0, talk while it runs. If it's truly stuck, have a fallback notebook with pre-installed environment. |
| Attendees have trouble with Colab | Have a TA or co-instructor who can help with setup while you continue. |
| Section runs over time | Mark "optional" sub-sections that can be skipped. Sections 4 and 6 are the most flexible. |
| MP API key issues | Include pre-downloaded example structures as fallback. Don't gate the tutorial on everyone having an API key. |

---

## Open Questions for Ryotaro

1. **Are there other presenters for this session?** If so, what are they covering? This outline assumes you own the materials/crystal portion. If someone else covers VAEs/GANs/flows conceptually, Section 3 can be shortened.
2. **What GPU will the venue WiFi support?** If attendees can't get Colab GPUs, we need a CPU-compatible demo path.
3. **Do you want to include quantum device optimization?** The session description mentions it. If another presenter covers it, great. If not, it could be a brief mention in Section 6.
4. **Time allocation flexibility?** Is the 4-hour block all yours, or shared with other presenters?
5. **Will attendees have pre-setup instructions?** (e.g., "install X before the tutorial"). If yes, Section 0 can be shorter.
