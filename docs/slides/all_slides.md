# APS Tutorial T4: Generative AI for Physics — From Models to Materials

## SCIGEN: Structural Constraint Integration in a Generative Model for the Discovery of Quantum Materials

*Ryotaro Okabe et al., Nature Materials (2025)*

Ryotaro Okabe | rokabe@mit.edu

---

## Tutorial materials

**GitHub:** [github.com/RyotaroOKabe/APS_demo_SCIGEN](https://github.com/RyotaroOKabe/APS_demo_SCIGEN)

**Paper:** [doi.org/10.1038/s41563-025-02355-y](https://doi.org/10.1038/s41563-025-02355-y)

**Colab:** Click the badges in the README

<!-- QR CODE PLACEHOLDER: insert QR code image for the GitHub tutorial page here -->
```
 ┌─────────────────┐
 │                 │
 │   [QR CODE]     │
 │                 │
 │  Scan to open   │
 │  tutorial repo  │
 │                 │
 └─────────────────┘
```

---

## Tutorial overview

**Goal:** Learn to generate new crystal structures with targeted lattice geometries using a diffusion model, and evaluate them with machine-learning interatomic potentials.

| # | Notebook | Topic | Time |
|---|----------|-------|------|
| 00 | **Setup** | GPU, dependencies, model weights | 5 min |
| 01 | **Crystal Structures** | pymatgen, visualization, tight-binding bands | 15-20 min |
| 02 | **Generative Concepts** | Diffusion models for physicists | 10 min |
| 03 | **DiffCSP** | Crystal diffusion: noise & denoising | 10 min |
| 04 | **SCIGEN** | Constrained crystal generation (capstone) | 20 min |
| 05 | **MLIP Evaluation** | CHGNet, phonons, stability screening | 10-15 min |

---

## The big picture

```
Known materials (databases)
        |
        v
  Why are exotic lattices rare?  -->  Need generative models
        |
        v
  DiffCSP: diffusion for crystals (unconstrained)
        |
        v
  SCIGEN: add structural constraints (kagome, honeycomb, ...)
        |
        v
  Screen with MLIPs (CHGNet)  -->  DFT validation  -->  Synthesis
```

---

## Setup checklist

- [ ] Google Colab with GPU runtime (T4 or better)
- [ ] Clone the repository
- [ ] Install dependencies (~2 min)
- [ ] Download pretrained model weights (~1 min)

**Please run Notebook 00 now while I introduce the tutorial.**

---

<!-- ============================================================ -->
<!-- PART 1: CRYSTAL STRUCTURES (Notebook 01)                      -->
<!-- ============================================================ -->

# Part 1: Crystal Structures & Materials Data

---

## What is a crystal?

A crystal = **lattice** + **basis**

- **Lattice:** periodic box defined by vectors **a**, **b**, **c**
  - Lengths: *a, b, c* (Angstroms)
  - Angles: *alpha, beta, gamma* (degrees)
- **Basis:** atoms inside one unit cell
  - Species (element type)
  - Fractional coordinates (0 to 1)

The entire infinite crystal is the unit cell repeated in all directions.

---

## The three components: (L, X, A)

Every crystal in a computer is fully defined by three components:

| Component | Symbol | Description |
|---|---|---|
| **Lattice** | **L** | Periodic box: lengths (a, b, c) + angles (alpha, beta, gamma) |
| **Fractional coordinates** | **X** | Atom positions in [0, 1) |
| **Atom types** | **A** | Chemical species at each site |

Crystal = (**L**, **X**, **A**) --- this is exactly what SCIGEN denoises.

---

## Fractional vs Cartesian coordinates

| | Fractional | Cartesian |
|---|---|---|
| Range | [0, 1) | Angstroms |
| Periodicity | Natural (x and x+1 are same) | Must handle wrapping |
| Cell-independent | Yes | No |

Fractional coordinates are what diffusion models operate on.

---

## Conventional vs. primitive unit cell

The **same crystal** can be described by different unit cells:

- **Conventional:** standard cell showing full symmetry (NaCl: 8 atoms, cubic)
- **Primitive:** smallest cell that tiles space (NaCl: 2 atoms, rhombohedral)

Both produce the same infinite crystal when repeated.

---

## Graph representation of crystals

Neural networks see crystals as **graphs**:

| Graph element | Crystal meaning |
|---|---|
| **Node** | Atom (features: element, position) |
| **Edge** | Neighbor bond within cutoff radius |

Key: edges include **periodic boundary** connections --- graphs naturally encode translational symmetry.

---

## Key invariances

| Invariance | Meaning |
|---|---|
| **Permutation** | Reordering atoms doesn't change the crystal |
| **Periodic translation** | Shifting by a lattice vector = same crystal |
| **Unit cell choice** | Conventional vs. primitive = same crystal |
| **Rotation** | Rotating the crystal doesn't change properties |

SCIGEN's architecture respects all four.

---

## pymatgen: the Swiss army knife

```python
from pymatgen.core import Structure, Lattice

lattice = Lattice.cubic(5.64)
nacl = Structure(lattice, ['Na', 'Cl', ...], frac_coords)
```

- Parse CIF files, symmetry analysis (space groups)
- Neighbor searches, distances
- Interface to Materials Project

> Ong et al., *Comput. Mater. Sci.* 68, 314 (2013)

---

## The MP-20 dataset

- **45,231** structures from the Materials Project
- All with **<= 20 atoms** per unit cell
- SCIGEN's training data

We'll explore:
- Property distributions (formation energy, band gap, stability)
- Element frequency (periodic table heatmap)
- Space group distribution

> Jain et al., *APL Materials* 1, 011002 (2013)

---

## Why kagome?

The **kagome lattice** (corner-sharing triangles) creates:

- **Geometric frustration** in magnetic materials
- **Flat electronic bands** (from tight-binding)
- **Quantum spin liquids**, anomalous Hall effects

**Problem:** kagome materials are *extremely rare* in nature.

**Solution:** Generate them with SCIGEN!

---

## Tight-binding: geometry -> electronics

| Lattice | Band structure | Key feature |
|---------|---------------|-------------|
| **Kagome** | Flat band + Dirac cone | Strong correlations |
| **Honeycomb** | Dirac cone at K | Graphene-like |
| **Triangular** | Cosine band | No exotic features |

The flat band is a **geometric property** --- it's perfectly flat regardless of hopping strength.

---

<!-- ============================================================ -->
<!-- PART 2: GENERATIVE AI CONCEPTS (Notebook 02)                  -->
<!-- ============================================================ -->

# Part 2: Generative AI Concepts --- From DDPM to Materials

---

## Discriminative vs Generative

| | Discriminative | Generative |
|---|---|---|
| Task | Predict label from data | Create new data |
| Learns | P(label \| data) | P(data) |
| Example | "Is this stable?" | "Make a new crystal" |

---

## Diffusion models: the key idea

**Forward process (heating):**
Clean data --> add noise --> ... --> pure noise

**Reverse process (generation):**
Pure noise --> remove noise --> ... --> new data

*Physics analogy: melting a crystal, then controlled re-crystallization*

> Ho, Jain & Abbeel, "Denoising Diffusion Probabilistic Models," *NeurIPS* (2020)

---

## DDPM equations

**Forward** (add noise at step t):
$$q(x_t | x_0) = N(\sqrt{\bar\alpha_t} \cdot x_0,\; (1 - \bar\alpha_t) \cdot I)$$

**Training loss** (predict the noise):
$$L = E[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 ]$$

**Reverse** (denoise):
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta\right) + \sigma_t z$$

---

## Classifier-free guidance

Train the model to work both WITH and WITHOUT class labels:
- Randomly drop the label during training (with probability p=0.1)
- At generation time, interpolate between conditional and unconditional predictions:

$$\epsilon = (1 + w) \cdot \epsilon_\text{conditional} - w \cdot \epsilon_\text{unconditional}$$

- w = 0: unconditional | w = 2: strong class guidance

> Ho & Salimans, "Classifier-Free Diffusion Guidance," *NeurIPS Workshop* (2022)

---

## Hands-on: DDPM on MNIST

We train a real diffusion model on handwritten digits:
- **UNet** with time + class embedding (~1.7M parameters)
- **2 epochs** on MNIST (~5 min on T4 GPU)
- **Conditional generation**: specify which digit to generate
- **Guidance strength**: control the class-specificity

Demo:
1. Train the model (watch loss decrease)
2. Generate all 10 digit classes
3. Visualize the denoising trajectory (noise -> digit)

---

## From images to crystals

| MNIST (images) | Materials (crystals) |
|----------------|---------------------|
| Pixel values [0, 1] | Fractional coordinates [0, 1) |
| 28x28 grid | Variable number of atoms |
| UNet predicts noise | GNN (CSPNet) predicts noise |
| Class label (0-9) | Structural constraint (kagome, ...) |
| Classifier-free guidance | Inpainting with fixed positions |

Same DDPM core --- different representations and network architectures.

---

## Crystal generative models

| Model | Year | Key feature | Reference |
|-------|------|-------------|-----------|
| CDVAE | 2022 | First crystal generative model (VAE) | Xie et al., *ICLR* (2022) |
| DiffCSP | 2023 | Diffusion for crystal structure prediction | Jiao et al., *NeurIPS* (2023) |
| **SCIGEN** | **2025** | **Structural constraints via inpainting** | **Okabe et al., *Nat. Mater.* (2025)** |
| MatterGen | 2025 | Property-guided generation | Zeni et al., *Nature* (2025) |

All build on the DDPM framework we just trained on MNIST.

---

<!-- ============================================================ -->
<!-- PART 3: DiffCSP --- CRYSTAL DIFFUSION (Notebook 03)           -->
<!-- ============================================================ -->

# Part 3: DiffCSP --- Crystal Diffusion

> Jiao et al., "Crystal Structure Prediction by Joint Equivariant Diffusion," *NeurIPS* (2023)

---

## DiffCSP: diffusion for crystals

**Three things to diffuse simultaneously:**

| Component | Representation | Noise model |
|-----------|---------------|-------------|
| Atom positions | Frac coords F in [0,1)^{Nx3} | **Wrapped normal** |
| Lattice | 3x3 matrix L | Standard Gaussian |
| Atom types | One-hot A in R^{NxK} | Standard Gaussian |

Key insight: **fractional coordinates are periodic**
(x = 0.95 + 0.1 = 0.05, not 1.05)

---

## The wrapped normal distribution

Standard Gaussian doesn't respect periodicity.

**Wrapped normal:** sum over all periodic images

$$WN(x; \mu, \sigma^2) = \sum_{k=-\infty}^{\infty} N(x + k;\; \mu, \sigma^2)$$

This ensures the noise "wraps around" the unit cell boundaries.

The **score** of the wrapped normal is used in the loss function,
not simple Gaussian noise prediction.

---

## Noise schedules

**Lattice:** Cosine schedule for alpha_bar_t
- alpha_bar goes from ~1 (clean) to ~0 (pure noise)
- Controls how much of the original lattice survives

**Coordinates:** Exponential sigma schedule
- sigma goes from 0.005 (tiny perturbation) to 0.5 (large noise)
- Separate from lattice schedule because coordinates are periodic

---

## Forward process visualization

```
t=0          t=200        t=500        t=800        t=999
[crystal]    [noisy]      [very noisy] [almost gone] [pure noise]
```

Watch:
- Atoms scatter from lattice sites
- Unit cell distorts and eventually collapses
- Atom types become random

---

## Reverse process: predictor-corrector

**Predictor** (reverse SDE step):
$$L_{t-1} = \frac{1}{\sqrt{\alpha}} (L_t - \epsilon_\text{pred}) + \sigma z$$

**Corrector** (Langevin dynamics for coordinates):
$$F_{t-1/2} = F_t + \eta \cdot s_\text{pred} + \sqrt{2\eta}\, z \pmod{1}$$

The corrector step refines the coordinate prediction using
the score function, with periodic wrapping.

> Song et al., "Score-Based Generative Modeling through SDEs," *ICLR* (2021)

---

## DiffCSP -> SCIGEN: what changes?

| | DiffCSP | SCIGEN |
|---|---------|--------|
| Generation | Unconditional | Constrained |
| Atom positions | All freely diffused | Known atoms inpainted |
| Lattice | Free 3x3 matrix | Constrained geometry |
| Use case | General generation | Targeted discovery |

SCIGEN = DiffCSP + structural constraints (inpainting)

---

<!-- ============================================================ -->
<!-- PART 4: SCIGEN --- CONSTRAINED GENERATION (Notebook 04)       -->
<!-- ============================================================ -->

# Part 4: SCIGEN --- Constrained Crystal Generation (Capstone)

> Okabe et al., *Nature Materials* (2025)

---

## SCIGEN: the key innovation

DiffCSP generates *any* crystal. But we want *specific* geometries.

**SCIGEN adds structural constraints via inpainting:**

1. Specify a geometric pattern (kagome, honeycomb, triangular, ...)
2. Fix those atom positions during generation
3. The model fills in the rest of the crystal

Like painting a picture with some pixels already fixed.

---

## Available structural constraints

| Code | Lattice | Known atoms | Physics |
|------|---------|-------------|---------|
| `kag` | **Kagome** | 3 | Frustrated magnets, flat bands |
| `hon` | **Honeycomb** | 2 | Dirac fermions, topological |
| `tri` | **Triangular** | 1 | Simplest 2D lattice |
| `sqr` | **Square** | 1 | Cuprate-like |
| `lieb` | **Lieb** | 3 | Flat bands |
| ... | + 8 more | | Complex tilings |

---

## The inpainting approach

```
Step 1:  Define constraint         [kagome Mn atoms at fixed positions]
Step 2:  Start from noise          [all other atoms random]
Step 3:  Denoise (1000 steps)      [model generates complementary atoms]
Step 4:  Reveal constraint         [replace model's kagome sites with true positions]
Step 5:  Final structure           [complete crystal with kagome sublattice]
```

The model learns to *anticipate* the constraint during training.

---

## Generation parameters

```python
SC_TYPE    = 'kag'    # Structural constraint
ELEMENT    = 'Mn'     # Element for constrained sites
BATCH_SIZE = 4        # Structures per batch
STEP_LR    = 5e-6     # Diffusion step size
```

Try different combinations:
- `'hon'` + `'Fe'` --- honeycomb iron compounds
- `'tri'` + `'Co'` --- triangular cobalt
- `'van'` --- unconstrained (DiffCSP mode)

---

## What we generate and analyze

For each generated structure:

1. **Composition** --- what elements did the model choose?
2. **Lattice parameters** --- are a, b, c, angles reasonable?
3. **Bond distances** --- NN distances in physical range (1.5-3.5 A)?
4. **Space group** --- what symmetry does the structure have?
5. **SMACT screening** --- is the composition charge-balanced?
6. **XRD pattern** --- fingerprint for experimental comparison

> SMACT: Davies et al., *JOSS* 4, 1361 (2019)

---

## The SCIGEN screening pipeline (10M -> 24K)

```
Choose constraint (kagome, honeycomb, ...)
    |
    v  Generate 10,000,000 candidates
    |
    v  Pre-screen: composition + bond distances
    |
    v  MLIP screen: CHGNet / GNN models
    |
    v  DFT validation
    |
    v  24,743 validated materials
    |
    v  Experimental synthesis
```

---

<!-- ============================================================ -->
<!-- PART 5: MLIP EVALUATION (Notebook 05)                         -->
<!-- ============================================================ -->

# Part 5: Evaluating Generated Materials --- MLIPs

---

## Why evaluate?

Generation without evaluation is useless.

For each generated structure, we need to ask:
1. **Physically reasonable?** (bond lengths, coordination)
2. **Thermodynamically stable?** (low energy, near convex hull)
3. **Dynamically stable?** (no imaginary phonon frequencies)
4. **Interesting properties?** (magnetic, electronic)

---

## DFT vs MLIP

| | DFT (VASP/QE) | MLIP (CHGNet) |
|---|---|---|
| Accuracy | High (reference) | Good (~30 meV/atom MAE) |
| Speed | Hours per structure | **Seconds** per structure |
| Scaling | O(N^3) | O(N) |
| Use case | Final validation | **Initial screening** |

For 10,000 candidates: DFT = weeks, CHGNet = minutes.

---

## CHGNet: universal potential

**Crystal Hamiltonian Graph Neural Network**

- Trained on Materials Project (> 1.5M DFT calculations)
- Predicts: energy, forces, stress, magnetic moments
- Works for any inorganic material
- ~400K parameters, runs on GPU

```python
from chgnet.model.model import CHGNet
chgnet = CHGNet.load()
prediction = chgnet.predict_structure(struct)
```

> Deng et al., *Nature Machine Intelligence* 5, 1031 (2023)

---

## Structure relaxation demo

Start from a **deliberately perturbed** structure:
1. Take a known stable crystal
2. Add random noise (~0.3 A) to atom positions
3. Relax with CHGNet
4. Watch energy decrease and atoms settle

**Three things to visualize:**
- Energy trajectory (step by step)
- Force convergence (log scale)
- Animated structure (atoms moving to equilibrium)

---

## Phonon calculation

**Phonons = lattice vibrations**

- No imaginary frequencies -> dynamically stable
- Imaginary frequencies -> structure is a saddle point

We use ASE + CHGNet:
1. Finite displacement method (2x2x2 supercell)
2. Compute force constants
3. Plot phonon DOS

> ASE: Larsen et al., *J. Phys.: Condens. Matter* 29, 273002 (2017)

---

## Energy above hull (E_hull)

The **convex hull** defines thermodynamic stability:

- E_hull = 0: on the hull (stable)
- E_hull < 0.025 eV/atom: likely synthesizable
- E_hull > 0.1 eV/atom: will decompose

```python
from pymatgen.analysis.phase_diagram import PhaseDiagram
pd = PhaseDiagram(entries)
e_hull = pd.get_e_above_hull(entry)
```

---

## Screening SCIGEN-generated materials

Apply CHGNet to **SCIGEN-generated structures**:
1. Predict energy, forces, **magnetic moments** across constraint types
2. Compare energy/volume distributions with MP-20 training data
3. Relax structures, visualize before/after displacement
4. Compute **E_hull** for generated materials
5. Rank **top candidates** by stability + magnetic properties

**Result:** Rapid identification of stable magnetic candidates.

---

## The full discovery pipeline

| Stage | Tool | Throughput |
|-------|------|-----------|
| Generate | SCIGEN diffusion | ~1,000/hr |
| Pre-screen | Composition + bonds | instant |
| **MLIP screen** | **CHGNet** | **~1,000/hr** |
| DFT validate | VASP | ~10/hr |

10M generated -> 100K pre-screened -> **24,743 DFT-confirmed**

MLIP screening reduced DFT workload by **~400x**.

---

<!-- ============================================================ -->
<!-- SUMMARY & CONCLUSION                                          -->
<!-- ============================================================ -->

# Summary & Conclusion

---

## What we covered today

| Notebook | Key concept | Takeaway |
|----------|------------|----------|
| **01** Crystal Structures | (L, X, A) representation | Crystals = lattice + coords + atom types |
| **02** Generative AI | DDPM on MNIST | Forward = noise, Reverse = generate |
| **03** DiffCSP | Wrapped normals, predictor-corrector | Diffusion works on periodic crystals |
| **04** SCIGEN | Inpainting with constraints | Generate targeted lattice geometries |
| **05** MLIP Evaluation | CHGNet screening | Fast stability assessment (~3000x vs DFT) |

---

## The pipeline: from noise to discovery

```
  Random noise
       |
       v
  SCIGEN diffusion (1000 steps, ~5 sec/structure)
       |
       v
  Structural constraint inpainting (kagome, honeycomb, ...)
       |
       v
  10,000,000 candidate structures
       |
       v
  Pre-screen: composition + bond distance filters  -->  ~100,000 pass
       |
       v
  CHGNet: energy, forces, magnetic moments          -->  ~30,000 pass
       |
       v
  DFT validation (VASP)                             -->  24,743 confirmed
       |
       v
  Experimental synthesis candidates
```

---

## Key numbers

| Metric | Value |
|--------|-------|
| Training data | 45,231 structures (MP-20) |
| Constraint types | 13 lattice geometries |
| Candidates generated | 10,000,000 |
| DFT-validated materials | **24,743** |
| Generation speed | ~5 sec/structure (GPU) |
| CHGNet screening | ~1 sec/structure (GPU) |
| DFT speedup via MLIP | **~3,000x** |
| CHGNet energy MAE | ~30 meV/atom |

---

## Key takeaways

1. **Crystal = (L, X, A)** --- lattice + fractional coordinates + atom types
2. **Diffusion models** generate crystals by learning to reverse a noising process
3. **Wrapped normals** handle the periodicity of fractional coordinates
4. **SCIGEN's inpainting** enables targeted generation of exotic lattice geometries
5. **MLIPs (CHGNet)** make screening 1000x faster than DFT
6. **The full pipeline** reduced 10M candidates to 24,743 DFT-validated materials
7. **Kagome materials** --- rare in nature but now accessible via constrained generation

---

## Try it yourself

- Change `SC_TYPE` to explore different lattice geometries:
  - `'kag'` (kagome), `'hon'` (honeycomb), `'tri'` (triangular), `'sqr'` (square), `'lieb'` (Lieb)
- Change `ELEMENT` to explore different chemistries:
  - `'Mn'`, `'Fe'`, `'Co'`, `'Ni'`, `'Ti'`
- Evaluate your structures with CHGNet in Notebook 05

---

## References

**Crystal generation:**
- **SCIGEN:** Okabe et al., "Structural constraint integration in a generative model for the discovery of quantum materials," *Nature Materials* (2025). [DOI](https://doi.org/10.1038/s41563-025-02355-y)
- **DiffCSP:** Jiao et al., "Crystal Structure Prediction by Joint Equivariant Diffusion," *NeurIPS* (2023). [arXiv:2309.04475](https://arxiv.org/abs/2309.04475)
- **CDVAE:** Xie et al., "Crystal Diffusion Variational Autoencoder," *ICLR* (2022). [arXiv:2110.06197](https://arxiv.org/abs/2110.06197)
- **MatterGen:** Zeni et al., "A generative model for inorganic materials design," *Nature* (2025). [DOI](https://doi.org/10.1038/s41586-025-08628-5)

**Diffusion models:**
- **DDPM:** Ho, Jain & Abbeel, "Denoising Diffusion Probabilistic Models," *NeurIPS* (2020). [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- **Classifier-Free Guidance:** Ho & Salimans, *NeurIPS Workshop* (2022). [arXiv:2207.12598](https://arxiv.org/abs/2207.12598)
- **Score-Based SDEs:** Song et al., "Score-Based Generative Modeling through SDEs," *ICLR* (2021). [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)

**Evaluation tools:**
- **CHGNet:** Deng et al., "CHGNet as a pretrained universal neural network potential," *Nature Machine Intelligence* 5, 1031 (2023). [DOI](https://doi.org/10.1038/s42256-023-00716-3)
- **pymatgen:** Ong et al., *Comput. Mater. Sci.* 68, 314 (2013). [GitHub](https://github.com/materialsproject/pymatgen)
- **ASE:** Larsen et al., *J. Phys.: Condens. Matter* 29, 273002 (2017). [wiki.fysik.dtu.dk/ase](https://wiki.fysik.dtu.dk/ase/)
- **SMACT:** Davies et al., *JOSS* 4, 1361 (2019). [GitHub](https://github.com/WMD-group/SMACT)

**Databases:**
- **Materials Project:** Jain et al., *APL Materials* 1, 011002 (2013). [materialsproject.org](https://materialsproject.org)
- **SCIGEN dataset (24,743 structures):** [Figshare](https://doi.org/10.6084/m9.figshare.27778134)

---

## Tutorial materials & QR code

**GitHub:** [github.com/RyotaroOKabe/APS_demo_SCIGEN](https://github.com/RyotaroOKabe/APS_demo_SCIGEN)

<!-- QR CODE PLACEHOLDER: replace with actual QR code image -->
```
 ┌─────────────────┐
 │                 │
 │   [QR CODE]     │
 │                 │
 │  Scan to open   │
 │  tutorial repo  │
 │                 │
 └─────────────────┘
```

**Paper:** [doi.org/10.1038/s41563-025-02355-y](https://doi.org/10.1038/s41563-025-02355-y)

**Contact:** rokabe@mit.edu

---

## Announcement: 2026 Conference on Physics and AI (PAI26)

**June 10-12, 2026 | Stanford University**

[datascience.stanford.edu/.../2026-conference-physics-and-ai-pai26](https://datascience.stanford.edu/events/center-decoding-universe/2026-conference-physics-and-ai-pai26)

**Paper submission is open!** Four tracks:
- **Research** --- New AI methods for physics, physics-inspired AI
- **Datasets & Benchmarks** --- Public physics datasets with strong baselines
- **Perspectives** --- Viewpoints on the state and future of AI x physics
- **Education & Training** --- High-impact teaching materials, courses, or programs

**Key dates:**
| | |
|---|---|
| Submission deadline | **April 15 (AoE)** |
| Author notifications | April 29 |
| Conference | **June 10-12, 2026** |

---

## Thank you!

**Questions?**

Ryotaro Okabe | rokabe@mit.edu

- **Paper:** [doi.org/10.1038/s41563-025-02355-y](https://doi.org/10.1038/s41563-025-02355-y)
- **GitHub:** [github.com/RyotaroOKabe/APS_demo_SCIGEN](https://github.com/RyotaroOKabe/APS_demo_SCIGEN)
- **PAI26:** [datascience.stanford.edu](https://datascience.stanford.edu/events/center-decoding-universe/2026-conference-physics-and-ai-pai26)

---
