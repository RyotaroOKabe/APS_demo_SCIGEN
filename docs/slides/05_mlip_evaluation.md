# Notebook 05: MLIP Evaluation

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

- Trained on Materials Project (> 1M DFT calculations)
- Predicts: energy, forces, stress, magnetic moments
- Works for *any* inorganic material
- 400K parameters, runs on GPU

```python
from chgnet.model.model import CHGNet
chgnet = CHGNet.load()
prediction = chgnet.predict_structure(struct)
```

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

Apply CHGNet to **24,743 SCIGEN-generated structures**:
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

MLIP screening reduced DFT workload by ~400x.

---

## What we'll do in this notebook

1. Load CHGNet pretrained model
2. Predict energy/forces for a test structure
3. **Relax a perturbed crystal** (animated trajectory)
4. Compare CHGNet vs DFT (parity plot, MAE)
5. **Phonon DOS** (dynamic stability check)
6. **E_hull** (thermodynamic stability)
7. **Screen SCIGEN materials** (energy, magnets, relaxation, E_hull)
8. **Top candidates** ranked by stability + magnetic properties

---
