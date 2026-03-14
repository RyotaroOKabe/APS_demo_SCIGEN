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

## What we'll do in this notebook

1. Load CHGNet pretrained model
2. Predict energy/forces for a test structure
3. **Relax a perturbed crystal** (animated trajectory with forces)
4. Compare CHGNet vs DFT (parity plot, MAE)
5. Compute **phonon DOS** (dynamic stability check)
6. Estimate **E_hull** (thermodynamic stability)
7. Connect to SCIGEN's full screening pipeline

---
