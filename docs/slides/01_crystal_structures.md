# Notebook 01: Crystal Structures & Materials Data

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
| **Lattice** | **L** | Periodic box: lengths (a, b, c) + angles (α, β, γ) |
| **Fractional coordinates** | **X** | Atom positions in [0, 1) |
| **Atom types** | **A** | Chemical species at each site |

Crystal = (**L**, **X**, **A**) — this is exactly what SCIGEN denoises.

---

## Conventional vs. primitive unit cell

The **same crystal** can be described by different unit cells:

- **Conventional:** standard cell showing full symmetry (NaCl → 8 atoms, cubic)
- **Primitive:** smallest cell that tiles space (NaCl → 2 atoms, rhombohedral)

Both produce the same infinite crystal when repeated.

---

## Graph representation of crystals

Neural networks see crystals as **graphs**:

| Graph element | Crystal meaning |
|---|---|
| **Node** | Atom (features: element, position) |
| **Edge** | Neighbor bond within cutoff radius |

Key: edges include **periodic boundary** connections → graphs naturally encode translational symmetry.

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

## Fractional vs Cartesian coordinates

| | Fractional | Cartesian |
|---|---|---|
| Range | [0, 1) | Angstroms |
| Periodicity | Natural (x and x+1 are same) | Must handle wrapping |
| Cell-independent | Yes | No |

Fractional coordinates are what diffusion models operate on.

---

## pymatgen: the Swiss army knife

```python
from pymatgen.core import Structure, Lattice

lattice = Lattice.cubic(5.64)
nacl = Structure(lattice, ['Na', 'Cl', ...], frac_coords)
```

- Parse CIF files
- Symmetry analysis (space groups)
- Neighbor searches, distances
- Interface to Materials Project

---

## The MP-20 dataset

- **45,231** structures from the Materials Project
- All with **<= 20 atoms** per unit cell
- SCIGEN's training data

We'll explore:
- Property distributions (formation energy, band gap, stability)
- Element frequency (periodic table heatmap)
- Space group distribution

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

The flat band is a **geometric property** — it's perfectly flat
regardless of hopping strength.

---

## What we'll do in this notebook

1. Build NaCl from scratch with pymatgen
2. Understand the (L, X, A) representation
3. Compare conventional vs. primitive unit cells
4. Build supercells, crystal graphs, and explore invariances
5. Load and explore the MP-20 dataset
6. Build a kagome lattice + tight-binding band structures
7. Space group analysis + simulated XRD

---
