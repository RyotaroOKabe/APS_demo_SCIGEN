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
2. Load and explore the MP-20 dataset
3. Visualize structures (matplotlib + PyVista)
4. Build a kagome lattice
5. Compute tight-binding band structures
6. Space group analysis + simulated XRD

---
