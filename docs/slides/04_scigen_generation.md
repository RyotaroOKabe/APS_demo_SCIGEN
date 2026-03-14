# Notebook 04: SCIGEN — Constrained Crystal Generation

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
- `'hon'` + `'Fe'` — honeycomb iron compounds
- `'tri'` + `'Co'` — triangular cobalt
- `'van'` — unconstrained (DiffCSP mode)

---

## What we generate and analyze

For each generated structure:

1. **Composition** — what elements did the model choose?
2. **Lattice parameters** — are a, b, c, angles reasonable?
3. **Bond distances** — NN distances in physical range (1.5–3.5 A)?
4. **Space group** — what symmetry does the structure have?
5. **SMACT screening** — is the composition charge-balanced?
6. **XRD pattern** — fingerprint for experimental comparison

---

## The full SCIGEN pipeline (10M -> 24K)

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

## What we'll do in this notebook

1. Load pretrained SCIGEN model
2. Choose structural constraint (kagome)
3. **Generate crystal structures** (~30s for 4 structures)
4. **Visualize diffusion trajectory** (noise -> crystal, animated)
5. Inspect compositions, lattice params, bond distances
6. PyVista 3D rendering
7. Space group analysis, SMACT screening, XRD
8. Export as CIF files
9. **Try different constraints** (honeycomb, triangular, ...)

---
