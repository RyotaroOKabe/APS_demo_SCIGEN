# Notebook 00: Setup

---

## Tutorial Overview

**APS Tutorial T4: Generative AI for Physics — From Models to Materials**

SCIGEN: Structural Constraint Integration in a Generative Model
for the Discovery of Quantum Materials

*Nature Materials (2025)*

---

## What we'll cover today

| # | Notebook | Time |
|---|----------|------|
| 00 | **Setup** — GPU, dependencies, model weights | 5 min |
| 01 | **Crystal Structures** — pymatgen, visualization, TB bands | 15–20 min |
| 02 | **Generative Concepts** — diffusion models for physicists | 10 min |
| 03 | **DiffCSP** — crystal diffusion: noise & denoising | 10 min |
| 04 | **SCIGEN** — constrained crystal generation | 20 min |
| 05 | **MLIP Evaluation** — CHGNet, phonons, stability | 10–15 min |

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

## Key links

- **Paper:** [doi.org/10.1038/s41563-025-02355-y](https://doi.org/10.1038/s41563-025-02355-y)
- **GitHub:** [github.com/RyotaroOKabe/APS_demo_SCIGEN](https://github.com/RyotaroOKabe/APS_demo_SCIGEN)
- **Colab:** Click the badges in the README

---
