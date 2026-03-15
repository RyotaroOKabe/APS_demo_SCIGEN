# Slide Presentation Script: 00 — Setup & Tutorial Overview

**Duration:** ~3 minutes (while participants run setup)

---

## Slide 1: Title

*Read:*
"Welcome to APS Tutorial T4: Generative AI for Physics — From Models to Materials."

"Today we'll learn how to generate new crystal structures using diffusion models — specifically SCIGEN, which was published in Nature Materials earlier this year."

---

## Slide 2: Tutorial Overview (schedule table)

*Read:*
"Here's our plan for today. Six notebooks, each building on the last."

"We start with setup and crystal structure basics. Then we'll train a diffusion model on MNIST to build intuition. We'll see how diffusion extends to crystals with DiffCSP. The capstone is SCIGEN — generating materials with structural constraints like kagome. Finally, we evaluate with machine-learning interatomic potentials."

"Total time is about 70-90 minutes, with hands-on Colab work throughout."

---

## Slide 3: The big picture

*Read:*
"Here's the big picture. Known materials databases are huge — the Materials Project has over 150,000 entries. But exotic lattice materials — kagome, honeycomb — are extremely rare."

"Generative models let us explore beyond the known. DiffCSP generates unconstrained crystals. SCIGEN adds structural constraints — you specify the geometry you want. Then we screen with MLIPs and validate with DFT."

"The published pipeline: 10 million candidates generated, screened to 24,743 DFT-validated new materials."

---

## Slide 4: Setup checklist

*Read:*
"Please open Notebook 00 in Colab now and start running the cells. You need a GPU runtime — T4 or better."

"While the installation runs — about 2 to 5 minutes — I'll introduce the next topic."

---

## Slide 5: Key links

*Read:*
"The paper DOI and GitHub repo are on this slide. Everything is open source."
