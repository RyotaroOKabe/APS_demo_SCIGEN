# Slide Presentation Script: 04 — SCIGEN Generation (Capstone)

**Duration:** ~5 minutes

---

## Slide 1: SCIGEN — the key innovation

*Read:*
"DiffCSP generates any crystal. SCIGEN generates crystals with specific geometric patterns."

"The approach: inpainting. Specify a target sublattice — kagome, honeycomb, triangular. Fix those atom positions. Let the model generate everything else."

"Like painting a picture with some pixels already decided."

---

## Slide 2: Available structural constraints

*Read:*
"13 lattice types, ranging from simple to complex."

"Kagome — 3 known atoms — corner-sharing triangles for frustrated magnetism. Honeycomb — 2 known atoms — like graphene, for topological materials. Triangular — just 1 known atom — the simplest 2D lattice."

"All the way up to great rhombotrihexagonal with 12 known atoms — a complex Archimedean tiling."

---

## Slide 3: The inpainting approach

*Read:*
"Step by step:"
1. "Define the constraint — say kagome Mn atoms at specific fractional positions"
2. "Start from noise — all other atoms are random"
3. "Denoise for 1000 steps — the model generates complementary atoms"
4. "At the final step, replace the constrained sites with their true positions"
5. "Result: a complete crystal with the kagome sublattice built in"

"The model learns during training to anticipate these constraints."

---

## Slide 4: Generation parameters

*Read:*
"The parameters you'll control: lattice type, element for constrained sites, batch size, step size."

"Try different combinations: kagome Mn for frustrated magnets, honeycomb Fe for topological materials, triangular Co for simple 2D systems."

---

## Slide 5: What we analyze

*Read:*
"For each generated structure, we check six things:"
- "Composition — what elements did the model choose?"
- "Lattice parameters — reasonable lengths and angles?"
- "Bond distances — nearest neighbors in 1.5-3.5 Angstrom range?"
- "Space group — what symmetry?"
- "SMACT screening — charge-balanced?"
- "XRD pattern — experimental fingerprint"

---

## Slide 6: The full pipeline

*Read:*
"In the published work: start with a constraint, generate 10 million candidates, pre-screen by composition and bond distances, MLIP screening with CHGNet, DFT validation, and finally 24,743 confirmed materials."

"Today we'll do a mini version of this pipeline in the notebook."

---

## Slide 7: What we'll do

*Read:*
"In the notebook: load the model, choose kagome Mn, generate structures, visualize the diffusion trajectory, inspect compositions and symmetries, compare against the full SCIGEN dataset, and export CIF files."

"This is the hands-on capstone — take your time, try different constraints. Let's go."
