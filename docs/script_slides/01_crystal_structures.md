# Slide Presentation Script: 01 — Crystal Structures

**Duration:** ~5 minutes

---

## Slide 1: What is a crystal?

*Read:*
"A crystal is a lattice plus a basis. The lattice defines the periodic box — three vectors a, b, c with lengths and angles. The basis defines what's inside — atom species and their fractional coordinates between 0 and 1."

"The entire infinite crystal is just the unit cell copied in all directions."

---

## Slide 2: The three components (L, X, A)

*Read:*
"In a computer, every crystal is fully described by three things:"
- "L — the lattice: a 3x3 matrix (or equivalently, 6 parameters: a, b, c, alpha, beta, gamma)"
- "X — fractional coordinates: each atom's position in [0, 1)"
- "A — atom types: which element sits at each site"

"This (L, X, A) tuple is exactly what SCIGEN denoises during generation."

---

## Slide 3: Conventional vs. primitive

*Read:*
"Quick but important distinction. The same crystal has multiple valid unit cells. The conventional cell shows the full symmetry — NaCl has 8 atoms in a cube. The primitive cell is the smallest repeating unit — NaCl has just 2 atoms in a rhombohedron."

"Both produce the same infinite crystal. For machine learning, we typically use the primitive cell — fewer atoms means faster computation."

---

## Slide 4: Graph representation

*Read:*
"Neural networks process crystals as graphs. Atoms become nodes with element and position features. Bonds become edges — including connections across periodic boundaries."

"This graph representation naturally encodes all the symmetries of crystals: permutation, periodic translation, and rotation invariance."

---

## Slide 5: Fractional vs. Cartesian

*Read:*
"Why fractional coordinates? They're naturally periodic — x and x+1 are the same point. They're cell-independent. And they live in [0, 1), which is perfect for diffusion models."

"This periodicity is why we need the wrapped normal distribution for noise — standard Gaussian noise doesn't respect the periodic boundaries."

---

## Slide 6: pymatgen

*Read:*
"pymatgen is the Swiss army knife for computational materials science. We'll use it throughout: building structures, symmetry analysis, Materials Project queries, and more."

---

## Slide 7: MP-20 dataset

*Read:*
"SCIGEN trains on MP-20: 45,000 structures from the Materials Project, all with 20 or fewer atoms. We'll explore the distributions in the notebook."

---

## Slide 8: Why kagome?

*Read:*
"The kagome lattice — corner-sharing triangles — is special. It creates geometric frustration in magnetic materials, flat electronic bands, and potential quantum spin liquids."

"The problem: kagome materials are extremely rare in nature. Only a handful are known experimentally. This is the motivation for SCIGEN — generate what nature hasn't provided."

---

## Slide 9: Tight-binding bands

*Read:*
"This comparison is powerful. Kagome has a perfectly flat band — strong correlations. Honeycomb has Dirac cones — like graphene. Triangular has ordinary cosine bands."

"The flat band is purely geometric — it doesn't depend on chemistry. This is why targeting specific lattice geometries is so valuable."

---

## Slide 10: What we'll do

*Read:*
"In the notebook, you'll build NaCl from scratch, explore the MP-20 dataset, construct a kagome lattice, and compute tight-binding band structures. Let's go to Colab."
