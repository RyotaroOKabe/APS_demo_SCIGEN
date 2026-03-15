# Slide Presentation Script: 05 — MLIP Evaluation

**Duration:** ~4 minutes

---

## Slide 1: Why evaluate?

*Read:*
"Generation without evaluation is useless. For each structure, four questions:"
- "Physically reasonable — sensible bond lengths?"
- "Thermodynamically stable — low energy, near the convex hull?"
- "Dynamically stable — no imaginary phonon frequencies?"
- "Interesting properties — magnetic, electronic?"

---

## Slide 2: DFT vs. MLIP

*Read:*
"DFT is the gold standard — high accuracy, but hours per structure. CHGNet is a machine-learning interatomic potential — about 30 meV/atom accuracy, but seconds per structure."

"For 10,000 candidates: DFT takes weeks. CHGNet takes minutes. That's a 400x speedup — essential for screening generative model outputs."

---

## Slide 3: CHGNet

*Read:*
"CHGNet: Crystal Hamiltonian Graph Neural Network. Trained on over 1 million DFT calculations from the Materials Project."

"Predicts energy, forces, stress tensor, and magnetic moments. Works for any inorganic material. 400,000 parameters, runs on GPU."

"One line: `chgnet.predict_structure(struct)` — and you get everything."

---

## Slide 4: Structure relaxation

*Read:*
"Relaxation finds the nearest local energy minimum. We'll demo this by perturbing a known crystal — adding noise to atom positions — then relaxing with CHGNet."

"Watch the energy decrease and atoms settle. If the energy drop is small, the generated structure was already close to equilibrium — a good sign."

---

## Slide 5: Phonon calculation

*Read:*
"Phonons — lattice vibrations — test dynamic stability. No imaginary frequencies means the structure is at a true energy minimum."

"We use ASE with CHGNet as the calculator: finite displacement method on a supercell. The phonon DOS shows the frequency distribution."

---

## Slide 6: Energy above hull

*Read:*
"The convex hull is the thermodynamic ground state. E_hull = 0 means the structure is the most stable composition at that chemistry."

"Below 25 meV/atom: likely synthesizable. Above 100 meV/atom: will probably decompose. This is how we rank candidates for experimental synthesis."

---

## Slide 7: Screening SCIGEN materials

*Read:*
"We apply CHGNet to SCIGEN's 24,743 DFT-validated structures. Formation energy and magnetic moments, broken down by constraint type."

"Relax structures and measure displacement — small displacement means the generative model already found near-equilibrium geometries."

"Compute E_hull for all structures. Rank by stability and magnetic properties to find the best candidates."

---

## Slide 8: The full pipeline

*Read:*
"The complete discovery pipeline: generate with SCIGEN at 1000/hour, pre-screen instantly, MLIP screen with CHGNet at 1000/hour, DFT validate at 10/hour."

"10 million generated, 100K pre-screened, 24,743 DFT-confirmed. MLIPs made this possible."

"Let's do it hands-on in the notebook."
