# Slide Presentation Script: 03 — DiffCSP / Crystal Diffusion

**Duration:** ~4 minutes

---

## Slide 1: DiffCSP — diffusion for crystals

*Read:*
"DiffCSP extends DDPM to crystals. Three things diffuse simultaneously:"

"Atom positions use the wrapped normal — respecting periodicity. The lattice matrix uses standard Gaussian noise. Atom types also use Gaussian noise on one-hot vectors."

"The wrapped normal is the key innovation: noise wraps around the unit cell boundaries, so x=0.95 + noise=0.1 gives 0.05, not 1.05."

---

## Slide 2: Wrapped normal distribution

*Read:*
"Standard Gaussian doesn't know about periodicity. The wrapped normal sums over all periodic images — an infinite sum that converges quickly."

"The score of the wrapped normal — the gradient of log-probability — is what the network actually predicts for coordinates. This is different from the simple noise prediction used for the lattice."

---

## Slide 3: Noise schedules

*Read:*
"Two separate schedules. Lattice: cosine schedule for alpha-bar, goes from 1 to near 0. Coordinates: exponential sigma, from 0.005 to 0.5."

"They're separate because coordinates are periodic and need different noise levels than the lattice matrix."

---

## Slide 4: Forward process visualization

*Read:*
"We'll watch a real crystal dissolve. At t=0: ordered structure. At t=500: losing coherence. At t=999: pure noise — atoms random, cell collapsed."

"This is exactly what the model learns to reverse."

---

## Slide 5: Reverse process — predictor-corrector

*Read:*
"The reverse uses predictor-corrector sampling. The predictor does a reverse SDE step on the lattice. The corrector runs Langevin dynamics on the coordinates — with periodic wrapping."

"This two-step approach gives better results than a single reverse step, especially for the periodic coordinates."

---

## Slide 6: DiffCSP vs. SCIGEN

*Read:*
"What changes when we go from DiffCSP to SCIGEN? Generation goes from unconditional to constrained. Some atom positions are fixed — inpainted. The lattice geometry is constrained."

"Everything else — the noise model, the network, the sampling — is inherited from DiffCSP."

---

## Slide 7: What we'll do

*Read:*
"In the notebook: load the model, visualize noise schedules, watch the forward process dissolve a crystal, generate structures with the reverse process, and compare against training data. Let's go."
