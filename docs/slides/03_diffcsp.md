# Notebook 03: DiffCSP — Crystal Diffusion

---

## DiffCSP: diffusion for crystals

**Three things to diffuse simultaneously:**

| Component | Representation | Noise model |
|-----------|---------------|-------------|
| Atom positions | Frac coords F in [0,1)^{Nx3} | **Wrapped normal** |
| Lattice | 3x3 matrix L | Standard Gaussian |
| Atom types | One-hot A in R^{NxK} | Standard Gaussian |

Key insight: **fractional coordinates are periodic**
(x = 0.95 + 0.1 = 0.05, not 1.05)

---

## The wrapped normal distribution

Standard Gaussian doesn't respect periodicity.

**Wrapped normal:** sum over all periodic images

$$WN(x; mu, sigma^2) = sum_{k=-inf}^{inf} N(x + k; mu, sigma^2)$$

This ensures the noise "wraps around" the unit cell boundaries.

The **score** of the wrapped normal is used in the loss function,
not simple Gaussian noise prediction.

---

## Noise schedules

**Lattice:** Cosine schedule for alpha_bar_t
- alpha_bar goes from ~1 (clean) to ~0 (pure noise)
- Controls how much of the original lattice survives

**Coordinates:** Exponential sigma schedule
- sigma goes from 0.005 (tiny perturbation) to 0.5 (large noise)
- Separate from lattice schedule because coordinates are periodic

---

## Forward process visualization

```
t=0          t=200        t=500        t=800        t=999
[crystal]    [noisy]      [very noisy] [almost gone] [pure noise]
```

Watch:
- Atoms scatter from lattice sites
- Unit cell distorts and eventually collapses
- Atom types become random

---

## Reverse process: predictor-corrector

**Predictor** (reverse SDE step):
$$L_{t-1} = (1/sqrt(alpha)) * (L_t - noise_pred) + sigma * z$$

**Corrector** (Langevin dynamics for coordinates):
$$F_{t-1/2} = F_t + eta * score_pred + sqrt(2*eta) * z  (mod 1)$$

The corrector step refines the coordinate prediction using
the score function, with periodic wrapping.

---

## DiffCSP -> SCIGEN: what changes?

| | DiffCSP | SCIGEN |
|---|---------|--------|
| Generation | Unconditional | Constrained |
| Atom positions | All freely diffused | Known atoms inpainted |
| Lattice | Free 3x3 matrix | Constrained geometry |
| Use case | General generation | Targeted discovery |

SCIGEN = DiffCSP + structural constraints (inpainting)

---

## What we'll do in this notebook

1. Load the pretrained model
2. Visualize noise schedules (cosine beta, exponential sigma)
3. Apply noise to a real crystal at different timesteps
4. **Animated forward process** (crystal dissolving into noise)
5. Generate structures (vanilla mode, no constraints)
6. **Animated reverse process** (noise crystallizing into a structure)

---
