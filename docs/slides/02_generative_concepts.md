# Notebook 02: Generative AI Concepts

---

## Discriminative vs Generative

| | Discriminative | Generative |
|---|---|---|
| Task | Predict label from data | Create new data |
| Learns | P(label \| data) | P(data) |
| Example | "Is this stable?" | "Make a new crystal" |

---

## Diffusion models: the key idea

**Forward process (heating):**
Clean crystal --> add noise --> ... --> pure noise

**Reverse process (crystallization):**
Pure noise --> remove noise --> ... --> new crystal

*Physics analogy: melting a crystal, then controlled re-crystallization*

---

## DDPM equations

**Forward** (add noise at step t):

$$q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)$$

**Reverse** (neural network learns to denoise):

$$p(x_{t-1} | x_t) = N(mu_theta(x_t, t), sigma_t^2 * I)$$

**Training loss** (predict the noise):

$$L = E[ || epsilon - epsilon_theta(x_t, t) ||^2 ]$$

---

## The score function

The network learns: "which direction should I move each atom
to make this look more like a real crystal?"

$$s_theta(x_t, t) = nabla_x log p(x_t)$$

This is the gradient of the log-probability — it points
toward high-density regions (valid crystal structures).

---

## Conditional generation

| Type | Example | Method |
|------|---------|--------|
| Property-guided | "Band gap > 2 eV" | Classifier guidance |
| Composition | "Must contain Mn, O" | Fix atom types |
| **Structure** | **"Kagome sublattice"** | **Inpainting** |

SCIGEN uses **inpainting**: fix the constrained atoms,
let the model generate everything else.

---

## What we'll do in this notebook

1. Visualize forward process (2D toy example)
2. Visualize reverse process (denoising)
3. See the noise schedule (beta_t, alpha_bar_t)
4. Understand the inpainting approach
5. Survey of crystal generative models (CDVAE, DiffCSP, SCIGEN, MatterGen)

---
