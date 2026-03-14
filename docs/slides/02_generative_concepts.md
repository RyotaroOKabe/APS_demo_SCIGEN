# Notebook 02: Generative AI Concepts — From DDPM to Materials

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
Clean data --> add noise --> ... --> pure noise

**Reverse process (generation):**
Pure noise --> remove noise --> ... --> new data

*Physics analogy: melting a crystal, then controlled re-crystallization*

---

## DDPM equations

**Forward** (add noise at step t):
$$q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)$$

**Training loss** (predict the noise):
$$L = E[ || epsilon - epsilon_theta(x_t, t) ||^2 ]$$

**Reverse** (denoise):
$$x_{t-1} = (1/sqrt(alpha_t)) * (x_t - noise_pred * beta_t/sqrt(1-alpha_bar_t)) + sigma_t * z$$

---

## Classifier-free guidance

Train the model to work both WITH and WITHOUT class labels:
- Randomly drop the label during training (with probability p=0.1)
- At generation time, interpolate between conditional and unconditional predictions:

$$eps = (1 + w) * eps_conditional - w * eps_unconditional$$

- w = 0: unconditional
- w = 2: strong class guidance

---

## Hands-on: DDPM on MNIST

We train a real diffusion model on handwritten digits:
- **UNet** with time + class embedding (~1.7M parameters)
- **10 epochs** on MNIST (~5 min on T4 GPU)
- **Conditional generation**: specify which digit to generate
- **Guidance strength**: control the class-specificity

Demo:
1. Train the model (watch loss decrease)
2. Generate all 10 digit classes
3. Visualize the denoising trajectory (noise -> digit)

---

## From images to crystals

| MNIST (images) | Materials (crystals) |
|----------------|---------------------|
| Pixel values [0, 1] | Fractional coordinates [0, 1) |
| 28x28 grid | Variable number of atoms |
| UNet predicts noise | GNN (CSPNet) predicts noise |
| Class label (0-9) | Structural constraint (kagome, ...) |
| Classifier-free guidance | Inpainting with fixed positions |

Same DDPM core — different representations and network architectures.

---

## Crystal generative models

| Model | Year | Key feature |
|-------|------|-------------|
| CDVAE | 2022 | First crystal generative model (VAE) |
| DiffCSP | 2023 | Diffusion for crystal structure prediction |
| SCIGEN | 2025 | Structural constraints via inpainting |
| MatterGen | 2025 | Property-guided generation |

All build on the DDPM framework we just trained on MNIST.

---

## What we'll do in this notebook

1. Review DDPM theory (forward/reverse process, equations)
2. 2D toy example (ring of Gaussians)
3. **Train a DDPM on MNIST** (~5 min)
4. **Generate digits** with different guidance strengths
5. **Visualize denoising trajectory** (noise -> digit)
6. Connect to materials generation

---
