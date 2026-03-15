# Slide Presentation Script: 02 — Generative AI Concepts

**Duration:** ~5 minutes

---

## Slide 1: Discriminative vs. Generative

*Read:*
"Quick distinction. Discriminative models predict labels — 'is this stable?'. Generative models create new data — 'make me a new stable crystal'."

"A generative model learns the probability distribution over training data, then samples from it to create new examples."

---

## Slide 2: Diffusion models — the key idea

*Read:*
"The core idea of diffusion is beautiful in its simplicity."

"Forward: take clean data, gradually add noise until it's pure Gaussian noise. Like heating a crystal until it melts."

"Reverse: start from noise, gradually remove it to create new data. Like controlled crystallization from a melt."

"The neural network learns the reverse process — how to undo the noise at each step."

---

## Slide 3: DDPM equations

*Read:*
"The math is elegant. Forward: sample noisy data at any timestep t directly, using alpha-bar-t — the cumulative signal retention."

"Training loss: predict the noise epsilon, minimize MSE. One line."

"Reverse: subtract the predicted noise, add a small stochastic kick. Repeat from t=T down to t=0."

---

## Slide 4: Classifier-free guidance

*Read:*
"For conditional generation — generating a specific digit or lattice type — we use classifier-free guidance."

"During training, randomly drop the condition 10% of the time. At generation, interpolate: the guided prediction minus the unguided prediction, scaled by w."

"w=0 is unconditional. w=2 is strongly conditional. This same idea applies to SCIGEN's structural constraints."

---

## Slide 5: Hands-on DDPM on MNIST

*Read:*
"We'll train a real DDPM — a UNet with 1.7 million parameters. Two epochs as a quick demo, then load a pretrained checkpoint for quality generation."

"The key parallel: MNIST pixel values in [0,1] are just like fractional coordinates in [0,1). The UNet is replaced by a GNN. The digit class is replaced by a lattice constraint."

---

## Slide 6: From images to crystals

*Read:*
"This table is the Rosetta Stone of the tutorial. Same algorithm, different representation."

"Pixels become fractional coordinates. Grid becomes graph. UNet becomes CSPNet. Class label becomes structural constraint. Classifier-free guidance becomes inpainting."

"If you understand DDPM on MNIST, you understand the core of SCIGEN."

---

## Slide 7: Crystal generative models

*Read:*
"For context: CDVAE in 2022 was the first crystal generative model. DiffCSP in 2023 applied diffusion. SCIGEN in 2025 added structural constraints. MatterGen adds property guidance."

"All use the same DDPM foundation we're about to train."

---

## Slide 8: What we'll do

*Read:*
"In the notebook: toy 2D demo, train DDPM on MNIST, generate digits with guidance, visualize the denoising trajectory. Let's go."
