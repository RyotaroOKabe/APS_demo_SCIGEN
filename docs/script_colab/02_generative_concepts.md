# Colab Demo Script: Notebook 02 — Generative AI Concepts

**Estimated time: 15 min** | **Cells: 27** (14 markdown, 13 code) | **Long-running: 2 cells**

---

## Cell 0 — Markdown: Title & learning objectives

"This is Notebook 2: Generative AI Concepts — From DDPM to Materials. In this notebook we'll build intuition for what generative models do, how DDPM — denoising diffusion probabilistic models — works, using physics analogies you'll recognize. We'll do a hands-on exercise training a DDPM on MNIST, explore conditional generation, and see why diffusion is so well-suited for materials discovery. This notebook trains a small diffusion model on MNIST, which takes about 5 minutes on GPU."

📖 *Same text as notebook — read from screen*

---

## Cell 1 — Code: Setup (install scipy, check GPU)

**[Run this cell.]**

"Quick setup — installs scipy and checks the GPU. You should see 'Device: cuda' and 'Ready.'"

---

## Cell 2 — Markdown [2.1]: 2.1 Discriminative vs. generative models

"Section 2.1 — discriminative versus generative models. There's a comparison table on screen. On the left, discriminative models: the task is to predict a label from data. They learn P of label given data. A materials example would be 'Is this crystal stable?' On the right, generative models: the task is to create new data. They learn P of data itself. A materials example is 'Generate me a new stable crystal structure.'"

"The physics analogy here: think of training data as sampling an energy landscape. Generative sampling is then finding low-energy configurations on that landscape."

📖 *Same text as notebook — read from screen*

Emphasize: "The key distinction: discriminative models answer questions about data — 'Is this stable?' Generative models create new data — 'Make me a new stable crystal.' We want the second one."

---

## Cell 3 — Markdown [2.2]: 2.2 Taxonomy of generative models

"Section 2.2 — a taxonomy of generative models. There are four major families. VAEs — variational autoencoders — encode data into a latent space and decode back. GANs — generative adversarial networks — pit a generator against a discriminator. Normalizing flows use invertible transforms. And diffusion models add noise then learn to remove it — think of it as heating a crystal until it melts, then annealing it back."

"Diffusion models have become the dominant approach for crystal generation because they naturally handle the periodicity and symmetry of crystal structures. That's what we'll use."

📖 *Same text as notebook — read from screen*

---

## Cell 4 — Markdown [2.3]: 2.3 How diffusion models work (DDPM equations)

"Section 2.3 — how diffusion models actually work. This cell has the math, but I'll highlight the key ideas."

"There are three things to remember. First, the forward process: you add Gaussian noise step by step via q of x_t given x_{t-1}. The key insight is you can jump to any noise level t in one step using alpha-bar-t — the cumulative signal retention factor."

"Second, the training loss: predict the noise epsilon, minimize mean squared error. That's the whole training objective. One line."

"Third, the reverse: subtract the predicted noise, add a small stochastic kick, repeat from t=T to t=0."

"The physics analogies: forward is melting a crystal. Reverse is controlled crystallization from a melt. The neural network has learned the 'crystallization dynamics.'"

"There's also a score function perspective on screen: the network learns the gradient of log-probability — 'which direction should I move each atom to make this look more like a real crystal?'"

📖 *Same text as notebook — read from screen*

---

## Cell 5 — Markdown [2.4]: 2.4 Toy forward process

"Section 2.4 — let's see this concretely with a toy example. We have a ring of 6 clusters, like atoms sitting at lattice sites. We'll add increasing amounts of noise and watch the structure dissolve."

📖 *Same text as notebook — read from screen*

---

## Cell 6 — Code [2.4]: Toy forward process (ring of Gaussians)

**[Run this cell.]**

"Here's a 2D example. Six clusters of points — think of these as atoms at lattice sites."

*Point to each panel:* "At sigma=0, perfect crystal order. At sigma=0.3, some thermal motion. At sigma=0.8, losing structure. At sigma=2.0, pure noise — you can't tell where the clusters were."

---

## Cell 7 — Markdown [2.4.1]: Toy reverse process

"Now the reverse process. We estimate the score — the gradient of log probability — using k-nearest neighbors, then run Langevin dynamics: take a step in the score direction and add a little noise. Starting from pure noise, we should see the data distribution crystallize back."

📖 *Same text as notebook — read from screen*

---

## Cell 8 — Code [2.4.1]: Toy reverse process

**[Run this cell.]**

"Now the reverse. Starting from pure noise on the left."

*Point to panels:* "First step: starting to cluster. Second step: clear structure. Third step: crystallized — back to 6 clusters, matching the original distribution."

"This toy denoiser uses k-nearest neighbors to estimate the score function. A real DDPM uses a neural network instead."

---

## Cell 9 — Markdown [2.4.2]: Animated view

"Let's see an animated version of this forward-reverse cycle."

---

## Cell 10 — Code [2.4.2]: Animation

**[Run this cell.]**

"Watch the forward → reverse cycle. Blue frames dissolve the structure. Orange frames rebuild it. This is the full diffusion process in miniature."

---

## Cell 11 — Markdown [2.4.3]: The noise schedule

"Now let's look at the noise schedule — the specific values of beta-t that control how much noise we add at each step. In standard DDPM, beta increases linearly from about 1e-4 to about 0.02. Alpha-bar-t is the cumulative product of (1 minus beta), representing how much signal is retained at step t. It drops from 1 — meaning clean data — to nearly 0 — meaning pure noise — by step T."

📖 *Same text as notebook — read from screen*

---

## Cell 12 — Code [2.4.3]: Noise schedule visualization

**[Run this cell.]**

"Two plots. Left: beta-t increases linearly — more noise added per step as t grows. Right: alpha-bar-t — the cumulative signal. It starts at 1 (clean) and drops to nearly 0 (pure noise) by step 1000."

"At t=0, the structure is clean. At t=1000, alpha-bar is essentially zero — pure noise."

---

## Cell 13 — Markdown [2.4.3]: Summary connecting to crystals

"Let's summarize the three core ideas before moving on. First, the forward process destroys structure by adding noise. Second, the reverse process creates structure by removing noise. Third, a neural network learns this reverse process. For crystals, the same principle applies in 3D with periodic boundary conditions — we diffuse atom positions, atom types, and lattice parameters simultaneously."

📖 *Same text as notebook — read from screen*

---

## Cell 14 — Markdown [2.5]: 2.5 Hands-on DDPM on MNIST

"Section 2.5 — hands-on DDPM on MNIST. We're going to build and train a real diffusion model. The architecture is a UNet with residual convolutional blocks. It uses a sinusoidal time embedding so the model knows where it is in the diffusion process. There's a class embedding — one-hot encoding of the digit 0 through 9 — so we can do conditional generation. And we use classifier-free guidance: during training, we randomly drop the class label with probability 0.1, so at inference time the model can generate both conditionally and unconditionally."

📖 *Same text as notebook — read from screen*

"The time embedding tells the model where it is in the diffusion process. The class embedding tells it which digit to generate. Classifier-free guidance lets us control how strongly we condition on the class."

---

## Cell 15 — Code [2.5]: Model definition (~300 lines)

**[Run this cell.]**

"This defines all the model classes — ResidualConvBlock, UNet encoder/decoder, time and class embeddings, the DDPM wrapper with noise scheduling. About 1.7 million parameters."

"You should see 'Model classes defined.' and the parameter count."

---

## Cell 16 — Markdown [2.5.2]: Quick training demo

"Now we'll do a quick training demo — just 2 epochs. The results will be blurry, and that's expected. Two epochs isn't enough for sharp digits, but it proves the pipeline works. Afterward we'll load a pretrained checkpoint for quality results."

📖 *Same text as notebook — read from screen*

---

## Cell 17 — Code [2.5.2]: Train DDPM on MNIST ⏳ TAKES LONG (~5 min)

**[Run this cell.]**

"Now we train for real. Two epochs on MNIST — about 5 minutes on GPU."

### 💬 While waiting — fill time with the MNIST-to-crystals connection:

"While we wait, let me explain why we're training on handwritten digits when our goal is crystal structures."

"The DDPM algorithm is the same in both cases. The key mapping is:"

"Pixel values in [0, 1] become fractional coordinates in [0, 1). The 28×28 image grid becomes a variable-sized set of atoms. The UNet becomes a graph neural network called CSPNet. The digit class label (0 through 9) becomes a structural constraint — kagome, honeycomb, triangular. And classifier-free guidance becomes inpainting with fixed atom positions."

"If you understand what's happening on MNIST, you understand the core of SCIGEN. The extensions for crystals are: coordinates are periodic, so we need a wrapped normal distribution instead of standard Gaussian noise. We also have to diffuse the lattice matrix and atom types, not just positions. And graph neural networks replace convolutions because crystals are graphs, not grids."

"We'll see all of these extensions in Notebook 03."

*Check progress — watch the tqdm bar.* "You should see the loss decreasing. After epoch 1 it's around 0.04. After epoch 2 it drops a bit more."

"The blurry samples at the end are expected — 2 epochs isn't enough for sharp digits. That's why we load a pretrained checkpoint next."

---

## Cell 18 — Markdown [2.5.3]: Load pretrained and generate

"Now we load a pretrained model trained for 20 epochs and generate digits at three different guidance strengths. w=0 means unconditional — no class information, just 'generate any digit.' w=0.5 is mild guidance — some class influence. w=2.0 is strong guidance — the model is heavily steered toward the target digit class. This lets us see the diversity-fidelity tradeoff."

📖 *Same text as notebook — read from screen*

---

## Cell 19 — Code [2.5.3]: Load pretrained + generate ⏳ TAKES LONG (~1-2 min)

**[Run this cell.]**

"Loading the pretrained checkpoint — 20 epochs of training. Then generating 40 digits at three guidance strengths."

### 💬 While waiting:

"Classifier-free guidance works by interpolating between the conditional prediction — 'generate a 7' — and the unconditional prediction — 'generate any digit.' The parameter w controls the interpolation. At w=0, unconditional: diverse but ambiguous. At w=2, strongly conditional: sharp and class-specific."

"For SCIGEN, this same idea becomes inpainting: the constraint (kagome positions) plays the role of the class label, and the model interpolates between constrained and unconstrained generation."

*Check for output plots.* "Here are the results. Three rows — one per guidance strength."

"w=0: diverse, some digits are hard to read. w=0.5: clearer, good variety. w=2.0: sharp, every digit is clearly recognizable. This is the diversity-fidelity tradeoff."

---

## Cell 20 — Markdown [2.5.4]: Denoising trajectory

"Now let's visualize the denoising trajectory — what happens at each time step during generation. Each row in the plot corresponds to a snapshot in time. The top rows are early in the reverse process, so they're still mostly noise. The middle rows show coarse shapes emerging. The bottom rows show the final sharp digits. This coarse-to-fine progression is exactly what happens in crystal generation too."

📖 *Same text as notebook — read from screen*

---

## Cell 21 — Code [2.5.4]: Denoising trajectory visualization

**[Run this cell.]**

"This is my favorite visualization in the tutorial. Each column is a digit class (0 through 9). Each row is a snapshot in time."

"Top rows: pure noise. You can't see anything. Middle rows: coarse shapes emerge — the model has 'decided' which digit to draw. Bottom rows: fine details sharpen."

"This is exactly what happens in crystal generation. Global structure first — the lattice forms, the rough atom arrangement appears. Then local refinement — precise bond lengths, correct chemistry."

---

## Cell 22 — Markdown [2.5.5]: MNIST-to-crystals connection table

"Here's the key comparison table — I think of this as the Rosetta Stone of the tutorial. On the MNIST side: pixel intensities, a 28-by-28 grid, a UNet architecture, class labels for conditioning, and classifier-free guidance. On the crystal side: fractional coordinates, a variable number of atoms, a graph neural network, structural constraints like kagome or honeycomb, and inpainting to fix certain atom positions. The key differences for crystals: coordinates are periodic so we use a wrapped normal distribution, we diffuse lattice parameters and atom types in addition to positions, and GNNs replace CNNs because crystals are graphs not grids."

📖 *Same text as notebook — read from screen*

"This table is the Rosetta Stone of the tutorial. If you remember one thing, it's this mapping."

---

## Cell 23 — Markdown [2.6]: 2.6 Crystal generative models landscape

"Section 2.6 — the landscape of crystal generative models. Let me walk through the timeline. CDVAE in 2022 was a VAE-based approach — the first serious crystal generative model. DiffCSP in 2023 brought diffusion to crystal structure prediction. SCIGEN in 2025 — that's what we're using today — added structural constraints like kagome and honeycomb lattices. MatterGen, also 2025, focuses on property-guided generation. And UniMat, again 2025, aims to be a universal materials generative model. The field is moving fast."

📖 *Same text as notebook — read from screen*

---

## Cell 24 — Markdown [2.6]: Key takeaways

"Five things to remember from this notebook. One: diffusion models reverse a noising process — forward destroys, reverse creates. Two: the training loss is just mean squared error on noise prediction. Three: classifier-free guidance enables conditional generation by interpolating between conditional and unconditional predictions. Four: the exact same framework extends to crystals with periodic coordinates, lattice diffusion, and graph neural networks. Five: we just demonstrated all of this hands-on with MNIST."

📖 *Same text as notebook — read from screen*

---

## Cell 25 — Markdown [2.6]: References

"The references are in the notebook — DDPM, classifier-free guidance, score matching, and the MNIST code we adapted."

---

## Cell 26 — Markdown [2.6]: What's next?

"Now we apply this exact framework to crystal structures. Notebook 03: Diffusion for Materials — where we handle periodic coordinates, lattice matrices, and atom types."

"Please open Notebook 03."
