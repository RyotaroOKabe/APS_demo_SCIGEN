# Colab Demo Script: Notebook 05 — MLIP Evaluation

**Estimated time: 15 min** | **Cells: 47** | **Long-running: ~12 cells**

---

## Cell 0 — Markdown: Title and learning objectives

"Welcome to Notebook 5: Evaluating Generated Materials with Machine-Learning Interatomic Potentials. This is the final notebook in the tutorial. Our learning objectives are: use CHGNet for fast property prediction, relax structures to their equilibrium positions, evaluate thermodynamic stability via the convex hull, screen SCIGEN materials across different constraint types, and understand the full Generate-Screen-Validate pipeline."

> :book: *Same text as notebook — read from screen*

"This is the final notebook. We've generated crystal structures with SCIGEN — now we need to know if they're real. Are they thermodynamically stable? Do they have interesting properties? DFT would take weeks for thousands of candidates. CHGNet does it in minutes."

---

## Cell 1 — Code: Setup (clone, install) :hourglass_flowing_sand: TAKES LONG (~1-3 min first run)

**[Run this cell.]**

If NB00 was already run: "This should be fast — just verifying packages."

### :speech_balloon: While waiting (fresh session only)

"CHGNet — Crystal Hamiltonian Graph Neural Network — is a universal machine-learning interatomic potential. It was trained on over 1.5 million DFT calculations from the Materials Project and can predict energies, forces, stresses, and magnetic moments for any inorganic material. One forward pass takes milliseconds on a GPU, versus hours for a DFT calculation. This speed difference is what makes MLIP-based screening practical for generative materials discovery."

---

## Cell 2 — Markdown [5.1]: 5.1 The evaluation problem

"Section 5.1: The evaluation problem. For every generated structure, we need to answer three questions. First, is it physically reasonable — do the bond lengths and angles make sense? Second, is it thermodynamically stable — is it low energy, near the convex hull? Third, does it have interesting properties — for example, magnetic moments? The traditional approach is density functional theory, or DFT. It's accurate, but it takes hours per structure. The modern alternative is machine-learning interatomic potentials, or MLIPs. They're trained on DFT data but run in seconds — roughly a thousand times faster."

> :book: *Same text as notebook — read from screen*

"Four questions we need to answer for every generated structure. Is it physically reasonable — sensible bond lengths? Is it thermodynamically stable — low energy, near the convex hull? Is it dynamically stable — no imaginary phonon frequencies? Does it have interesting properties — magnetic moments? We'll address all four."

---

## Cell 3 — Markdown [5.2]: 5.2 CHGNet introduction (accuracy table)

"Section 5.2: CHGNet. CHGNet is a universal graph neural network that predicts energy with about 30 meV per atom accuracy, forces at about 70 meV per angstrom, stress at about 0.3 GPa, and magnetic moments. It was trained on 1.5 million DFT calculations from the Materials Project. The reference paper is Deng et al., Nature Machine Intelligence, 2023."

> :book: *Same text as notebook — read from screen*

Emphasize: "The accuracy numbers: about 30 meV per atom for energy, 70 meV per angstrom for forces, 0.3 GPa for stress. These are good enough for screening — we're ranking candidates, not computing exact phase diagrams. The magnetic moments are semi-quantitative: they capture trends but not precise values."

---

## Cell 4 — Code [5.2]: Load CHGNet model :hourglass_flowing_sand: TAKES LONG (~30-60 sec first run)

**[Run this cell.]**

"This downloads and loads the CHGNet pretrained weights — about 400,000 parameters. First run downloads from the CHGNet package cache."

### :speech_balloon: While waiting

"CHGNet's architecture is a graph neural network, similar in spirit to SCIGEN's CSPNet but designed for property prediction rather than generation. It takes a crystal structure as input — atoms, bonds, lattice — and outputs energy per atom, force vectors on every atom, the 3×3 stress tensor, and magnetic moments. The 'charge-informed' part of the name means it explicitly models oxidation states, which is important for transition metal compounds where magnetism depends on the charge state."

---

## Cell 5 — Markdown [5.3]: 5.3 Predict properties for a test structure

"Section 5.3: Predict properties for a test structure. This is a sanity check. We'll take a known structure from the MP-20 test set, predict its properties with CHGNet, and compare against the DFT reference values."

> :book: *Same text as notebook — read from screen*

---

## Cell 6 — Code [5.3]: Load test structure from MP-20

**[Run this cell.]**

"Loads a crystal from the test set and prints its formula and DFT formation energy. This is our reference structure for the sanity check."

---

## Cell 7 — Code [5.3]: Predict with CHGNet

**[Run this cell.]** Fast (< 1 sec).

"One line: chgnet.predict_structure. You should see four outputs: energy in eV per atom, forces as a vector on each atom (should be small for an equilibrium structure), the stress tensor diagonal, and magnetic moments if the structure is magnetic. Compare the predicted energy to the DFT reference from the previous cell — they should be within about 30 meV per atom."

---

## Cell 8 — Markdown [5.4]: 5.4 Structure relaxation from a perturbed state

"Section 5.4: Structure relaxation. To demonstrate relaxation, we deliberately perturb a known structure by adding about 0.3 angstroms of random noise to atom positions. Then we relax it with CHGNet and watch the energy decrease as atoms settle back toward equilibrium."

> :book: *Same text as notebook — read from screen*

"We deliberately break a known stable structure by adding random noise to atom positions — about 0.3 angstroms of displacement. Then we let CHGNet relax it back. This simulates what happens with a generated structure that hasn't been relaxed yet."

---

## Cell 9 — Code [5.4]: Relax perturbed structure with CHGNet :hourglass_flowing_sand: TAKES LONG (~5-30 sec)

**[Run this cell.]**

"The StructOptimizer from CHGNet runs an energy minimization. It moves atoms downhill on the energy surface until forces converge to near zero."

### :speech_balloon: While waiting

"Relaxation is key for evaluating generated structures. SCIGEN generates structures that are approximately correct — the diffusion model learned the distribution of stable configurations — but they're not at exact energy minima. Relaxation with CHGNet takes each structure to the nearest local minimum. If the energy drops a lot during relaxation, the generated structure was far from equilibrium. If it barely changes, the diffusion model did a good job."

---

## Cell 10 — Code [5.4]: Compare perturbed, relaxed, and original structures

**[Run this cell.]**

"This prints the RMSD between the perturbed structure and the original, then between the relaxed structure and the original. The relaxed RMSD should be much smaller — CHGNet successfully found its way back close to the true equilibrium."

---

## Cell 11 — Markdown [5.4.1]: Relaxation trajectory

"Now let's look at the relaxation trajectory. We'll plot energy versus relaxation step. You should see a rapid drop in the first few steps, then convergence to a plateau. The force magnitude decreases monotonically as the structure approaches equilibrium."

> :book: *Same text as notebook — read from screen*

---

## Cell 12 — Code [5.4.1]: Plot relaxation energy trajectory :hourglass_flowing_sand: TAKES LONG (~5-30 sec)

**[Run this cell.]**

"An energy-versus-step plot. You should see a rapid drop in the first few steps, then convergence to a plateau. The force magnitude should decrease monotonically. If the energy oscillates, the step size may be too large — but for most structures, convergence is smooth."

---

## Cell 13 — Markdown [5.4.2]: Structure animation during relaxation

"Here's a structure animation of the relaxation. Colored spheres show the current atom positions at each step, red arrows show the forces acting on each atom, and green dots mark the equilibrium positions. Watch how the atoms migrate toward equilibrium as the relaxation proceeds."

> :book: *Same text as notebook — read from screen*

---

## Cell 14 — Code [5.4.2]: Relaxation animation :hourglass_flowing_sand: TAKES LONG (~5-30 sec)

**[Run this cell.]**

"An animation showing atoms moving from their perturbed positions back to equilibrium. If rendering works, you'll see the structure tighten up as relaxation proceeds — bond angles and distances snapping into proper values. If the animation doesn't render, the static plots above tell the same story."

---

## Cell 15 — Markdown [5.5]: 5.5 Batch evaluation on MP-20 test set

"Section 5.5: Batch evaluation on the MP-20 test set. We predict formation energy for multiple structures and create a parity plot comparing CHGNet predictions against DFT reference values. We'll compute the mean absolute error. Good agreement here means we can trust CHGNet for screening generated structures where no DFT reference exists."

> :book: *Same text as notebook — read from screen*

---

## Cell 16 — Code [5.5]: Batch CHGNet predictions :hourglass_flowing_sand: TAKES LONG (~30-90 sec)

**[Run this cell.]**

"Predicting formation energies for about 20 structures from the test set."

### :speech_balloon: While waiting

"Each prediction takes about 50 milliseconds. We're running sequentially here, but in production you'd batch them on GPU for even faster throughput. The Materials Project uses similar MLIP-based workflows to triage millions of hypothetical structures before committing to expensive DFT calculations.

The parity plot we're about to see shows predicted versus actual formation energy. Perfect predictions would lie on the diagonal line. Scatter around the line tells you the prediction error. For CHGNet, the scatter is typically about 30 meV per atom — small enough to rank candidates correctly even if the absolute values aren't perfect."

---

## Cell 17 — Markdown [5.5.1]: Speed comparison DFT vs CHGNet

"Let's put the speed difference in concrete terms. DFT takes 1 to 10 hours per structure. CHGNet takes less than 1 second. If you have 10,000 candidates — which is a typical SCIGEN output — DFT would require about 10,000 hours, roughly one year of CPU time. CHGNet does the same screening in about 3 hours on a GPU. That's roughly a 3,000-times speedup."

> :book: *Same text as notebook — read from screen*

---

## Cell 18 — Code [5.5.1]: Speed comparison bar chart

**[Run this cell.]** Fast.

"A bar chart showing the dramatic speed difference. DFT: hours per structure. CHGNet: under one second. For 10,000 SCIGEN candidates, that's about 10,000 hours of DFT versus about 3 hours of CHGNet. This 3,000× speedup is what makes the SCIGEN screening pipeline feasible."

---

## Cell 19 — Code [5.5.1]: DFT vs CHGNet accuracy scatter plot

**[Run this cell.]** Fast.

"The parity plot. Each dot is one test structure. The x-axis is DFT formation energy, the y-axis is CHGNet prediction. Points should cluster along the diagonal. The MAE printed below the plot tells you the average prediction error."

---

## Cell 20 — Markdown [5.6]: 5.6 Phonon calculation

"Section 5.6: Phonon calculation. Phonons are the vibrational modes of a crystal. If all phonon frequencies are real and positive, the structure is dynamically stable — atoms vibrate around their equilibrium positions. If any frequency is imaginary — shown as a negative value — the structure is sitting at a saddle point on the energy surface and will spontaneously distort. We use ASE's Phonons module with CHGNet as the calculator."

> :book: *Same text as notebook — read from screen*

"Phonons test dynamic stability. If a structure has imaginary phonon frequencies — shown as negative values — it's not at a true energy minimum. It's sitting on a saddle point and will spontaneously distort."

---

## Cell 21 — Code [5.6]: Phonon DOS with ASE + CHGNet :hourglass_flowing_sand: TAKES LONG (~2-5 min)

**[Run this cell.]**

"This is the longest single calculation: computing the phonon density of states using finite displacements in a supercell with CHGNet as the calculator."

### :speech_balloon: While waiting (~2-5 min)

"Let me explain what's happening. The phonon calculation works by slightly displacing each atom — about 0.01 angstroms — in each direction, computing the forces with CHGNet, and building the force constant matrix from the force differences. This is the finite displacement method.

We use a 2×2×2 supercell, which means 8 copies of the unit cell. For a structure with, say, 8 atoms per cell, that's 64 atoms in the supercell, times 3 directions, times 2 displacements per direction — that's 384 CHGNet evaluations. Each one is fast, but they add up.

The result is the phonon density of states: a histogram of vibrational frequencies. In a stable crystal, all frequencies are positive — atoms vibrate around their equilibrium positions. If any frequency is imaginary — which shows up as negative on the DOS plot — the structure is dynamically unstable. It will spontaneously distort to a lower-symmetry phase.

For SCIGEN-generated structures, phonon stability is an important filter. A structure can be thermodynamically stable — low energy, near the convex hull — but dynamically unstable if it's at a saddle point rather than a true minimum. The phonon check catches these cases.

In the published SCIGEN pipeline, phonon calculations were not performed on all 24,743 structures — that would be too expensive even with CHGNet. Instead, they were applied selectively to the most promising candidates after energy and force-based screening. For a full phonon calculation with DFT accuracy, you'd use VASP or Quantum ESPRESSO with the phonopy package, which takes several hours per structure."

---

## Cell 22 — Markdown [5.7]: 5.7 Thermodynamic stability — convex hull

"Section 5.7: Thermodynamic stability and the convex hull. The convex hull is the fundamental tool for assessing whether a material is stable against decomposition. E_hull equals zero means the structure is on the hull — it's the ground state for that composition. E_hull less than 25 meV per atom is the commonly used threshold for synthesizability. Between 25 and 100 meV per atom is metastable — might be stabilizable under certain conditions. Above 100 meV per atom, the structure will most likely decompose into competing phases. We use pymatgen's PhaseDiagram to compute these values."

> :book: *Same text as notebook — read from screen*

---

## Cell 23 — Code [5.7]: Build phase diagram, compute hull distances :hourglass_flowing_sand: TAKES LONG (~10-30 sec)

**[Run this cell.]**

"This builds a phase diagram from the test set entries and computes E_hull for each structure. You should see a plot of the convex hull with test structures colored by their distance from the hull."

---

## Cell 24 — Markdown [5.8]: 5.8 Screening SCIGEN-generated materials

"Section 5.8: Screening SCIGEN-generated materials. Now we bring it all together. We'll download the 24,743 DFT-validated structures from the published SCIGEN dataset, sample across constraint types, run CHGNet predictions, relax selected structures, and assess stability. This demonstrates the MLIP as a fast post-generation screening tool."

> :book: *Same text as notebook — read from screen*

"Now we apply everything we just learned to real SCIGEN-generated structures. We'll predict energies, forces, and magnetic moments, then relax structures and compute E_hull."

---

## Cell 25 — Code [5.8]: Download SCIGEN DFT structures :hourglass_flowing_sand: TAKES LONG (~1-2 min)

**[Run this cell.]**

"Downloading the SCIGEN DFT-validated structures from Figshare."

### :speech_balloon: While waiting

"These 24,743 structures survived the full four-stage screening pipeline. We're sampling about 5 per constraint type — roughly 55 structures total — to keep the computation fast for the tutorial. In practice, you'd evaluate all of them."

---

## Cell 26 — Code [5.8]: Sample 5 structures per constraint type

**[Run this cell.]** Fast.

"This samples a handful of structures per constraint type — kagome, honeycomb, triangular, square, and so on. You should see a summary of how many structures were selected per type."

---

## Cell 27 — Markdown [5.8.1]: Formation energy by constraint type

"Now let's look at formation energy by constraint type. We'll create boxplots. What to look for: formation energies should be in a reasonable range, roughly negative 2 to 0 eV per atom. Narrow distributions suggest consistent generation quality. Watch for outliers with positive formation energies, which indicate unstable compositions. Also look for differences between constraint types."

> :book: *Same text as notebook — read from screen*

---

## Cell 28 — Code [5.8.1]: Batch predict formation energies

**[Run this cell.]** Takes a few seconds.

"CHGNet predicts the formation energy for each sampled structure. No long wait — these are individual forward passes."

---

## Cell 29 — Code [5.8.1]: Formation energy boxplot

**[Run this cell.]** Fast.

"Boxplots showing formation energy by constraint type. Lower is more favorable. Look for: are all constraint types in a reasonable range of negative 1 to negative 3 eV per atom? Are there outliers with positive energies? Do some constraint types consistently produce more stable compositions?"

---

## Cell 30 — Markdown [5.8.2]: Magnetic moments by constraint type

"Next, magnetic moments by constraint type. This is where the physics motivation comes through. Kagome and triangular lattices are specifically targeted for frustrated magnetism. Structures with magnetic moments above 1 Bohr magneton contain magnetically active elements — Fe, Mn, Co, Ni, Cr — and are the most interesting candidates for exotic magnetic behavior."

> :book: *Same text as notebook — read from screen*

Emphasize: "Magnetic properties are a primary motivation for SCIGEN. Kagome and triangular lattices are targeted specifically for frustrated magnetism. Structures with moments above 1 Bohr magneton contain magnetically active elements and are the most interesting candidates."

---

## Cell 31 — Code [5.8.2]: Magnetic moment boxplot

**[Run this cell.]** Fast.

"Boxplots of maximum atomic magnetic moment per structure. Kagome structures with Mn, Fe, or Co should show significant moments — typically 1 to 4 Bohr magnetons. Constraint types that favor non-magnetic elements will show moments near zero."

---

## Cell 32 — Markdown [5.8.3]: Comparison with MP-20 training data

"Let's compare SCIGEN-generated structures against the MP-20 training data. We want to see good overlap in the property distributions. If the generated structures have formation energies and other properties in the same range as the training data, that confirms the model is producing physically realistic materials. Systematic shifts would indicate a bias in the generation or screening pipeline."

> :book: *Same text as notebook — read from screen*

---

## Cell 33 — Code [5.8.3]: CHGNet predictions on training data :hourglass_flowing_sand: TAKES LONG (~10-60 sec)

**[Run this cell.]**

"Predicting energies for a random sample of training structures for comparison."

### :speech_balloon: While waiting

"The purpose of this comparison is to check whether SCIGEN-generated structures have properties in the same range as the training data. A well-trained generative model should produce structures that are broadly similar to what it learned from. Systematic shifts — for example, if all generated structures have much higher energies than training data — would indicate a problem with the generation or screening pipeline."

---

## Cell 34 — Markdown [5.8.4]: Relaxation analysis

"Section on relaxation analysis. RMSD — root-mean-square displacement — measures how far each generated structure moved during relaxation. It tells us how close the diffusion model got to the true equilibrium geometry. Small RMSD, less than about 0.1 angstroms, means the model already found near-equilibrium structures. Large RMSD suggests significant adjustment was needed. We measure both the energy drop and the position RMSD."

> :book: *Same text as notebook — read from screen*

---

## Cell 35 — Code [5.8.4]: Relax SCIGEN structures :hourglass_flowing_sand: TAKES LONG (~1-5 min)

**[Run this cell.]**

"Relaxing about 20 SCIGEN structures with CHGNet's StructOptimizer."

### :speech_balloon: While waiting (~1-5 min)

"Each relaxation runs until forces converge below a threshold or a maximum number of steps is reached. For well-generated structures, convergence is fast — maybe 10 to 50 steps. For structures that are far from equilibrium, it may take hundreds of steps.

The energy drop during relaxation is a quality metric for the generative model. If SCIGEN generates structures that are already close to stable configurations — small energy drops, small RMSDs — that means the diffusion model has learned the distribution of stable structures well. In the original SCIGEN paper, most generated structures had relaxation displacements under 0.3 angstroms, which is quite good.

The structures that require large relaxation displacements are often the ones with unusual compositions — the model was asked to fill in atoms around a kagome sublattice and chose an unusual combination that doesn't quite work. These are typically filtered out by the energy-based screening."

---

## Cell 36 — Code [5.8.4]: Re-relax and visualize one structure :hourglass_flowing_sand: TAKES LONG (~1-3 min)

**[Run this cell.]**

"This picks the structure with the largest relaxation displacement and re-relaxes it with trajectory tracking. You should see a before-and-after 3D comparison showing how atoms moved during relaxation."

---

## Cell 37 — Code [5.8.4]: Relaxation summary plots (energy drop, RMSD)

**[Run this cell.]** Fast.

"Two plots. Left: energy drop during relaxation — negative values mean the structure got more stable. Right: RMSD — the average atomic displacement during relaxation. Most structures should have RMSD under 0.5 angstroms."

---

## Cell 38 — Markdown [5.8.5]: Screening summary

"Screening summary. We now apply threshold filters to identify the most promising candidates. The filters are: maximum force below 0.1 eV per angstrom, negative formation energy, and RMSD below 0.5 angstroms. Structures that pass all three filters are promoted to the more expensive E_hull evaluation."

> :book: *Same text as notebook — read from screen*

---

## Cell 39 — Code [5.8.5]: Apply screening filters

**[Run this cell.]** Fast.

"Prints how many structures passed each filter and the final count after all filters. This simulates the pre-screening step that the SCIGEN pipeline applies before the more expensive E_hull calculation."

---

## Cell 40 — Markdown [5.8.6]: Energy above hull for generated structures

"Now we compute E_hull — energy above the convex hull — for the structures that passed screening. We build local phase diagrams using reference data for each chemical system. Remember the interpretation: E_hull equals zero means the structure sits on the hull, the most stable configuration. Less than 25 meV per atom is the threshold most researchers use for likely synthesizability. Less than 100 meV per atom is metastable. Above 100 meV per atom, the structure is thermodynamically unstable and will likely decompose."

> :book: *Same text as notebook — read from screen*

---

## Cell 41 — Code [5.8.6]: Compute E_hull :hourglass_flowing_sand: TAKES LONG (~10-30 sec)

**[Run this cell.]**

"Building local phase diagrams from the test set for each structure's chemical system, then computing E_hull."

### :speech_balloon: While waiting

"The convex hull is built per chemical system — for example, all Mn-O compounds form one hull, all Mn-S compounds form another. A structure's E_hull is its energy distance above this hull. We use the MP-20 test set as the reference for hull construction, which is an approximation — in principle you'd want all known phases. But for a tutorial-level screening, this gives a good ranking."

---

## Cell 42 — Markdown [5.8.7]: Top candidates

"Here are our top candidates, ranked by E_hull. For each structure, the table shows its composition, constraint type, formation energy, E_hull, magnetic moment, and relaxation RMSD. The best candidates are the ones that combine a low E_hull with significant magnetic moments and a small RMSD — that means they're thermodynamically competitive, magnetically interesting, and were already near equilibrium when generated."

> :book: *Same text as notebook — read from screen*

---

## Cell 43 — Code [5.8.7]: Display ranked candidates table

**[Run this cell.]** Fast.

"The final table. Candidates ranked by E_hull, lowest first. For each, you see: composition, constraint type, formation energy, E_hull, maximum magnetic moment, and relaxation RMSD. The best candidates combine low E_hull with significant magnetic moments and small relaxation displacement. These would be the ones you'd send to DFT validation in a real discovery campaign."

---

## Cell 44 — Markdown [5.9]: 5.9 Full discovery pipeline summary

"Section 5.9: The full discovery pipeline. Let me walk you through the throughput at each stage. Generation with SCIGEN produces about 1,000 structures per hour on a GPU. Pre-screening with composition and geometry filters is essentially instant. MLIP screening with CHGNet also handles about 1,000 structures per hour. DFT validation with VASP is the bottleneck at about 10 structures per hour. In the published work, the pipeline went from 10 million generated structures to 100,000 after pre-screening, to 25,000 sent to DFT, to the final 24,743 validated structures. The MLIP screening step — what we just demonstrated — reduced the DFT workload by roughly 400 times."

> :book: *Same text as notebook — read from screen*

---

## Cell 45 — Markdown [5.9]: Key takeaways

"Let me highlight the key takeaways from this notebook. First, generation without evaluation is useless — you need the screening pipeline. Second, MLIPs like CHGNet are roughly 1,000 times faster than DFT, which makes large-scale screening possible. Third, structure relaxation verifies that generated structures are near equilibrium. Fourth, phonon calculations check dynamic stability — no imaginary frequencies means the structure is at a true energy minimum. Fifth, E_hull tells you about thermodynamic stability — how likely the material is to actually exist. Sixth, magnetic moments connect directly to the target physics of frustrated magnetism. Seventh, SCIGEN-generated structures fall in realistic property ranges compared to the training data. And eighth, the full pipeline took 10 million generated structures down to 24,743 DFT-validated materials."

> :book: *Same text as notebook — read from screen*

Emphasize these five points:
1. "CHGNet: universal potential, seconds per structure, about 30 meV per atom accuracy."
2. "Four screening levels: physical reasonableness, thermodynamic stability, dynamic stability, target properties."
3. "MLIPs reduce DFT workload by 3,000×."
4. "Full pipeline: 10 million generated, 100K pre-screened, 24,743 DFT-confirmed."
5. "Top candidates combine low E_hull, magnetic moments, and small relaxation displacement."

---

## Cell 46 — Markdown [5.9]: What's next / closing

"And that brings us to the end of the tutorial. Let me recap what we covered across all five notebooks. In Notebook 01, we learned how crystal structures are represented mathematically as lattice, coordinates, and atom types. In Notebook 02, we built intuition for generative AI by training a diffusion model on MNIST digits. In Notebook 03, we applied diffusion to crystal structures with DiffCSP. In Notebook 04, we used SCIGEN to generate materials with specific geometric constraints like the kagome lattice. And here in Notebook 05, we evaluated those generated materials using CHGNet as a fast surrogate for DFT."

> :book: *Same text as notebook — read from screen*

"That completes the full SCIGEN tutorial. We started by learning how crystals are represented as (L, X, A) in Notebook 01. We trained a diffusion model on MNIST in Notebook 02. We applied diffusion to crystals in Notebook 03. We generated kagome materials with SCIGEN in Notebook 04. And now we've evaluated them with CHGNet in Notebook 05.

The code and notebooks are all open source. I encourage you to go back to Notebook 04 and try different constraint types — honeycomb Fe, triangular Co, Lieb Ni — and then bring the results here for evaluation.

Thank you for participating. The paper is in Nature Materials 2025, the code is on GitHub, and the dataset of 24,743 DFT-validated materials is on Figshare. If you have questions, feel free to reach out."
