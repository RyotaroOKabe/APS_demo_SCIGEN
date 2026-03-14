# Notebook Enhancement Options

Ideas to improve the APS tutorial notebooks — organized by category, with effort estimates.

---

## 1. Embed Existing Assets (Quick Wins)

The repo already has illustrations in `assets/` that aren't used in the notebooks.

| Enhancement | Notebook | Effort |
|---|---|---|
| Embed `SI_arch_lattice_unit_bk.png` (lattice type illustrations) in the constraint table | 03, 05 | 5 min |
| Embed `figure1.png` (SCIGEN pipeline overview from the paper) | 05 | 5 min |
| Embed `scigen_logo.png` in notebook headers | 00, 05 | 5 min |

---

## 2. Better Crystal Structure Visualization

**Current state:** Static matplotlib 3D scatter plots with colored dots. No bonds, no unit cell context, no interactivity.

| Enhancement | Notebook | Effort | Notes |
|---|---|---|---|
| **Supercell views** — show 2x2x1 or 3x3x1 to reveal periodicity and constraint patterns | 01, 05 | Low | `structure.make_supercell([2,2,1])` before plotting |
| **Draw bonds** — connect nearest-neighbor atoms in 3D plots | 01, 05 | Low | Use `structure.get_neighbors()` with cutoff |
| **Unit cell box** — draw lattice vectors on generated structures | 05 | Low | Already done in 01's `plot_structure()` but not in 05's `plot_structure_3d()` |
| **Interactive 3D viewer** — replace matplotlib with `py3Dmol` or `nglview` | 01, 05 | Medium | `py3Dmol` is lightweight (~30 KB JS), works in Colab. `nglview` is heavier but more powerful |
| **Side-by-side comparison** — show generated structure next to a known MP structure with same elements | 05 | Medium | Helps users judge quality |
| **Highlight constrained atoms** — color kagome sublattice atoms differently from generated atoms | 05 | Low | Use `num_known` to split atoms into two groups |

---

## 3. Data Visualization Upgrades (Notebook 02)

**Current state:** Basic histograms and bar charts.

| Enhancement | Effort | Notes |
|---|---|---|
| **Periodic table heatmap** — color each element by frequency in training data | Medium | More intuitive than a bar chart; use a simple grid layout or `pymatviz` |
| **Composition space scatter** — plot structures in a ternary diagram (e.g., metal-oxygen-other) | Medium | Shows where training data is concentrated |
| **Spacegroup distribution** — bar chart of most common space groups | Low | `train_df['spacegroup.number'].value_counts().head(20)` |
| **Atoms-per-cell histogram** — show the distribution of unit cell sizes | Low | Motivates SCIGEN's 1–20 atom range |

---

## 4. Diffusion Process Visualization (Notebook 03)

**Current state:** Static 2D toy examples (ring of Gaussians). Effective but could be more engaging.

| Enhancement | Effort | Notes |
|---|---|---|
| **Animated GIF/video** of the forward+reverse process | Medium | Use `matplotlib.animation.FuncAnimation`, save as HTML5 video for inline display |
| **Score function visualization** — overlay vector field arrows showing "which direction atoms should move" | Medium | Makes the score function concept tangible |
| **3D toy example** — extend the 2D ring to a 3D lattice (closer to real crystals) | Medium | |
| **Real structure trajectory** — show actual SCIGEN denoising snapshots from a saved trajectory | High | Requires running generation with `save_traj_idx`, then loading and plotting frames |
| **Noise schedule plot** — show how noise level changes over diffusion steps | Low | Simple line plot, helps explain the schedule |

---

## 5. MLIP Evaluation Enhancements (Notebook 04)

**Current state:** Single-structure prediction, relaxation, batch scatter plot.

| Enhancement | Effort | Notes |
|---|---|---|
| **Relaxation trajectory** — plot energy vs. optimization step | Low | `result['trajectory'].energies` is already available |
| **Before/after overlay** — superimpose original and relaxed structures to show atomic displacement | Medium | Two-color scatter in same 3D plot |
| **Timing comparison bar chart** — "DFT: ~1 hour, CHGNet: 0.05 seconds" | Low | Visual impact for the speedup argument |
| **Stability classification** — label structures as stable/unstable based on energy above hull | Medium | More practical than raw energy numbers |

---

## 6. Generation Output Analysis (Notebook 05)

**Current state:** Table printout + 3D scatter plots. No quality analysis.

| Enhancement | Effort | Notes |
|---|---|---|
| **Constraint verification** — check that kagome atoms are at the expected fractional positions | Medium | Compare generated positions to SC constraint definition |
| **Bond length distribution** — histogram of nearest-neighbor distances in generated structures | Low | Quick sanity check; physically unreasonable structures have very short bonds |
| **Lattice parameter distributions** — violin or box plots across the generated batch | Low | Show spread of a, b, c, alpha, beta, gamma |
| **Composition pie chart** — element distribution across generated structures | Low | |
| **Compare to training data** — overlay generated lattice parameters on training data distribution | Medium | Shows if generation is in-distribution |
| **Try multiple constraints** — generate kag, hon, tri side-by-side and compare | Medium | Compelling visual showing constraint diversity |

---

## 7. Interactivity (ipywidgets)

**Current state:** No interactive elements in any notebook.

| Enhancement | Notebook | Effort | Notes |
|---|---|---|---|
| **Structure browser** — dropdown to select and visualize any structure from training data | 01 | Medium | `ipywidgets.interact` with index slider |
| **Lattice type selector** — dropdown that updates the constraint description and shows the lattice illustration | 05 | Medium | Better than editing code cells |
| **Supercell slider** — adjust nx, ny, nz and see the supercell update | 01 | Medium | Shows periodicity concept |
| **Noise level slider** — scrub through diffusion noise levels interactively | 03 | Medium | More engaging than static 4-panel figure |
| **Element picker** — periodic table click to select element for generation | 05 | High | Requires custom widget or `ipywidgets` grid |

---

## 8. Exercises & Engagement

**Current state:** Only notebooks 01 and 04 have exercises.

| Notebook | Suggested Exercise |
|---|---|
| 02 | "How many structures have formation energy < -2 eV/atom? What elements dominate?" |
| 02 | "Find the structure with the largest band gap. What is it?" |
| 03 | "Modify the toy code: use 4 Gaussians in a square instead of 6 in a ring. Run the denoising." |
| 05 | "Generate with `hon` (honeycomb) instead of `kag`. Compare the compositions." |
| 05 | "Change ELEMENT to `Fe`. How do the lattice parameters differ from Mn?" |

---

## 9. Narrative & Explanation

| Enhancement | Notebook | Effort |
|---|---|---|
| **"Why kagome?" sidebar** — 2-3 sentences on frustrated magnetism, flat bands, spin liquids | 01 | Low |
| **Fractional vs. Cartesian coordinates** — brief explanation with diagram | 01 | Low |
| **Periodic boundary conditions** — show what happens at the unit cell edges | 01 | Low |
| **"What makes a good generated structure?"** — checklist (reasonable bonds, valid composition, low energy) | 05 | Low |
| **Pipeline flowchart** — visual diagram: Generate → Screen → Relax → Validate | 05 | Low |
| **Failure modes** — "What does a bad generated structure look like?" with examples | 05 | Medium |

---

## 10. Robustness & Accessibility

| Enhancement | Notebook | Effort |
|---|---|---|
| **Pre-computed fallback outputs** — pickle or JSON with example generation results, load if GPU fails | 04, 05 | Medium |
| **Colorblind-friendly palettes** — replace `Set1` with `tab10` or element-specific CPK colors | All | Low |
| **Larger font sizes** in plot labels and titles | All | Low |
| **Error recovery messages** — "If you see X, try Y" after risky cells | 00, 04, 05 | Low |
| **Progress bars** — `tqdm` for batch evaluation and generation loops | 04, 05 | Low |

---

## Priority Ranking

### Do First (low effort, high impact)
1. Embed `SI_arch_lattice_unit_bk.png` in notebooks 03 and 05
2. Highlight constrained vs. generated atoms in notebook 05
3. Add supercell views (2x2x1) in notebooks 01 and 05
4. Add exercises to notebooks 02, 03, 05
5. Colorblind-friendly palettes across all plots

### Do Next (medium effort, high impact)
6. Bond length distribution histogram in notebook 05
7. Relaxation trajectory plot in notebook 04
8. Animated diffusion process in notebook 03
9. Interactive structure browser (ipywidgets) in notebook 01
10. Periodic table heatmap in notebook 02

### Nice to Have (higher effort)
11. `py3Dmol` interactive 3D viewer in notebooks 01, 05
12. Real SCIGEN denoising trajectory visualization in notebook 03/05
13. Constraint verification (check kagome geometry) in notebook 05
14. Pre-computed fallback outputs for offline use
15. Side-by-side generation with multiple constraint types
