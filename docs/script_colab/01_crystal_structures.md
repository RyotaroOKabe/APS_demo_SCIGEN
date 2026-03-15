# Colab Demo Script: Notebook 01 — Crystal Structures

**Estimated time: 15-20 min** | **Cells: 68** | **Long-running: 1 cell (setup)**

---

## Cell 0 — Markdown: Title and learning objectives

"Welcome to Notebook 1: Crystal Structures and Materials Data. This notebook covers a lot of ground. By the end, you will be able to represent crystal structures in Python using pymatgen, create structures from scratch, understand the L-X-A representation that SCIGEN uses, compare conventional versus primitive unit cells, build supercells, represent crystals as graphs for neural networks, load structures from CIF files, visualize them in 3D, and appreciate why generative models are needed for exotic lattice geometries like kagome."

> 📖 *Same text as notebook — read from screen*

---

## Cell 1 — Code: Environment setup (clone, install, download)

**[Run this cell.]** ⚠️ TAKES LONG (~5-10 min first run)

"This cell clones the SCIGEN repository, installs all dependencies, and downloads the pretrained model weights from Figshare. If you already ran Notebook 00, this will detect the existing installation and skip most steps. Either way, go ahead and run it now."

### While waiting

"While that installs, let me set the stage for what we are about to do. This tutorial walks you through the representation of crystal structures — the language that SCIGEN speaks. A crystal is defined by a periodic box, atom positions inside that box, and the chemical species at each position. This is the L, X, A triple: lattice, fractional coordinates, and atom types. SCIGEN learns to generate all three simultaneously from random noise using a diffusion process."

"Why do we care about specific lattice geometries? The kagome lattice — a pattern of corner-sharing triangles — gives rise to perfectly flat electronic bands. Flat bands mean electrons are localized, which leads to exotic physics: frustrated magnetism, quantum spin liquids, and strongly correlated electron behavior. These materials are extraordinarily rare in nature. When SCIGEN was trained on the Materials Project database of about 45,000 structures and then asked to generate 10 million candidates with a kagome constraint, it identified roughly 25,000 unique, stable-looking kagome materials. That is the power of constrained generative modeling."

"The setup should be finishing up. Let us move on once it completes."

---

## Cell 2 — Markdown [1.1]: Section 1.1 "What is a crystal structure?"

"So what is a crystal structure? A crystal is defined by two things: a lattice and a basis. The lattice is the periodic box — described by three vectors a, b, c and three angles alpha, beta, gamma. The basis tells you what atoms sit inside that box, given by their chemical species and their fractional coordinates. In pymatgen, both of these are combined into a single Structure object."

> 📖 *Same text as notebook — read from screen*

---

## Cell 3 — Markdown [1.1.1]: "Fractional vs. Cartesian coordinates"

"Now an important distinction: fractional versus Cartesian coordinates. Cartesian coordinates give absolute positions in angstroms. Fractional coordinates give positions relative to the lattice vectors, ranging from 0 to 1. Fractional coordinates are much more natural for crystals because they are independent of cell size. An atom at fractional coordinate 0.5, 0.5, 0.5 is always at the center of the cell, no matter how big or small that cell is. This distinction matters a lot for generative models — fractional coordinates are bounded between 0 and 1, which makes them natural for diffusion on a torus."

> 📖 *Same text as notebook — read from screen*

---

## Cell 4 — Code [1.1.1]: Import pymatgen and numpy

**[Run this cell.]**

"This just imports pymatgen and numpy. Should be instant."

---

## Cell 5 — Markdown [1.2]: Section 1.2 "Creating a structure from scratch"

"Now we are going to build NaCl — table salt — from scratch. We need three ingredients: a cubic lattice with a equal to 5.64 angstroms, the species Na and Cl, and the fractional coordinates for each atom. Notice the last line of the notebook text: this is the same L, X, A representation that SCIGEN will later generate from noise."

> 📖 *Same text as notebook — read from screen*

---

## Cell 6 — Code [1.2]: Define cubic NaCl lattice and create Structure

**[Run this cell.]**

"Here we define the cubic lattice, list the 8 atoms — 4 Na and 4 Cl — with their fractional coordinates, and create the Structure object. The output shows the full NaCl structure: 8 sites in a cubic cell with a equal to 5.64 angstroms."

---

## Cell 7 — Markdown [1.2.1]: "Inspecting a structure" attribute table

"Pymatgen gives you direct access to all crystallographic information through simple Python attributes. The table here lists the key ones: dot-lattice gives you the lattice matrix, dot-species gives you the list of elements, dot-frac_coords and dot-cart_coords give you fractional and Cartesian coordinates respectively, dot-volume gives the cell volume, dot-composition gives the chemical formula, and dot-num_sites gives the atom count. These are the building blocks we will use throughout the tutorial."

> 📖 *Same text as notebook — read from screen*

---

## Cell 8 — Code [1.2.1]: Print structure properties

**[Run this cell.]**

"You should see the formula NaCl, 8 atoms, a volume of about 179 cubic angstroms, a density around 2.16 grams per cubic centimeter — which matches the experimental value for rock salt. Then the lattice parameters and all 8 fractional coordinate positions."

---

## Cell 9 — Markdown [1.3]: Section 1.3 The (L, X, A) representation

"This is the central representation for the entire tutorial. Every crystal is described by exactly three pieces: L for the lattice matrix — that is the lengths and angles of the box. X for the fractional coordinates — each in the range 0 to 1. And A for the atom types — which element sits at each position. Crystal equals L, X, A. This is exactly the tuple that SCIGEN's diffusion model generates — it learns a joint distribution over all three components simultaneously."

> 📖 *Same text as notebook — read from screen*

---

## Cell 10 — Code [1.3]: Print the (L, X, A) decomposition of NaCl

**[Run this cell.]**

"This prints out the three components explicitly. You can see the lattice matrix — for a cubic cell, it is diagonal with 5.64 on each axis. Then the fractional coordinates, each between 0 and 1. And finally the atom types: Na or Cl at each site. This is what SCIGEN starts from noise and denoises into."

---

## Cell 11 — Code [1.3]: Visual decomposition (three-panel matplotlib plot)

**[Run this cell.]**

"This gives you a three-panel 3D plot. The left panel shows just the lattice box — the periodic container. The middle adds the atomic positions as dots. The right panel colors and sizes the atoms by element — Na and Cl. This visual makes the L, X, A decomposition concrete."

---

## Cell 12 — Markdown [1.3]: Why fractional coordinates matter for generative models

"Why do fractional coordinates matter specifically for generative models? Three key points. First, they are bounded in the interval 0 to 1, which makes diffusion on a torus well-defined. Second, they are independent of absolute cell size — the same fractions describe the same relative arrangement no matter the lattice parameters. Third, they respect periodicity: coordinate 0.0 and coordinate 1.0 are the same point. These properties are why SCIGEN uses fractional coordinates rather than Cartesian."

> 📖 *Same text as notebook — read from screen*

---

## Cell 13 — Markdown [1.4]: Section 1.4 Conventional vs. primitive unit cell

"The same crystal can be described by different unit cells. The conventional cell shows the full symmetry — for NaCl, that is the familiar cubic cell with 8 atoms. The primitive cell is the smallest cell that still tiles to fill all of space — for NaCl, that is just 2 atoms in a rhombohedral cell. SCIGEN uses conventional cells, matching the MP-20 dataset convention."

> 📖 *Same text as notebook — read from screen*

---

## Cell 14 — Code [1.4]: Symmetry analysis — conventional vs. primitive

**[Run this cell.]**

"SpacegroupAnalyzer finds the symmetry of our NaCl structure, then extracts both the conventional and primitive cells. You should see: conventional has 8 atoms in a cubic cell, primitive has 2 atoms in a rhombohedral cell. The volumes differ by a factor of 4, because the conventional cell contains 4 copies of the primitive cell."

---

## Cell 15 — Code [1.4]: Side-by-side visualization of conventional vs. primitive

**[Run this cell.]**

"This gives you a side-by-side 3D plot comparing the two unit cells. On the left, the conventional cubic cell with 8 atoms — the familiar rock salt structure. On the right, the primitive rhombohedral cell with just 2 atoms. Both describe the same infinite crystal, just packaged differently."

---

## Cell 16 — Markdown [1.5]: Section 1.5 Periodicity, graph representation, and invariances

"We now move to three essential concepts for understanding how neural networks work with crystals. We will cover supercells, graph representations, and the invariances that any generative model must respect."

> 📖 *Same text as notebook — read from screen*

---

## Cell 17 — Markdown [1.5.1]: 1.5.1 Supercells and translational symmetry

"A supercell just replicates the unit cell multiple times along the lattice directions. A 2x2x2 supercell of NaCl is still NaCl — same physics, same crystal, just described with more atoms. Translational symmetry means shifting everything by a lattice vector changes nothing about the crystal."

> 📖 *Same text as notebook — read from screen*

---

## Cell 18 — Code [1.5.1]: Build and visualize supercells

**[Run this cell.]**

"You should see that the 1x1x1 cell has 8 atoms and the 2x2x2 has 64 atoms — exactly 8 times more, as expected for 2 cubed. The volume scales by a factor of 8. The side-by-side plot shows the unit cell on the left and the supercell on the right. Same structure, just tiled."

---

## Cell 19 — Markdown [1.5.2]: 1.5.2 Graph representation of crystals

"This is how neural networks see crystals. Atoms become nodes with features like element type and coordinates. Bonds become edges with distance and direction information. The critical difference from molecular graphs is that crystal graphs must handle periodicity. An atom near a cell boundary bonds to atoms in neighboring periodic images. SCIGEN uses something called radius_graph_pbc — which connects all atom pairs within a cutoff distance, including across periodic boundaries."

> 📖 *Same text as notebook — read from screen*

---

## Cell 20 — Code [1.5.2]: Build a crystal graph for NaCl

**[Run this cell.]**

"With a cutoff of 3.5 angstroms — just beyond the nearest-neighbor distance of about 2.82 angstroms — we get the neighbor list for each atom. You should see 8 nodes and a set of directed edges. Each Na has 6 nearest-neighbor Cl atoms, and vice versa — the classic octahedral coordination of rock salt."

---

## Cell 21 — Code [1.5.2]: Visualize the crystal graph with matplotlib

**[Run this cell.]**

"This plot shows two views. On the left, the 3D structure projected to 2D with bonds drawn explicitly — you can see bonds crossing the periodic boundary. On the right, a schematic graph representation showing nodes and edges. This is what the GNN inside SCIGEN actually operates on."

---

## Cell 22 — Markdown [1.5.3]: 1.5.3 Why graphs handle translational symmetry

"Here is an important insight about translational symmetry in graphs. If we shift all atoms by the same fractional vector and wrap around via modulo 1, the resulting graph is identical: same nodes, same edges, same distances. This is because edges are defined by inter-atomic distances, which do not change under uniform translation."

> 📖 *Same text as notebook — read from screen*

---

## Cell 23 — Code [1.5.3]: Demonstrate translational invariance

**[Run this cell.]**

"We shift all atoms by a random vector — 0.3, 0.15, 0.42 — wrap the coordinates modulo 1, and rebuild the neighbor list. The output should show that the sorted distance lists are identical, confirming that the graph is invariant under translation. The maximum difference should be essentially zero, within numerical precision."

---

## Cell 24 — Markdown [1.5.4]: 1.5.4 Summary table of four invariances

"This table summarizes the four invariances a crystal generative model must respect. First, permutation of atoms — the GNN message passing is order-independent. Second, periodic translation — handled by fractional coordinates on a torus. Third, unit cell choice — handled by training on a consistent convention. And fourth, rotation or reflection — handled by SCIGEN's SE(3)-equivariant architecture. Any valid crystal model must get all four of these right."

> 📖 *Same text as notebook — read from screen*

---

## Cell 25 — Markdown [1.6]: Section 1.6 Loading real structures from MP-20

"We move from hand-built structures to real data. The Materials Project has over 150,000 computed structures. SCIGEN is trained on a subset called MP-20: about 45,000 structures with 20 or fewer atoms per unit cell. The 20-atom cutoff keeps the diffusion model tractable. The dataset is stored as CSV files with CIF strings — the standard crystallographic format — embedded in each row."

> 📖 *Same text as notebook — read from screen*

---

## Cell 26 — Code [1.6]: Load train.csv with pandas

**[Run this cell.]**

"This loads the MP-20 training CSV. You should see around 36,000 to 45,000 structures and the column names: material_id, pretty_formula, the CIF string, formation energy, band gap, energy above hull, space group number, and elements."

---

## Cell 27 — Code [1.6]: Display first entries

**[Run this cell.]**

"A quick look at the first 10 rows. Notice the variety of compositions — binary oxides, ternary compounds, different space groups. The formation energy and energy above hull tell you about thermodynamic stability."

---

## Cell 28 — Code [1.6]: Parse the first CIF string into a Structure

**[Run this cell.]**

"Here we take the CIF string from the first row and parse it back into a pymatgen Structure object. You should see the formula, atom count, and DFT formation energy. This is how we go from a database row to a working crystal structure we can analyze and visualize."

---

## Cell 29 — Code [1.6]: Element distribution histograms

**[Run this cell.]**

"Three histograms appear. The formation energy distribution is centered around negative 1 to negative 2 eV per atom — these are thermodynamically stable materials. The band gap distribution shows a large peak at zero — metals — plus a spread of semiconductors and insulators. The energy above hull distribution is tightly peaked near zero, meaning most training structures are thermodynamically stable or nearly so."

---

## Cell 30 — Code [1.6]: Periodic table heatmap

**[Run this cell.]**

"This periodic table heatmap shows which elements appear most frequently in the training data. You will see that transition metals — especially 3d metals like Fe, Co, Ni, Mn — dominate, along with oxygen, sulfur, and nitrogen on the non-metal side. Rare earths and noble gases are underrepresented. This distribution matters because SCIGEN can only generate what it has seen — elements absent from the training data will not appear in generated structures."

---

## Cell 31 — Markdown [1.6.1]: "How many atoms per unit cell?"

"How many atoms per unit cell are in the training data? Most structures in MP-20 have 4 to 12 atoms per cell — small enough for fast DFT calculations. The 20-atom cutoff captures the majority of binary and ternary compounds. Generating larger cells with 50 to 200 atoms would require architectural changes to the GNN."

> 📖 *Same text as notebook — read from screen*

---

## Cell 32 — Code [1.6.1]: Atom count histogram

**[Run this cell.]**

"This histogram shows the distribution of atoms per unit cell. You should see a peak around 4 to 8 atoms, tapering off toward 20. This confirms that most training structures are relatively small, which is why SCIGEN can handle them efficiently."

---

## Cell 33 — Markdown [1.6.2]: "How rare are kagome materials?"

"Now the motivating question: how rare are kagome materials? Very few exist in current databases. Kagome magnets are rare but extremely interesting for physics — frustrated magnetism, quantum spin liquids, flat electronic bands, anomalous Hall effects. SCIGEN's goal is to generate new candidate materials with these special lattice geometries, expanding the search space far beyond what has been experimentally observed or computationally catalogued."

> 📖 *Same text as notebook — read from screen*

---

## Cell 34 — Code [1.6.2]: Count kagome/Mn materials in the dataset

**[Run this cell.]**

"This counts how many Mn-containing structures are in the training data. You will see it is a small fraction — only a few percent of the total dataset. And of those, even fewer have a kagome sublattice. This scarcity is precisely why we need generative models: to synthesize new candidates in underrepresented regions of materials space."

---

## Cell 35 — Code [1.6.2]: 3D structure visualization helper (plot_structure function)

**[Run this cell.]**

"This defines a helper function for 3D matplotlib visualizations with element-specific colors and sizes. It uses CPK-inspired colors — the same color scheme you see in chemistry textbooks. No output yet; we will use this function in the next cells."

---

## Cell 36 — Markdown [1.6.3]: "Browse dataset interactively"

"The next cell provides an interactive widget to browse individual structures from the dataset. You can select a structure, adjust atom sizes and camera angles, and see a 3D rendering. This gives you a feel for the variety of crystal geometries that SCIGEN learns from."

> 📖 *Same text as notebook — read from screen*

---

## Cell 37 — Code [1.6.3]: Interactive widget for browsing structures

**[Run this cell.]**

"You should see a dropdown or slider widget that lets you pick structures from the dataset and visualize them. If the widget does not render, it may be a Colab backend issue — you can still see the static plots from earlier cells. Try clicking through a few different structures to see the diversity of crystal geometries in the training data."

---

## Cell 38 — Markdown [1.6.4]: "Why are kagome materials interesting?"

"So why are kagome materials so interesting? Three reasons. First, geometric frustration: on a triangle of antiferromagnetically coupled spins, you cannot satisfy all pairwise interactions simultaneously, which suppresses conventional magnetic order. Second, flat electronic bands, which lead to strongly correlated physics. Third, anomalous Hall effects that are useful for spintronic devices. Despite their scientific importance, very few kagome compounds are known — that is the generation problem we are trying to solve."

> 📖 *Same text as notebook — read from screen*

---

## Cell 39 — Code [1.6.4]: Visualize the dataset structure

**[Run this cell.]**

"This uses the plot_structure function we defined earlier to render the first structure from the dataset. You should see a 3D ball-and-stick model with element labels."

---

## Cell 40 — Code [1.6.4]: PyVista rendering of key structures

**[Run this cell.]**

"This attempts to render a grid of structures using PyVista for higher-quality 3D visualization. If PyVista is not available or the Colab backend does not support it, the cell will fall back gracefully. When it works, you get interactive 3D models you can rotate and zoom."

---

## Cell 41 — Markdown [1.6.5]: Exercise section

"The notebook has a few optional exercises here: pick a different structure from the dataset, try building a honeycomb lattice from scratch, or create a 2x2x1 supercell of the kagome structure. These reinforce the L, X, A concepts we just covered."

> 📖 *Same text as notebook — read from screen*

"We will skip these for now and move on to the kagome lattice section."

---

## Cell 42 — Markdown [1.7]: Section 1.7 "The kagome lattice — our running example"

"The kagome lattice is named after the Japanese basket-weaving pattern — kagome literally means 'basket eye' in Japanese. It consists of corner-sharing triangles arranged in a 2D plane. The minimal unit cell has just 3 atoms at specific fractional positions: (0.5, 0, 0), (0, 0.5, 0), and (0.5, 0.5, 0) — these are the midpoints of the lattice edges. The cell is hexagonal with gamma equal to 120 degrees to create the six-fold symmetry."

> 📖 *Same text as notebook — read from screen*

---

## Cell 43 — Code [1.7]: Build a kagome lattice with Mn atoms

**[Run this cell.]**

"We create a hexagonal cell with a equal to 5 angstroms and c equal to 6 angstroms, then place 3 Mn atoms at the kagome positions: (0.5, 0, 0), (0, 0.5, 0), and (0.5, 0.5, 0). The output shows the lattice parameters and the 3-atom structure. These are exactly the fractional positions that SCIGEN targets when you set the structural constraint to kagome."

---

## Cell 44 — Code [1.7]: Visualize kagome lattice (2D supercell)

**[Run this cell.]**

"This builds a supercell and projects the kagome lattice into 2D so you can see the characteristic pattern of corner-sharing triangles and hexagonal voids. This is the basket-weaving pattern the lattice is named after. Each triangle shares its corners with neighboring triangles — this connectivity is what creates geometric frustration."

---

## Cell 45 — Markdown [1.7]: Geometric frustration in magnets

"Here is why the kagome lattice matters for magnetism. The corner-sharing triangle pattern means that if you try to make neighboring spins antiparallel — the lowest-energy arrangement for antiferromagnets — you cannot satisfy all three bonds on a triangle simultaneously. This frustration prevents conventional magnetic ordering and opens the door to exotic quantum states like spin liquids. In Notebook 04, we will use SCIGEN to generate new crystal structures that contain this kagome sublattice."

> 📖 *Same text as notebook — read from screen*

---

## Cell 46 — Markdown [1.7]: Section 1.8 Tight-binding band structures

"This section answers the question: why do physicists care so much about specific lattice geometries? Because the lattice shape directly determines the electronic band structure. Using a simple tight-binding model, we can show that the kagome lattice produces a perfectly flat band plus a Dirac cone, the honeycomb lattice produces Dirac cones like in graphene, and the triangular lattice produces ordinary dispersive bands. These are qualitatively different electronic properties arising purely from geometry."

> 📖 *Same text as notebook — read from screen*

---

## Cell 47 — Code [1.7]: Compute tight-binding band structures for kagome, honeycomb, triangular

**[Run this cell.]**

"This cell installs PythTB if needed, then computes the tight-binding band structures for all three lattice types along the standard hexagonal high-symmetry path: Gamma to M to K to Gamma. It should run in a few seconds."

"Look at the kagome panel first. The perfectly flat band — that horizontal line — is the signature feature. Electrons in a flat band have zero group velocity; they are effectively localized. This means electron-electron interactions dominate over kinetic energy, which is exactly the regime where exotic correlated physics emerges: fractional quantum Hall-like states, unconventional superconductivity, and magnetic ordering driven purely by interactions. The flat band is not a fine-tuning accident — it is a topological consequence of the kagome geometry. It persists regardless of the hopping parameter value."

"Now compare with honeycomb — you see two Dirac cones, the same band structure as graphene. And the triangular lattice has simple cosine-like dispersive bands with no special features. Geometry alone determines these qualitative differences."

---

## Cell 48 — Markdown [1.8.1]: "What happens with interlayer atoms?"

"In real materials, kagome layers are never isolated. There are always atoms between the layers — filler or interstitial sites. These hybridize with the kagome orbitals and can destroy the flat band by giving it dispersion. The next cell models this with a coupling parameter called t-prime."

> 📖 *Same text as notebook — read from screen*

---

## Cell 49 — Code [1.8.1]: Kagome + interstitial atom at varying coupling strengths

**[Run this cell.]**

"You should see four panels showing the kagome band structure with an interstitial atom coupled at different strengths: t-prime equals 0, 0.2, 0.5, and 1.0. At t-prime equals zero, you recover the pure kagome flat band. As t-prime increases, the flat band broadens and gains dispersion. This is why the choice of interstitial atoms matters in real kagome materials — and why SCIGEN's ability to control the full 3D structure, not just the kagome layer, is so important."

---

## Cell 50 — Markdown [1.8.2]: "Effect of hopping strength on band width"

"Different elements on the kagome sites correspond to different hopping integrals. 3d transition metals like Mn, Fe, and Co have narrow d-bands with small hopping. 4d and 5d metals have larger hopping. But here is the key insight: the flat band remains perfectly flat regardless of hopping strength. Only the dispersive bands change their bandwidth. This robustness is a topological property of the kagome geometry — it is not a chemical accident."

> 📖 *Same text as notebook — read from screen*

---

## Cell 51 — Code [1.8.2]: Kagome with different hopping strengths

**[Run this cell.]**

"Four panels with different hopping values mimicking different atom types: Mn-like, Fe-like, Co-like, and Ru-like. In every case, the flat band remains perfectly flat. The dispersive bands scale linearly with the hopping parameter. This confirms that the flat band is a geometric invariant — it does not depend on chemistry, only on the lattice topology."

---

## Cell 52 — Markdown [1.8.2]: Section 1.9 Interactive 3D visualization

"For publication-quality rendering, we can use PyVista. The helper module crystal_viz.py provides Jmol-standard element colors, van der Waals radii for atom sizing, periodic boundary atom images, and interactive controls for atom scale, camera angle, and supercell size."

> 📖 *Same text as notebook — read from screen*

---

## Cell 53 — Code [1.8.2]: Setup for interactive PyVista viewer

**[Run this cell.]**

"This cell sets up the PyVista-based interactive viewer by importing the crystal_viz helper module and building a list of structures to browse. If PyVista is not available, it will print a fallback message. On Colab, the trame backend is needed for interactivity."

---

## Cell 54 — Code [1.8.2]: Interactive kagome viewer widget

**[Run this cell.]**

"This creates a dedicated viewer for the kagome structure with dropdown options for different supercell sizes: 1x1x1, 2x2x1, and 3x3x1. You can rotate and zoom the 3D model. The 3x3x1 supercell is particularly nice — you can clearly see the repeating pattern of triangles and hexagons characteristic of the kagome lattice."

---

## Cell 55 — Markdown [1.10]: Section 1.10 Space group analysis

"There are exactly 230 space groups — this is the complete classification of all 3D crystal symmetries. Every crystal belongs to one of them. Space groups matter for SCIGEN because the training data is not uniformly distributed across them. Some symmetries are much more common than others. Generated structures should reproduce this distribution if the model has learned realistic crystallography."

> 📖 *Same text as notebook — read from screen*

---

## Cell 56 — Code [1.10]: SpacegroupAnalyzer on NaCl and dataset structure

**[Run this cell.]**

"You should see the space group analysis for NaCl: Fm-3m, space group number 225, cubic crystal system, face-centered cubic lattice. Then the same analysis for whichever structure was loaded from the dataset. This shows how pymatgen automatically determines the symmetry of any structure."

---

## Cell 57 — Code [1.10]: Space group distribution plot

**[Run this cell.]**

"This bar chart shows how the 230 space groups are distributed in the MP-20 training data. You will see that a handful of space groups dominate — especially some cubic and hexagonal groups — while many others have only a few or zero representatives. The colored bands mark the seven crystal systems: triclinic, monoclinic, orthorhombic, tetragonal, trigonal, hexagonal, and cubic. This non-uniform distribution is something SCIGEN's diffusion model must learn and reproduce."

---

## Cell 58 — Markdown [1.10]: Section 1.11 Simulated XRD

"X-ray diffraction is the primary experimental technique for determining crystal structures. Pymatgen can simulate the powder XRD pattern from any structure. This is useful for comparing generated structures against experimental data, for identifying crystal phases, and for validating that a generated structure matches the intended symmetry."

> 📖 *Same text as notebook — read from screen*

---

## Cell 59 — Code [1.10]: Simulated XRD patterns

**[Run this cell.]**

"You should see two XRD patterns stacked vertically — one for NaCl and one for the dataset structure. The peaks are at specific 2-theta angles determined by the lattice spacing via Bragg's law. The strongest peaks are labeled with their Miller indices. If you were to synthesize one of SCIGEN's generated materials and measure its XRD pattern, you would compare it against a simulation like this to confirm the crystal phase."

---

## Cell 60 — Markdown [1.11.1]: Key takeaways

"Let me emphasize three key points from this notebook. First, materials databases like the Materials Project are large, but they mostly cover well-known, stable materials. Second, exotic geometries like kagome and honeycomb are extremely rare in these databases. Third, generative models give us the ability to explore the vast space of possible materials that have not been made yet — materials that might not even exist in nature. That is the core motivation for SCIGEN."

> 📖 *Same text as notebook — read from screen*

---

## Cell 61 — Markdown [1.11.1]: Appendix — Materials Project API queries

"This appendix section is optional. If you have a Materials Project API key, you can run live queries against their database to explore what is available. If you do not have one, no worries — all the data we need is already loaded in the CSV files. You can get a free API key at materialsproject.org/api if you want to try this later on your own."

> 📖 *Same text as notebook — read from screen*

---

## Cell 62 — Code [1.11.1]: Set MP API key (optional)

**[Run this cell.]**

"By default, the API key is set to None, so live queries are disabled. If you have a key, you can paste it here. Otherwise, the cell just prints a message saying it will use pre-downloaded data. We will skip the live queries for now and move on."

---

## Cell 63 — Markdown [A.1]: Querying for Mn-containing materials

"The notebook explains that if you have an API key, the next cells will query the Materials Project for stable Mn-containing materials with small unit cells, similar to what is in SCIGEN's training data. This is how researchers discover how few kagome compounds currently exist in the database."

> 📖 *Same text as notebook — read from screen*

"Since we do not have an API key set, the next two code cells will simply print skip messages."

---

## Cell 64 — Code [A.1]: MP API query for Mn materials (conditional)

**[Run this cell.]**

"This cell only runs the API query if USE_MP_API is True. Since we left the API key as None, it will print a skip message. If it were active, it would search for near-stable Mn-containing materials with 20 or fewer atoms per cell and print the first 10 results."

---

## Cell 65 — Code [A.1]: Download a specific structure from MP (conditional)

**[Run this cell.]**

"Same as above — this is conditional on having an API key. It would download the Mn metal structure (mp-35) from the Materials Project. With no key, it just prints a skip message."

---

## Cell 66 — Markdown [A.1]: References

"The notebook lists the references for the tools we used throughout: pymatgen for crystal structure manipulation, PythTB for tight-binding band structure calculations, PyVista for 3D visualization, the mp-api client for Materials Project queries, the Materials Project dataset itself, and the SCIGEN paper. All of these are open-source and freely available."

> 📖 *Same text as notebook — read from screen*

---

## Cell 67 — Markdown [A.1]: What's next?

"In Notebook 02, we will learn how generative diffusion models work and why they are well-suited for crystal structure generation. We will train a simple DDPM on MNIST to build intuition for the denoising process before applying it to crystals in later notebooks."

> 📖 *Same text as notebook — read from screen*

---

## Key takeaways to emphasize before moving on

1. **Crystal = (L, X, A)** — lattice + fractional coordinates + atom types. This is SCIGEN's language.
2. **Fractional coordinates are periodic** — they live on a torus, which is why SCIGEN uses wrapped diffusion.
3. **Crystal graphs** encode periodicity through PBC-aware neighbor lists. GNNs are naturally invariant to translation.
4. **Four invariances** — permutation, periodic translation, unit cell choice, rotation — must all be respected.
5. **Kagome materials are rare** in databases but scientifically valuable. This scarcity motivates generative modeling.
6. **Lattice geometry determines electronics** — the flat band is a topological property of the kagome lattice, not a chemical accident.

---

## Transition to Notebook 02

"Now you know how crystals are represented and why specific lattice geometries matter. In the next notebook, we will learn how diffusion models can generate new crystals from noise — starting with a hands-on DDPM training on MNIST to build intuition for the denoising process."
