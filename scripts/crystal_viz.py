"""crystal_viz.py — Crystal structure visualizer for Colab notebooks.

Adapted from SCIGEN_p_agent/utils/visualize/crystal_viz.py (PyVista backend).
Adds notebook-friendly API: accepts pymatgen Structure objects directly,
returns inline images via IPython display or PIL Image.

Public API (notebooks)
----------------------
plot_crystal(struct, **kwargs) → PIL.Image
    Render a pymatgen Structure and return a PIL Image (also displays inline).

plot_crystal_grid(structures, titles=None, **kwargs) → PIL.Image
    Render a grid of structures side-by-side.

plot_crystal_interactive(struct, **kwargs) → ipywidgets.VBox
    Interactive viewer with sliders for atom_scale, elevation, azimuth, zoom.

Public API (CLI / batch)
------------------------
visualize_crystal(cif_path, **kwargs)
    Render a CIF file → PNG or interactive window.

visualize_batch(cif_dir, output_dir, **kwargs)
    Batch-render all CIFs in a directory.

Dependencies: pyvista, pymatgen, numpy, PIL (Pillow)
"""
from __future__ import annotations

import csv
import io
import os
from itertools import product as iproduct
from math import cos, radians, sin
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_JMOL_CSV: Path = Path(__file__).parent / "jmol_param.csv"
_DEFAULT_COLOR: Tuple[float, float, float] = (0.5, 0.5, 0.5)
_DEFAULT_RADIUS: float = 1.5  # Å
_VIS_TOL: float = 0.02


# ===========================================================================
# Low-level utilities
# ===========================================================================

def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    """Convert '#E06633' → (0.878, 0.40, 0.20)."""
    h = hex_color.lstrip("#")
    if len(h) == 6:
        return (int(h[0:2], 16) / 255.0,
                int(h[2:4], 16) / 255.0,
                int(h[4:6], 16) / 255.0)
    return _DEFAULT_COLOR


def blend_colors(
    colors: List[Tuple[float, float, float]],
    fractions: List[float],
) -> Tuple[float, float, float]:
    """Weighted RGB average for partial-occupancy sites."""
    total = sum(fractions)
    if total == 0:
        return _DEFAULT_COLOR
    f = [x / total for x in fractions]
    return (sum(c[0] * w for c, w in zip(colors, f)),
            sum(c[1] * w for c, w in zip(colors, f)),
            sum(c[2] * w for c, w in zip(colors, f)))


def load_atom_colors(
    csv_path: Optional[Path] = None,
) -> Tuple[Dict[str, str], Dict[str, float]]:
    """Load Jmol standard colors and van-der-Waals radii from CSV.

    Returns (color_dict, radius_dict) where radii are in Angstroms.
    """
    if csv_path is None:
        csv_path = _JMOL_CSV
    color_dict: Dict[str, str] = {}
    radius_dict: Dict[str, float] = {}
    try:
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                sym = row["atom"].strip()
                color_dict[sym] = row["color"].strip()
                radius_dict[sym] = float(row["radii"].strip()) / 100.0
    except Exception as exc:
        print(f"Warning: could not load {csv_path}: {exc}")
    return color_dict, radius_dict


def frac_to_cart(frac: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Fractional → Cartesian. *matrix* rows are lattice vectors."""
    return frac @ matrix


# ===========================================================================
# Structure preprocessing
# ===========================================================================

def _boundary_atom_extras(
    frac_coords: np.ndarray,
    atom_types: List[str],
    site_occus: List[Dict[str, float]],
    lattice_mat: np.ndarray,
    vis_tol: float = _VIS_TOL,
) -> Tuple[np.ndarray, List[str], List[Dict[str, float]]]:
    """Add periodic-image atoms near cell faces for visual completeness."""
    extra_pos: List[np.ndarray] = []
    extra_types: List[str] = []
    extra_occus: List[Dict[str, float]] = []

    for frac, atype, occ in zip(frac_coords, atom_types, site_occus):
        axis_shifts: List[List[int]] = []
        for d in range(3):
            shifts = [0]
            if frac[d] < vis_tol:
                shifts.append(+1)
            if frac[d] > 1.0 - vis_tol:
                shifts.append(-1)
            axis_shifts.append(shifts)
        for combo in iproduct(*axis_shifts):
            if all(c == 0 for c in combo):
                continue
            new_frac = frac + np.array(combo, dtype=float)
            extra_pos.append(new_frac @ lattice_mat)
            extra_types.append(atype)
            extra_occus.append(occ)

    if extra_pos:
        return np.array(extra_pos), extra_types, extra_occus
    return np.empty((0, 3)), [], []


def _expand_supercell(
    cart_coords: np.ndarray,
    atom_types: List[str],
    site_occus: List[Dict[str, float]],
    lattice_mat: np.ndarray,
    supercell: Tuple[int, int, int],
) -> Tuple[np.ndarray, List[str], List[Dict[str, float]]]:
    """Replicate na x nb x nc unit cells."""
    na, nb, nc = supercell
    exp_pos, exp_types, exp_occus = [], [], []
    for i, j, k in iproduct(range(na), range(nb), range(nc)):
        T = np.array([i, j, k], dtype=float) @ lattice_mat
        for pos, at, occ in zip(cart_coords, atom_types, site_occus):
            exp_pos.append(pos + T)
        exp_types.extend(atom_types)
        exp_occus.extend(site_occus)
    return np.array(exp_pos), exp_types, exp_occus


# ===========================================================================
# Structure parsing (from pymatgen Structure or CIF)
# ===========================================================================

def _parse_structure(struct):
    """Extract visualization data from a pymatgen Structure object."""
    frac_coords = struct.frac_coords
    cart_coords = struct.cart_coords
    lattice_mat = struct.lattice.matrix
    formula = struct.composition.reduced_formula

    atom_types: List[str] = []
    site_occus: List[Dict[str, float]] = []
    for site in struct.sites:
        occ = {str(el): frac for el, frac in site.species.items()}
        dominant = max(occ, key=lambda e: occ[e])
        atom_types.append(dominant)
        site_occus.append(occ)

    return frac_coords, cart_coords, atom_types, site_occus, lattice_mat, formula


def _parse_cif(cif_path, primitive=False):
    """Parse a CIF file with pymatgen."""
    from pymatgen.io.cif import CifParser
    struct = CifParser(str(cif_path)).parse_structures(primitive=primitive)[0]
    return _parse_structure(struct)


# ===========================================================================
# PyVista rendering
# ===========================================================================

def _ensure_pyvista():
    """Import pyvista and configure for off-screen / Colab use."""
    import pyvista as pv
    # Start virtual framebuffer if no display available
    if not os.environ.get("DISPLAY"):
        try:
            pv.start_xvfb()
        except Exception:
            pass
    pv.global_theme.jupyter_backend = "static"
    return pv


def _site_color(atype, occ, color_dict):
    """Return RGB color for a site (blended if partially occupied)."""
    if len(occ) == 1:
        return hex_to_rgb(color_dict.get(atype, "#808080"))
    colors = [hex_to_rgb(color_dict.get(el, "#808080")) for el in occ]
    fractions = list(occ.values())
    return blend_colors(colors, fractions)


def _draw_atoms(pl, positions, atom_types, site_occus,
                color_dict, radius_dict, atom_scale, opacity,
                highlight_indices=None, highlight_color=(1.0, 0.0, 0.0)):
    """Add one sphere per atom. Returns RGB color list."""
    pv = _ensure_pyvista()
    atom_colors_out = []
    for idx, (pos, atype, occ) in enumerate(zip(positions, atom_types, site_occus)):
        color = _site_color(atype, occ, color_dict)
        radius = radius_dict.get(atype, _DEFAULT_RADIUS) * atom_scale
        sphere = pv.Sphere(radius=radius, center=pos,
                           theta_resolution=30, phi_resolution=30)
        pl.add_mesh(sphere, color=color, opacity=opacity,
                    show_edges=False, smooth_shading=True,
                    ambient=0.3, diffuse=0.7, specular=0.3)
        # Highlight ring for constrained atoms
        if highlight_indices and idx in highlight_indices:
            ring = pv.Sphere(radius=radius * 1.3, center=pos,
                             theta_resolution=30, phi_resolution=30)
            pl.add_mesh(ring, color=highlight_color, opacity=0.25,
                        show_edges=False, smooth_shading=True)
        atom_colors_out.append(color)
    return atom_colors_out


def _make_parallelepiped_lines():
    return np.array([
        2, 0, 1, 2, 1, 2, 2, 2, 3, 2, 3, 0,
        2, 4, 5, 2, 5, 6, 2, 6, 7, 2, 7, 4,
        2, 0, 4, 2, 1, 5, 2, 2, 6, 2, 3, 7,
    ], dtype=int)


def _draw_wireframe(pl, frac_corners, lattice_mat, color, line_width, opacity):
    pv = _ensure_pyvista()
    corners = np.array([frac_to_cart(fc, lattice_mat) for fc in frac_corners])
    mesh = pv.PolyData()
    mesh.points = corners
    mesh.lines = _make_parallelepiped_lines()
    pl.add_mesh(mesh, color=color, line_width=line_width, opacity=opacity)


def _draw_unit_cell(pl, lattice_mat, color="black", line_width=2.0, opacity=0.7):
    frac_corners = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=float)
    _draw_wireframe(pl, frac_corners, lattice_mat, color, line_width, opacity)


def _draw_supercell_box(pl, lattice_mat, supercell,
                        color="gray", line_width=3.0, opacity=0.5):
    na, nb, nc = supercell
    frac_corners = np.array([
        [0, 0, 0], [na, 0, 0], [na, nb, 0], [0, nb, 0],
        [0, 0, nc], [na, 0, nc], [na, nb, nc], [0, nb, nc],
    ], dtype=float)
    _draw_wireframe(pl, frac_corners, lattice_mat, color, line_width, opacity)


def _add_element_legend(pl, atom_types, atom_colors):
    seen = {}
    for atype, color in zip(atom_types, atom_colors):
        if atype not in seen:
            seen[atype] = color
    if not seen:
        return
    entries = [[sym, col] for sym, col in seen.items()]
    n = len(entries)
    height = min(0.04 * n + 0.04, 0.40)
    pl.add_legend(entries, bcolor="white", border=True,
                  size=(0.14, height), loc="lower right")


def _setup_camera(pl, positions, azimuth, elevation, zoom_padding):
    center = positions.mean(axis=0)
    bbox_min, bbox_max = positions.min(axis=0), positions.max(axis=0)
    max_dim = float(np.max(bbox_max - bbox_min))
    distance = max_dim * 1.8
    az_r, el_r = radians(azimuth), radians(elevation)
    cam_pos = center + np.array([
        distance * cos(el_r) * cos(az_r),
        distance * cos(el_r) * sin(az_r),
        distance * sin(el_r),
    ])
    pl.reset_camera(bounds=(bbox_min[0], bbox_max[0],
                            bbox_min[1], bbox_max[1],
                            bbox_min[2], bbox_max[2]))
    auto_dist = float(np.linalg.norm(np.array(pl.camera.position) - center))
    pl.camera.position = cam_pos
    pl.camera.focal_point = center
    pl.camera.up = (0, 1, 0) if elevation >= 60.0 else (0, 0, 1)
    new_dist = float(np.linalg.norm(cam_pos - center))
    pl.camera.zoom((auto_dist / new_dist) * zoom_padding)


def _setup_lighting(pl, positions, preset="standard", intensity=1.0):
    center = positions.mean(axis=0)
    max_size = float(np.max(positions.max(axis=0) - positions.min(axis=0)))
    dist = max_size * 2.5
    pv = _ensure_pyvista()
    try:
        renderer = pl.renderer
        for lt in list(renderer.GetLights()):
            renderer.RemoveLight(lt)
    except Exception:
        pass
    if preset == "diffused":
        for delta in ([1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1]):
            lt = pv.Light(position=center + np.array(delta) * dist,
                          focal_point=center, intensity=intensity * 0.6)
            pl.add_light(lt)
    else:
        lt = pv.Light(position=center + np.array([dist, dist, dist]),
                      focal_point=center, intensity=intensity)
        pl.add_light(lt)


# ===========================================================================
# Public API — notebook-friendly
# ===========================================================================

def plot_crystal(
    struct,
    *,
    supercell: Tuple[int, int, int] = (1, 1, 1),
    atom_scale: float = 0.25,
    show_unit_cell: bool = True,
    loop: bool = True,
    title: Optional[str] = None,
    azimuth: float = 30.0,
    elevation: float = 75.0,
    zoom_padding: float = 0.45,
    opacity: float = 1.0,
    lighting_preset: str = "standard",
    window_size: Tuple[int, int] = (800, 600),
    background_color: str = "white",
    highlight_indices: Optional[List[int]] = None,
    jmol_csv: Optional[Path] = None,
    display: bool = True,
):
    """Render a pymatgen Structure and return a PIL Image.

    Parameters
    ----------
    struct : pymatgen.core.Structure
    supercell : (na, nb, nc) replication
    atom_scale : scale factor for atomic radii
    show_unit_cell : draw unit cell wireframe
    loop : add periodic images at cell faces
    title : text overlay (default: formula)
    highlight_indices : atom indices to highlight (e.g. constrained sites)
    display : if True, display inline in notebook
    """
    pv = _ensure_pyvista()
    color_dict, radius_dict = load_atom_colors(jmol_csv)

    frac_coords, cart_coords, atom_types, site_occus, lattice_mat, formula = \
        _parse_structure(struct)

    # Boundary atoms
    all_cart = list(cart_coords)
    all_types = list(atom_types)
    all_occus = list(site_occus)
    if loop:
        ex_pos, ex_types, ex_occus = _boundary_atom_extras(
            frac_coords, atom_types, site_occus, lattice_mat)
        if len(ex_pos) > 0:
            all_cart = list(cart_coords) + list(ex_pos)
            all_types = all_types + ex_types
            all_occus = all_occus + ex_occus

    all_cart = np.array(all_cart)

    # Supercell expansion
    if supercell != (1, 1, 1):
        all_cart, all_types, all_occus = _expand_supercell(
            all_cart, all_types, all_occus, lattice_mat, supercell)

    # Render
    pl = pv.Plotter(window_size=list(window_size), off_screen=True)
    pl.background_color = background_color

    atom_colors_list = _draw_atoms(
        pl, all_cart, all_types, all_occus,
        color_dict, radius_dict, atom_scale=atom_scale, opacity=opacity,
        highlight_indices=highlight_indices,
    )

    if show_unit_cell:
        _draw_unit_cell(pl, lattice_mat)
        if supercell != (1, 1, 1):
            _draw_supercell_box(pl, lattice_mat, supercell)

    _add_element_legend(pl, all_types, atom_colors_list)
    pl.add_text(title if title else formula, position="upper_left", font_size=12)
    pl.add_axes()

    _setup_camera(pl, all_cart, azimuth, elevation, zoom_padding)
    _setup_lighting(pl, all_cart, preset=lighting_preset)

    # Screenshot → PIL Image
    pl.show(auto_close=False)
    img_array = pl.screenshot(return_img=True)
    pl.close()

    from PIL import Image
    img = Image.fromarray(img_array)

    if display:
        try:
            from IPython.display import display as ipy_display
            ipy_display(img)
        except ImportError:
            pass

    return img


def plot_crystal_grid(
    structures,
    titles=None,
    ncols: int = 4,
    cell_size: Tuple[int, int] = (400, 350),
    **kwargs,
):
    """Render multiple structures in a grid and return a combined PIL Image.

    Parameters
    ----------
    structures : list of pymatgen Structure objects
    titles : list of title strings (one per structure)
    ncols : number of columns in the grid
    cell_size : (width, height) per cell in pixels
    **kwargs : forwarded to plot_crystal
    """
    from PIL import Image

    n = len(structures)
    if titles is None:
        titles = [s.composition.reduced_formula for s in structures]

    nrows = (n + ncols - 1) // ncols
    grid_w = ncols * cell_size[0]
    grid_h = nrows * cell_size[1]
    grid = Image.new("RGB", (grid_w, grid_h), "white")

    kwargs["display"] = False
    kwargs["window_size"] = cell_size

    for i, (struct, title) in enumerate(zip(structures, titles)):
        row, col = divmod(i, ncols)
        img = plot_crystal(struct, title=title, **kwargs)
        img = img.resize(cell_size, Image.LANCZOS)
        grid.paste(img, (col * cell_size[0], row * cell_size[1]))

    try:
        from IPython.display import display as ipy_display
        ipy_display(grid)
    except ImportError:
        pass

    return grid


def plot_crystal_interactive(
    struct,
    *,
    supercell: Tuple[int, int, int] = (1, 1, 1),
    atom_scale_range: Tuple[float, float, float] = (0.1, 1.0, 0.05),
    atom_scale_default: float = 0.3,
    azimuth_default: float = 30.0,
    elevation_default: float = 75.0,
    zoom_default: float = 0.45,
    window_size: Tuple[int, int] = (600, 500),
    **kwargs,
):
    """Interactive crystal viewer using ipywidgets sliders (Colab-friendly).

    Re-renders the structure via :func:`plot_crystal` each time a slider
    changes, displaying the resulting PIL Image in a widgets.Output area.
    This avoids the trame/panel backend entirely, which is unreliable in
    Google Colab.

    Parameters
    ----------
    struct : pymatgen.core.Structure
        The structure to visualize.
    supercell : (na, nb, nc)
        Initial supercell replication (overridden by the dropdown).
    atom_scale_range : (min, max, step)
        Range for the atom-scale slider.
    atom_scale_default : float
        Initial atom-scale value.
    azimuth_default, elevation_default, zoom_default : float
        Initial camera parameters.
    window_size : (w, h)
        Pixel size of each rendered image.
    **kwargs
        Extra keyword arguments forwarded to :func:`plot_crystal`.

    Returns
    -------
    ipywidgets.VBox
        Widget container that notebooks can display with ``display(vbox)``.
        If ipywidgets is unavailable, falls back to a static
        :func:`plot_crystal` call and returns the PIL Image instead.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display as ipy_display, clear_output
    except ImportError:
        print("[crystal_viz] ipywidgets not available — falling back to static render.")
        return plot_crystal(struct, supercell=supercell,
                            atom_scale=atom_scale_default,
                            azimuth=azimuth_default,
                            elevation=elevation_default,
                            zoom_padding=zoom_default,
                            window_size=window_size, **kwargs)

    # --- slider widgets ---------------------------------------------------
    sc_min, sc_max, sc_step = atom_scale_range
    w_atom_scale = widgets.FloatSlider(
        value=atom_scale_default, min=sc_min, max=sc_max, step=sc_step,
        description="Atom scale", continuous_update=False,
        style={"description_width": "90px"},
    )
    w_elevation = widgets.FloatSlider(
        value=elevation_default, min=0.0, max=90.0, step=1.0,
        description="Elevation", continuous_update=False,
        style={"description_width": "90px"},
    )
    w_azimuth = widgets.FloatSlider(
        value=azimuth_default, min=0.0, max=360.0, step=1.0,
        description="Azimuth", continuous_update=False,
        style={"description_width": "90px"},
    )
    w_zoom = widgets.FloatSlider(
        value=zoom_default, min=0.1, max=1.0, step=0.05,
        description="Zoom", continuous_update=False,
        style={"description_width": "90px"},
    )

    supercell_options = [
        ("1x1x1", (1, 1, 1)),
        ("2x2x1", (2, 2, 1)),
        ("2x2x2", (2, 2, 2)),
        ("3x3x1", (3, 3, 1)),
    ]
    # Find the initial value that matches the supercell argument
    sc_init = supercell if supercell in [v for _, v in supercell_options] else (1, 1, 1)
    w_supercell = widgets.Dropdown(
        options=supercell_options, value=sc_init,
        description="Supercell",
        style={"description_width": "90px"},
    )

    out = widgets.Output()

    def _render(**_ignore):
        with out:
            clear_output(wait=True)
            try:
                img = plot_crystal(
                    struct,
                    supercell=w_supercell.value,
                    atom_scale=w_atom_scale.value,
                    elevation=w_elevation.value,
                    azimuth=w_azimuth.value,
                    zoom_padding=w_zoom.value,
                    window_size=window_size,
                    display=False,
                    **kwargs,
                )
                ipy_display(img)
            except Exception as exc:
                print(f"Render error: {exc}")

    # Wire up interactive output
    interactive_out = widgets.interactive_output(
        _render,
        {
            "atom_scale": w_atom_scale,
            "elevation": w_elevation,
            "azimuth": w_azimuth,
            "zoom": w_zoom,
            "supercell": w_supercell,
        },
    )

    controls = widgets.VBox([w_atom_scale, w_elevation, w_azimuth, w_zoom, w_supercell])
    container = widgets.VBox([controls, out])

    # Initial render
    _render()

    return container


# ===========================================================================
# Public API — CIF file / CLI
# ===========================================================================

def visualize_crystal(
    cif_path,
    *,
    output_file: Optional[str] = None,
    interactive: bool = False,
    primitive: bool = False,
    **kwargs,
):
    """Visualize a CIF file. Saves PNG or opens interactive window."""
    from pymatgen.io.cif import CifParser

    cif_path = Path(cif_path)
    struct = CifParser(str(cif_path)).parse_structures(primitive=primitive)[0]

    if interactive:
        pv = _ensure_pyvista()
        # For interactive, bypass plot_crystal and use show()
        kwargs["display"] = False
        img = plot_crystal(struct, **kwargs)
        print(f"[crystal_viz] Interactive mode requires X11. Saved static image instead.")

    if output_file is None:
        formula = struct.composition.reduced_formula
        output_file = f"{formula}_{cif_path.stem}.png"

    img = plot_crystal(struct, display=False, **kwargs)
    img.save(output_file)
    print(f"[crystal_viz] Saved → {output_file}")


def visualize_batch(cif_dir, output_dir, glob="*.cif", **kwargs):
    """Batch-render all CIFs in a directory."""
    cif_dir = Path(cif_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cif_files = sorted(cif_dir.glob(glob))
    if not cif_files:
        print(f"[crystal_viz] No CIF files matching '{glob}' in {cif_dir}")
        return

    print(f"[crystal_viz] Batch: {len(cif_files)} files → {output_dir}")
    for idx, cif_path in enumerate(cif_files, 1):
        out_png = output_dir / f"{cif_path.stem}.png"
        print(f"  [{idx}/{len(cif_files)}] {cif_path.name}")
        try:
            visualize_crystal(cif_path, output_file=str(out_png), **kwargs)
        except Exception as exc:
            print(f"    ERROR: {exc}")
