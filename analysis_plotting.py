import string

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import zoom


def plot_phase_wave_vectors(ax_x, ax_y, phases):
    """
    Plots the set of phase vectors forming the frame in R4 onto two polar charts
    (X-fundamental and Y-fundamental components).

    Parameters
    ----------
    ax_x, ax_y : matplotlib.axes.Axes
        Polar axes (`subplot_kw={'projection': 'polar'}`).
    phases : np.ndarray
        Array of shape (4, N_phases) where each column represents a phase vector
        [v_x_cos, v_x_sin, v_y_cos, v_y_sin].
    """
    n_phases = phases.shape[1]

    # Generate distinct colors for each phase
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_phases)))

    # --- Plot 1: X Fundamental Components (Rows 0 & 1) ---
    _plot_phase_vectors(
        ax=ax_x,
        cos_vals=phases[0, :],
        sin_vals=phases[1, :],
        colors=colors,
        title="X Fundamental Frame"
    )

    # --- Plot 2: Y Fundamental Components (Rows 2 & 3) ---
    _plot_phase_vectors(
        ax=ax_y,
        cos_vals=phases[2, :],
        sin_vals=phases[3, :],
        colors=colors,
        title="Y Fundamental Frame"
    )


def _plot_phase_vectors(ax, cos_vals, sin_vals, colors, title):
    """Helper function to plot a set of phase vectors on a polar axis."""
    n_phases = len(cos_vals)

    # Convert Cartesian components to polar
    r = np.hypot(cos_vals, sin_vals)
    theta = np.arctan2(sin_vals, cos_vals)
    zeros = np.zeros_like(theta)

    # Plot all phase vectors simultaneously via quiver
    ax.quiver(
        zeros, zeros, theta, r,
        angles='xy',
        scale_units='xy',
        scale=1,
        color=colors[:n_phases],
        width=0.012
    )

    # Add phase labels (P0, P1, P2, ...) at the end of each vector
    max_r = np.max(r) if len(r) > 0 else 1.0
    padding = max(0.05, max_r * 0.12)

    for idx, (th, rad, col) in enumerate(zip(theta, r, colors)):
        # Generate letter labels: A, B, C... (and A1, B1... if > 26 phases)
        label = (
            string.ascii_uppercase[idx % 26]
            if idx < 26
            else f"{string.ascii_uppercase[idx % 26]}{(idx // 26) + 1}"
        )

        ax.text(
            th, rad + padding, label,
            color=col,
            fontsize=20,
            fontweight='bold',
            fontfamily='sans-serif',
            ha='center',
            va='center'
        )

    # Clean up gridlines and ticks
    ax.set_rticks([])
    ax.set_thetagrids([])
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=14, pad=15)

    # Set radial limit with headroom for the labels
    ax.set_ylim(0, max_r + padding * 2.5)
    
    
def plot_copper(motor, ax):
    copper_rgb = (0.72, 0.45, 0.20)
    cmap = ListedColormap([(0, 0, 0, 0), copper_rgb])

    # 1. Base copper binary mask (1 where current exists, 0 elsewhere)
    copper_base = np.where((motor.coil.j_x != 0) | (motor.coil.j_y != 0), 1, 0)

    # Physical size of an unrotated coil in units of lambda
    coil_ny, coil_nx = motor.coil.j_x.shape
    coil_len_lambda = coil_nx / motor.coil.scaling  # length along X when unrotated
    coil_wid_lambda = coil_ny / motor.coil.scaling  # width along Y when unrotated

    all_x_min, all_x_max = [], []
    all_y_min, all_y_max = [], []

    for phase in motor.phases:
        c = copper_base.copy()

        # 2. Assign physical bounding dimensions based on rotation
        if phase.z_rotation:
            c = c.T  # Transpose grid (shape becomes nx, ny)
            # Swap physical extents so imshow doesn't stretch/squish the transposed array
            w_lambda = coil_wid_lambda  # Width along X is now the coil's Y-dimension
            h_lambda = coil_len_lambda  # Height along Y is now the coil's X-dimension
        else:
            w_lambda = coil_len_lambda
            h_lambda = coil_wid_lambda

        # 3. Compute physical extent centered on phase (theta, phi)
        x_min = phase.theta - w_lambda / 2.0
        x_max = phase.theta + w_lambda / 2.0
        y_min = phase.phi - h_lambda / 2.0
        y_max = phase.phi + h_lambda / 2.0

        all_x_min.append(x_min)
        all_x_max.append(x_max)
        all_y_min.append(y_min)
        all_y_max.append(y_max)

        # 4. Plot with matching extent
        ax.imshow(
            c,
            cmap=cmap,
            alpha=0.8,
            vmin=0,
            vmax=2,
            interpolation="nearest",
            extent=[x_min, x_max, y_min, y_max],
            origin="lower",
        )

    # 5. Framing and view limits
    pad = 0.1
    ax.set_xlim(min(all_x_min) - pad, max(all_x_max) + pad)
    ax.set_ylim(min(all_y_min) - pad, max(all_y_max) + pad)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Position ($x / \lambda$)")
    ax.set_ylabel(r"Position ($y / \lambda$)")

def plot_current_dist(motor, ax):
    # 1. Determine physical dimensions of the coil in units of lambda (lambda = 2 * pole_pitch)
    # coil.scaling is grid points per lambda
    ny, nx = motor.coil.j_x.shape
    coil_len_lambda = nx / motor.coil.scaling  # Physical length along X
    coil_wid_lambda = ny / motor.coil.scaling  # Physical width along Y

    # Physical extents centered at (0, 0)
    x_min, x_max = -coil_len_lambda / 2.0, coil_len_lambda / 2.0
    y_min, y_max = -coil_wid_lambda / 2.0, coil_wid_lambda / 2.0

    # 2. Create the background magnetic field grid matched strictly to the coil bounding box
    # Wavenumber k0 in units of 1/lambda is 2*pi
    k0 = 2.0 * np.pi
    n_bg_x = 500
    n_bg_y = max(2, int(round(n_bg_x * (coil_wid_lambda / coil_len_lambda))))

    x_bg = np.linspace(x_min, x_max, n_bg_x)
    y_bg = np.linspace(y_min, y_max, n_bg_y)
    xv, yv = np.meshgrid(x_bg, y_bg)

    # Background field in physical spatial coordinates (k0 * x)
    magnet_field = np.cos(k0 * xv) + np.cos(k0 * yv)

    ax.imshow(
        magnet_field,
        extent=(x_min, x_max, y_min, y_max),
        alpha=0.6,
        origin="lower",
        aspect="equal"
    )

    # 3. Resample current density for quiver arrows preserving aspect ratio
    # Pick target arrow count along x and scale y accordingly
    n_arrows_x = 40
    n_arrows_y = max(2, int(round(n_arrows_x * (coil_wid_lambda / coil_len_lambda))))

    zoom_x = n_arrows_x / nx
    zoom_y = n_arrows_y / ny

    disp_jx = zoom(motor.coil.j_x, (zoom_y, zoom_x), order=3)
    disp_jy = zoom(motor.coil.j_y, (zoom_y, zoom_x), order=3)

    # 4. Generate quiver arrow coordinates over the coil extent
    disp_x = np.linspace(x_min, x_max, n_arrows_x)
    disp_y = np.linspace(y_min, y_max, n_arrows_y)
    disp_xv, disp_yv = np.meshgrid(disp_x, disp_y)

    ax.quiver(
        disp_xv,
        disp_yv,
        disp_jx,
        disp_jy,
        pivot="mid",
        color="black",
        scale=30
    )

    # 5. Framing and labels
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_title("Coil Current Distribution", fontsize=16)
    ax.set_xlabel(r"Position ($x / \lambda$)")
    ax.set_ylabel(r"Position ($y / \lambda$)")


def plot_analysis(motor, phases):
    # make the pretty plots
    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral"
    })

    fig = plt.figure(figsize=(10, 10), layout='constrained')
    axs = fig.subplot_mosaic(
        [
            ["current_dist", "copper_plot"],
            ["fund_x", "fund_y"],
         ],
        per_subplot_kw={
            "fund_x": {"projection": "polar"},
            "fund_y": {"projection": "polar"},
        },
    )
    plot_copper(motor, axs["copper_plot"])
    plot_phase_wave_vectors(axs["fund_x"], axs["fund_y"], phases)
    plot_current_dist(motor, axs["current_dist"])
    plt.show()