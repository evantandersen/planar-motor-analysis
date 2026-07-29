import numpy as np

from analysis import Coil


def generate_racetrack_coil(
        pole_pitch: float,
        coil_width: float,
        bundle_width: float,
        coil_length: float,
        points_per_unit_length: float = 10.0,
) -> Coil:
    """
    Generates a Coil object with current density components (j_x, j_y) cropped
    strictly to the bounding box of a single racetrack coil.

    `scaling` is calculated as the number of spatial grid samples per wavelength
    (2 * pole_pitch).
    """
    # Racetrack bounding box limits:
    half_length = coil_length / 2.0
    half_width = coil_width / 2.0

    # Grid dimensions proportional to physical extent
    nx = max(2, int(round(coil_length * points_per_unit_length)))
    ny = max(2, int(round(coil_width * points_per_unit_length)))

    # Spatial step size along x-axis
    dx = coil_length / (nx - 1) if nx > 1 else pole_pitch

    # Calculate scaling: samples per motor wavelength (lambda = 2 * pole_pitch)
    wavelength = 2.0 * pole_pitch
    scaling = wavelength / dx

    # Create coordinate grid over the exact bounding box
    x = np.linspace(-half_length, half_length, nx)
    y = np.linspace(-half_width, half_width, ny)
    P_X, P_Y = np.meshgrid(x, y)

    # Initialize current density arrays
    j_x = np.zeros_like(P_X)
    j_y = np.zeros_like(P_Y)

    # Geometry thresholds
    coil_start_v = half_width - bundle_width
    coil_end_v = half_width

    coil_str_len = coil_length - coil_width
    coil_str_start = -coil_str_len / 2.0
    coil_str_end = coil_str_len / 2.0

    coil_start_h = coil_str_start - half_width
    coil_end_h = coil_str_end + half_width

    # --- Region 1: Top straight section (+X current) ---
    mask1 = (coil_start_v <= P_Y) & (P_Y <= coil_end_v) & \
            (coil_str_start <= P_X) & (P_X <= coil_str_end)
    j_x[mask1] = 1.0

    # --- Region 2: Bottom straight section (-X current) ---
    mask2 = (-coil_end_v <= P_Y) & (P_Y <= -coil_start_v) & \
            (coil_str_start <= P_X) & (P_X <= coil_str_end)
    j_x[mask2] = -1.0

    # --- Region 3: Left curved section ---
    mask3_rect = (-coil_end_v <= P_Y) & (P_Y <= coil_end_v) & \
                 (coil_start_h <= P_X) & (P_X <= coil_str_start)

    rel_x3 = P_X - coil_str_start
    rel_y3 = P_Y
    dist3 = np.hypot(rel_x3, rel_y3)

    mask3_circ = (coil_start_v <= dist3) & (dist3 <= coil_end_v)
    mask3_final = mask3_rect & mask3_circ

    angle3 = np.arctan2(rel_y3[mask3_final], rel_x3[mask3_final])
    j_x[mask3_final] = np.sin(angle3)
    j_y[mask3_final] = -np.cos(angle3)

    # --- Region 4: Right curved section ---
    mask4_rect = (-coil_end_v <= P_Y) & (P_Y <= coil_end_v) & \
                 (coil_str_end <= P_X) & (P_X <= coil_end_h)

    rel_x4 = P_X - coil_str_end
    rel_y4 = P_Y
    dist4 = np.hypot(rel_x4, rel_y4)

    mask4_circ = (coil_start_v <= dist4) & (dist4 <= coil_end_v)
    mask4_final = mask4_rect & mask4_circ

    angle4 = np.arctan2(rel_y4[mask4_final], rel_x4[mask4_final])
    j_x[mask4_final] = np.sin(angle4)
    j_y[mask4_final] = -np.cos(angle4)

    return Coil(j_x=j_x, j_y=j_y, scaling=scaling)