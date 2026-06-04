import numpy as np

from analysis import Coil


def racetrack_coil(pole_pitch, coil_width, bundle_width, coil_length, n_per_pole=100):

    #determine the minimum number of pole pairs to contain the coil
    span = int(max(np.ceil(coil_length/(pole_pitch*2)), np.ceil(coil_width/(pole_pitch*2))))
    npoles = span*2
    n = n_per_pole*npoles

    #compute some thresholds for the code below
    #this was adapted from a different method earlier and I don't want to rewrite it.
    #definitely not the simplest way but it works
    coil_start_v = (coil_width / 2) - bundle_width
    coil_end_v = (coil_width / 2)
    coil_str_len = coil_length - coil_width
    coil_str_start = -coil_str_len / 2
    coil_str_end = -coil_str_start

    coil_start_h = coil_str_start - coil_width / 2
    coil_end_h = -coil_start_h

    x = np.linspace(0, 1, n, endpoint=False)
    y = np.linspace(0, 1, n, endpoint=False)

    xv, yv = np.meshgrid(x, y)

    #following code vectorized correctly on the first attempt by gemini!

    # 1. Pre-calculate coordinate grids (Vectorized)
    P_X = (xv - 0.5) * npoles * pole_pitch
    P_Y = (yv - 0.5) * npoles * pole_pitch

    # Initialize output arrays
    X = np.zeros_like(xv)
    Y = np.zeros_like(yv)

    # --- Region 1: Coil top straight section ---
    mask1 = (coil_start_v <= P_Y) & (P_Y < coil_end_v) & \
            (coil_str_start <= P_X) & (P_X < coil_str_end)
    X[mask1] = 1
    Y[mask1] = 0

    # --- Region 2: Negative vertical section ---
    mask2 = (-coil_end_v <= P_Y) & (P_Y < -coil_start_v) & \
            (coil_str_start <= P_X) & (P_X < coil_str_end)
    X[mask2] = -1
    Y[mask2] = 0

    # --- Region 3: Left curved section ---
    # We define the mask first to avoid unnecessary sqrt/arctan calculations on the whole grid
    mask3_rect = (-coil_end_v <= P_Y) & (P_Y < coil_end_v) & \
                 (coil_start_h <= P_X) & (P_X < coil_str_start)

    rel_x3 = P_X - coil_str_start
    rel_y3 = P_Y
    dist3 = np.sqrt(rel_x3 ** 2 + rel_y3 ** 2)

    mask3_circ = (coil_width / 2 - bundle_width <= dist3) & (dist3 < coil_width / 2)
    mask3_final = mask3_rect & mask3_circ

    angle3 = np.arctan2(rel_y3[mask3_final], rel_x3[mask3_final])
    X[mask3_final] = np.sin(angle3)
    Y[mask3_final] = -np.cos(angle3)

    # --- Region 4: Right curved section ---
    mask4_rect = (-coil_end_v <= P_Y) & (P_Y < coil_end_v) & \
                 (coil_str_end <= P_X) & (P_X < coil_end_h)

    rel_x4 = P_X - coil_str_end
    rel_y4 = P_Y
    dist4 = np.sqrt(rel_x4 ** 2 + rel_y4 ** 2)

    mask4_circ = (coil_width / 2 - bundle_width <= dist4) & (dist4 < coil_width / 2)
    mask4_final = mask4_rect & mask4_circ

    angle4 = np.arctan2(rel_y4[mask4_final], rel_x4[mask4_final])
    X[mask4_final] = np.sin(angle4)
    Y[mask4_final] = -np.cos(angle4)


    total_area = coil_length * coil_width
    area_fraction = total_area / (pole_pitch * npoles) ** 2
    return Coil(j_x=X, j_y=Y, span=span, area=area_fraction)
