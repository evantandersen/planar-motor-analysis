import numpy as np

from analysis import Phase, Motor, Coil

span = 5


def generate_current_loop(periods=1, num_points=500, width=3.0, height=2.0, thickness=0.4):
    """
    Generates X and Y current distributions for a rectangular loop.

    Parameters:
    -----------
    periods : int
        Number of 2*pi spatial periods for the square domain side.
    num_points : int
        Grid resolution (num_points x num_points).
    width : float
        Horizontal size of the loop center-line (in radians).
    height : float
        Vertical size of the loop center-line (in radians).
    thickness : float
        The total thickness of the wire/current path (in radians).
    """
    # 1. Define the square domain from -pi*periods to +pi*periods
    limit = np.pi * periods
    x = np.linspace(-limit, limit, num_points)
    y = np.linspace(-limit, limit, num_points)
    X, Y = np.meshgrid(x, y)

    # Initialize current arrays to zero
    Jx = np.zeros_like(X)
    Jy = np.zeros_like(Y)

    adjusted_width = width # - thickness
    adjusted_height = height #- thickness

    # 2. Define the boundaries of the rectangular wire
    half_w = adjusted_width / 2.0
    half_h = adjusted_height / 2.0
    half_t = thickness / 2.0

    # 3. Create masks for the 4 segments of the rectangle
    # Top segment (Current flowing right: +X direction)
    top_mask = (X >= -half_w - half_t) & (X <= half_w + half_t) & \
               (Y >= half_h - half_t) & (Y <= half_h + half_t)

    # Bottom segment (Current flowing left: -X direction)
    bottom_mask = (X >= -half_w - half_t) & (X <= half_w + half_t) & \
                  (Y >= -half_h - half_t) & (Y <= -half_h + half_t)

    # Right segment (Current flowing down: -Y direction)
    right_mask = (X >= half_w - half_t) & (X <= half_w + half_t) & \
                 (Y >= -half_h - half_t) & (Y <= half_h + half_t)

    # Left segment (Current flowing up: +Y direction)
    left_mask = (X >= -half_w - half_t) & (X <= -half_w + half_t) & \
                (Y >= -half_h - half_t) & (Y <= half_h + half_t)

    # 4. Assign current values (normalized to 1 inside the wire)
    Jx[top_mask] += 1.0
    Jx[bottom_mask] += -1.0
    Jy[right_mask] += -1.0
    Jy[left_mask] += 1.0

    # Smooth out corners where segments overlap to maintain uniform magnitude if desired,
    # or just let them add/subtract. Here we clip to keep max magnitude reasonable.
    # For a pure vector field, we can normalize non-zero vectors.
    magnitude = np.sqrt(Jx ** 2 + Jy ** 2)
    mask = magnitude > 0
    Jx[mask] /= magnitude[mask]
    Jy[mask] /= magnitude[mask]

    total_domain_area = (2 * limit) ** 2

    # Method A: Analytical (Exact math)
    # Assumes width > thickness and height > thickness so the middle hole exists
    analytical_coil_area = 2 * thickness * (adjusted_width + adjusted_height)
    analytical_fraction = analytical_coil_area / total_domain_area

    # hack_area = thickness * 2 / (2*limit)

    return Coil(Jx, Jy, span, analytical_fraction)

def six_phase_interleaved():
    coil = generate_current_loop(periods=span, num_points=3000, width=7.0*np.pi, height=np.pi, thickness=np.pi/3)

    phases = [
        #y forcers
        Phase(theta=0, phi=-np.pi/3, z_rotation=False),
        Phase(theta=0, phi=0, z_rotation=False),
        Phase(theta=0, phi=np.pi/3, z_rotation=False),

        #x forcers
        Phase(theta=-np.pi/3, phi=0, z_rotation=True),
        Phase(theta=0, phi=0, z_rotation=True),
        Phase(theta=np.pi/3, phi=0, z_rotation=True),
    ]


    # phases = [
    #     #y forcers
    #     Phase(theta=0, phi=-np.pi / 3, z_rotation=False),
    #     Phase(theta=0, phi=-np.pi/6, z_rotation=False),
    #     Phase(theta=0, phi=0, z_rotation=False),
    #     Phase(theta=0, phi=np.pi/6, z_rotation=False),
    #     Phase(theta=0, phi=np.pi / 3, z_rotation=False),
    #     Phase(theta=0, phi=np.pi / 2, z_rotation=False),
    #
    #     #x forcers
    #     Phase(theta=-np.pi/3, phi=0, z_rotation=True),
    #     Phase(theta=-np.pi/6, phi=0, z_rotation=True),
    #     Phase(theta=0, phi=0, z_rotation=True),
    #     Phase(theta=np.pi/6, phi=0, z_rotation=True),
    #     Phase(theta=np.pi / 3, phi=0, z_rotation=True),
    #     Phase(theta=np.pi / 2, phi=0, z_rotation=True),
    # ]


    return Motor(coil=coil, phases=phases)

