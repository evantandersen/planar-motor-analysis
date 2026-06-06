import numpy as np

from analysis import Coil, Phase, Motor

span = 2

def ideal_combined_dist(n=1000):
    x = np.linspace(0, 1, n, endpoint=False)
    y = np.linspace(0, 1, n, endpoint=False)

    xv, yv = np.meshgrid(x, y)


    p_x = (xv) * span * 2 * np.pi
    p_y = (yv) * span * 2 * np.pi

    # phase shifts to look nice on psi plot. Doesn't affect analysis at all
    # (which makes sense, as we're just shifting the entire motor)
    X = np.cos(p_y + np.pi/2)
    Y = np.cos(p_x - np.pi/2)

    return Coil(X, Y, span, 1)

def get_5cell_clifford_phases():
    # Indices for the 5 vertices
    k = np.arange(5)

    phases = []
    for i in range(5):
        # Angles with winding speeds 1 and 2
        theta = 1 * i * 2 * np.pi / 5
        phi   = 2 * i * 2 * np.pi / 5
        phases.append(Phase(theta, phi, False))

    return phases

def ideal_5_phase():
    # a 5-phase coil has equal current in X and Y
    coil = ideal_combined_dist()

    # distribute the phases as equally spaced points on the clifford torus
    phases = get_5cell_clifford_phases()

    return Motor(coil=coil, phases=phases)


def ideal_4_phase():
    coil = ideal_combined_dist()

    phases = [
        Phase(theta=0, phi=0, z_rotation=False),
        Phase(theta=0, phi=np.pi, z_rotation=False),
        Phase(theta=np.pi/2, phi=np.pi/2, z_rotation=False),
        Phase(theta=-np.pi/2, phi=np.pi/2, z_rotation=False),
    ]

    return Motor(coil=coil, phases=phases)

