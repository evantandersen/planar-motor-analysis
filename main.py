import sys

import numpy as np

from analysis import analyze
from geometry.dlpm_2013 import double_layer_planar_motor
from geometry.dlpm_2022_Kleijer import dlpm_2022
from geometry.ideal_distributed import ideal_5_phase, ideal_4_phase
from linalg import *


def generate_circle_points(n):
    """
    Generates n points evenly distributed around the unit circle.
    Returns a list of 2x1 numpy column vectors.
    """
    # Create n evenly spaced angles from 0 to 2*pi
    # endpoint=False prevents the last point from overlapping the first (0 and 2pi)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # Calculate x and y coordinates
    x = np.cos(angles)
    y = np.sin(angles)

    # Stack them into column vectors: shape (n, 2)
    points = np.stack((x, y), axis=0)

    return points


def generate_minimal_8_frame(k):
    # 1. Radii
    r2 = 1 / np.sqrt(k ** 2 + 1)
    r1 = k * r2

    # 2. To kill the off-diagonals, we need sign symmetry.
    # We create 4 vectors for Set A and 4 for Set B with alternating signs.

    # Set A: (r1, 0, r2, 0) variations
    # We use a Hadamard-style sign pattern to ensure cross-terms cancel.
    vectors = [
        [r1, 0, r2, 0],
        [r1, 0, -r2, 0],
        [0, r1, 0, r2],
        [0, r1, 0, -r2],

        # Set B: (r2, 0, r1, 0) variations
        [r2, 0, 0, r1],
        [r2, 0, 0, -r1],
        [0, r2, r1, 0],
        [0, r2, -r1, 0]
    ]

    return np.array(vectors).T


def generate_generalized_clifford_frame(N, k):
    """
    Generates a tight frame for R^4 where vectors lie on generalized Clifford tori.

    Args:
        N (int): Number of vectors in the base 2D tight frame (N-gon).
        k (float): The ratio between the magnitudes of the two planes.

    Returns:
        np.ndarray: A 4 x (2 * N^2) matrix of column vectors.
    """
    # 1. Define the radii r1 and r2 based on the ratio k
    # We want r1^2 + r2^2 = 1 to keep vectors on the unit 3-sphere
    r2 = 1 / np.sqrt(k ** 2 + 1)
    r1 = k * r2

    # 2. Create the base 2D tight frame (Roots of Unity)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    if N == 2:
        angles = [0, np.pi/2]
    u = np.stack([np.cos(angles), np.sin(angles)], axis=0)  # Shape (2, N)

    vectors = []

    # 3. Generate Set A: (r1 * u_i, r2 * u_j)
    # This covers the ratio r1/r2 = k
    for i in range(N):
        for j in range(N):
            v_a = np.concatenate([r1 * u[:, i], r2 * u[:, j]])
            vectors.append(v_a)

    # 4. Generate Set B: (r2 * u_i, r1 * u_j)
    # This covers the flipped ratio r2/r1 = 1/k
    for i in range(N):
        for j in range(N):
            v_b = np.concatenate([r2 * u[:, i], r1 * u[:, j]])
            vectors.append(v_b)

    # Convert list to a 4 x (2*N^2) matrix
    return np.array(vectors).T


def generate_biased_basis(ratio_v1_v2=3.0):
    """
    Generates an orthonormal basis for R4 where:
    - Vectors 1 & 2 have 'ratio' times more energy in x1x2 than x3x4.
    - Vectors 3 & 4 have 1/'ratio' times the energy (the inverse).
    """
    # Calculate the angle required to get the specific energy distribution
    # Energy in x1x2 = cos(theta)^2
    # Energy in x3x4 = sin(theta)^2
    # ratio = cos^2 / sin^2  =>  tan^2 = 1/ratio
    theta = np.arctan(1 / np.sqrt(ratio_v1_v2))

    c, s = np.cos(theta), np.sin(theta)

    # Construction of the orthogonal matrix Q
    # This performs a simultaneous rotation in the (1,3) and (2,4) planes
    Q = np.array([
        [c, 0, -s, 0],
        [0, c, 0, -s],
        [s, 0, c, 0],
        [0, s, 0, c]
    ])

    return Q


def generate_tight_biased_frame_v3(ratio=3.0):
    # Energy distribution constants
    # k = ratio, w1 = k/(k+1), w2 = 1/(k+1)
    # c is sqrt(w1), s is sqrt(w2)
    theta = np.arctan(1 / np.sqrt(ratio))
    c, s = np.cos(theta), np.sin(theta)

    frame = np.zeros((4, 6))

    # We use 6 vectors total. To make it tight, we use 6-th roots of unity
    # spread across the two planes with a relative phase shift.
    angles = np.array([2 * np.pi * k / 6 for k in range(6)])

    # Plane 1 (x1, x2) uses the standard angles
    # Plane 2 (x3, x4) uses the same angles but 'skips' to create independence
    # This is equivalent to a Harmonic Frame construction.
    for i in range(6):
        # We alternate the 'heavy' side to keep the total energy balanced
        # Vectors 0,2,4 are heavy in x1x2; 1,3,5 are heavy in x3x4
        if i % 2 == 0:
            c_val, s_val = c, s
        else:
            c_val, s_val = s, c

        frame[0, i] = c_val * np.cos(angles[i])
        frame[1, i] = c_val * np.sin(angles[i])

        # Adding a specific phase skip (e.g., 2*angles) to the second plane
        # ensures the cross-correlations sum to zero.
        frame[2, i] = s_val * np.cos(2 * angles[i])
        frame[3, i] = s_val * np.sin(2 * angles[i])

    return frame



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=3)

    # motor = ideal_4_phase()
    # motor = double_layer_planar_motor()
    # motor = dlpm_2022()
    # motor = copam_2009()
    # motor = four_phase_linear()
    # motor = concentrated_5_phase()
    motor = ideal_4_phase()
    analyze(motor)
    sys.exit(0)

    frame = get_5cell_clifford_vertices() #generate_minimal_8_frame(4) #compute_simplex_frame(5) #generate_circle_points(3)
    S = frame @ frame.T

    print("Minimal Frame Matrix:")
    print(np.round(frame, 3))
    print("\nFrame Operator (S = FF^T):")
    print(np.round(S, 10))

    # The frame is tight if S is a multiple of the Identity matrix
    is_tight = np.allclose(S, np.eye(4) * S[0, 0])
    print(f"\nIs it a tight frame? {is_tight}")
    print(f"Frame Bound (A): {S[0, 0]}")

    motor_phases = frame

    print(motor_phases)

    area = motor_phases.shape[1]

    low, avg, high = power_range(motor_phases)
    print(f"{area}-phase motor requires a power between {low*area:.3f}x and {high*area:.3f}x (avg: {avg*area:.3f}x)")

    small_angle, large_angle = angle_range(motor_phases)

    print(f"Angles between vectors range from {np.rad2deg(small_angle):.1f}° to {np.rad2deg(large_angle):.1f}°")
    if np.isclose(small_angle, large_angle):
        print("Phases are equiangular!")

    norms = np.linalg.norm(motor_phases, axis=1)
    if np.allclose(norms, norms[0]):
        print("Phases have equal magnitude!")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
