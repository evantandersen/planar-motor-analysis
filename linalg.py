import numpy as np


def rotate_vector(x, y, radians):
    # Calculate cosine and sine
    cos_theta = np.cos(radians)
    sin_theta = np.sin(radians)

    # Apply the rotation matrix formulas
    new_x = x * cos_theta - y * sin_theta
    new_y = x * sin_theta + y * cos_theta

    return new_x, new_y

def compute_simplex_frame(n):
    if n <= 1:
        raise ValueError("Simplex must have more than 1 vertex")

    # 1. Create the all-ones vector and normalize it
    ones_vec = np.ones((n, 1))
    unit_ones = ones_vec / np.linalg.norm(ones_vec)

    # 2. Create a full rank matrix with the all-ones vector as the first column
    # Use an identity matrix to provide the remaining directions
    identity = np.eye(n)
    matrix = np.hstack([unit_ones, identity])

    # 3. Use QR decomposition to find an orthonormal basis for the whole space
    # The first column of Q will be our unit_ones vector.
    # The remaining columns will be orthogonal to it.
    q, r = np.linalg.qr(matrix)

    # 4. The nullspace basis consists of columns 1 through n-1
    # (Excluding column 0, which is the all-ones direction)
    null_basis = q[:, 1:n]

    #normalize to unit length and return as column vectors
    return np.sqrt(n/(n-1)) * null_basis.T


def power_range(phases):
    """
    Calculates the lower bound, average, and upper bounds of the power required to span all vectors in the space

    Parameters:
    phases (np.ndarray): A matrix where each column is a frame vector representing a physical phase

    Returns:
    tuple: (lower_bound, avg, upper_bound)
    """
    # Compute the frame operator S = A * A.H
    # (A.H is the conjugate transpose)
    frame_operator = np.dot(phases, phases.conj().T)

    # Step 2: Compute the eigenvalues of the Hermitian frame operator
    eigenvalues = np.linalg.eigvalsh(frame_operator)

    # The frame vectors describe the location of flux produced by the coils
    # Therefore the current required is the inverse of this frame,
    # which is called the canonical dual frame
    eigen_inv = 1/eigenvalues

    # the min, avg, and max power required comes directly from the inverted eigenvalues
    return np.min(eigen_inv), np.mean(eigen_inv), np.max(eigen_inv)

def angle_range(vectors):
    n = vectors.shape[1]
    n_angles = (n*(n-1))/2
    angles = np.empty(int(n_angles))
    k = 0
    for i in range(n):
        for j in range(n-(i+1)):
            a = vectors[:, i]
            b = vectors[:, j+i+1]
            num = a.T @ b
            demon = np.linalg.norm(a) * np.linalg.norm(b)
            angles[k] = np.arccos(num/demon)
            k += 1

    return np.min(angles), np.max(angles)


def find_minimal_energy_coefficients(F, x):
    """
    Finds the minimal L2-energy coefficients to represent vector x
    using the frame matrix F.

    Parameters:
    F (numpy.ndarray): An (N, M) matrix where columns are the frame vectors (M > N).
    x (numpy.ndarray): An (N, 1) target column vector.

    Returns:
    c (numpy.ndarray): An (M, 1) column vector of optimal coefficients.
    """
    # Ensure inputs are float arrays for numerical stability
    F = np.asarray(F, dtype=float)
    x = np.asarray(x, dtype=float)

    # Compute the Moore-Penrose pseudoinverse of F
    # F_pinv will have a shape of (M, N)
    F_pinv = np.linalg.pinv(F)

    # Calculate the minimal energy coefficients: c = F^\dagger * x
    c = np.dot(F_pinv, x)

    return c