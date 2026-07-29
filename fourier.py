import numpy as np
from planar_types import Coil


def validate_current_density(x_spectrum, y_spectrum, Kx, Ky, tolerance=1e-5):
    """
    Validates current density entirely in the frequency domain.
    Checks for:
    1. A non-zero DC component (extracted from the [0,0] FFT coefficient).
    2. Divergence (div J != 0).
    """
    # Total number of elements to normalize the DC component
    num_elements = x_spectrum.size

    # 1. Check for DC Component via the [0,0] frequency bin
    dc_x = np.abs(x_spectrum[0, 0]) / num_elements
    dc_y = np.abs(y_spectrum[0, 0]) / num_elements

    if dc_x > tolerance or dc_y > tolerance:
        print(f"[Warning] Non-zero DC component detected! Normalized |DC_x|: {dc_x:.2e}, |DC_y|: {dc_y:.2e}")

    # 2. Check for Divergence (div J = i*Kx*Jx_fft + i*Ky*Jy_fft)
    divergence_fft = 1j * Kx * x_spectrum + 1j * Ky * y_spectrum
    max_div = np.max(np.abs(divergence_fft)) / num_elements

    if max_div > tolerance:
        print(f"[Warning] Significant divergence detected in current density! Max |div(J)|: {max_div:.2e}")


def compute_streamfunction(Jx, Jy):
    """
    Computes the 2D FFT of the streamfunction (psi) from a 2D current density grid
    """

    # fourier transform the input current vector components
    x_spectrum = np.fft.fftn(Jx)
    y_spectrum = np.fft.fftn(Jy)

    # create a grid of wave numbers
    Ny, Nx = Jx.shape
    kx = np.fft.fftfreq(Nx) * Nx
    ky = np.fft.fftfreq(Ny) * Ny
    Kx, Ky = np.meshgrid(kx, ky)

    # check for non-zero divergence or DC component
    validate_current_density(x_spectrum, y_spectrum, Kx, Ky, tolerance=0.05)

    # compute psi
    K_squared = Kx ** 2 + Ky ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        psi_fft = (-1j * Kx * y_spectrum + 1j * Ky * x_spectrum) / K_squared

    # shift the frequency spectrumm so it's centered
    psi_fft_shifted = np.fft.fftshift(psi_fft)

    # The DC component is now in the center of the array instead of [0,0]
    dc = int(Nx / 2)

    # it's also a meaningless gauge (psi is relative), so set it to 0
    psi_fft_shifted[dc][dc] = 0.0

    return psi_fft_shifted


def compute_fundamental_components(coil: Coil) -> np.ndarray:
    """
    Computes the fundamental magnetic components [cos_x, sin_x, cos_y, sin_y]
    by direct integration of the Coil object's current densities.

    The motor wavelength is implicitly defined via `coil.scaling`
    (samples per wavelength).
    """
    ny, nx = coil.j_x.shape

    # 1D index arrays centered at origin
    mx = np.arange(nx) - (nx - 1) / 2.0
    my = np.arange(ny) - (ny - 1) / 2.0

    # Phase per grid sample for fundamental spatial wavenumber (k0 * x)
    # k0 = 2*pi / lambda,  x = mx * dx  =>  k0 * x = 2*pi * mx / scaling
    phase_x = 2.0 * np.pi * mx / coil.scaling
    phase_y = 2.0 * np.pi * my / coil.scaling

    exp_kx = np.exp(-1j * phase_x)  # (nx,)
    exp_ky = np.exp(-1j * phase_y)  # (ny,)

    # Normalized spatial integration element dx * dy / lambda^2 = 1 / scaling^2
    dA_normalized = 1.0 / (coil.scaling ** 2)

    # --- Integrate for kx component (k_x = k0, k_y = 0) ---
    # Jy(k0, 0) normalized by lambda^2
    Jy_k0_x = np.sum(coil.j_y @ exp_kx) * dA_normalized

    # Streamfunction component at (k0, 0): psi = -i * Jy / (k0 * lambda) = -i * Jy / (2*pi)
    psi_k0_x = -1j * Jy_k0_x

    # --- Integrate for ky component (k_x = 0, k_y = k0) ---
    # Jx(0, k0) normalized by lambda^2
    Jx_0_ky = np.sum(exp_ky @ coil.j_x) * dA_normalized

    # Streamfunction component at (0, k0): psi = i * Jx / (k0 * lambda) = i * Jx / (2*pi)
    psi_0_ky = 1j * Jx_0_ky

    # --- Extract fundamental force components ---
    cos_x = 2.0 * psi_k0_x.imag
    sin_x = 2.0 * psi_k0_x.real

    cos_y = -2.0 * psi_0_ky.imag
    sin_y = -2.0 * psi_0_ky.real

    return np.array([cos_x, sin_x, cos_y, sin_y])


def generate_magnet_spectrum(Nx, Ny, beta):
    """
    Directly injects the 45-degree rotated magnet single tones into specific
    pixel coordinates of the frequency spectra.
    """
    fft_Bx = np.zeros((Ny, Nx), dtype=complex)
    fft_By = np.zeros((Ny, Nx), dtype=complex)
    fft_Bz = np.zeros((Ny, Nx), dtype=complex)

    amp = 1.0 / (2.0 * np.sqrt(2))

    # Component 1: Plane wave traveling along X-axis (Ky = 0)
    fft_Bz[0, beta] = amp
    fft_Bx[0, beta] = -1j * amp
    fft_Bz[0, Nx - beta] = amp
    fft_Bx[0, Nx - beta] = 1j * amp

    # Component 2: Plane wave traveling along Y-axis (Kx = 0)
    fft_Bz[beta, 0] = amp
    fft_By[beta, 0] = -1j * amp
    fft_Bz[Ny - beta, 0] = amp
    fft_By[Ny - beta, 0] = 1j * amp

    return fft_Bx, fft_By, fft_Bz