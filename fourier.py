import numpy as np


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