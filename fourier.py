import numpy as np



def get_components(j_x, j_y, span):

    x_spectrum = np.fft.fftshift(np.fft.fftn(j_x))
    y_spectrum = np.fft.fftshift(np.fft.fftn(j_y))

    n=j_x.shape[0]
    dc = int(n / 2)

    result = np.empty(4)
    result[0] = x_spectrum[dc][dc + span].real
    result[1] = x_spectrum[dc][dc + span].imag
    result[2] = y_spectrum[dc + span][dc].real
    result[3] = y_spectrum[dc + span][dc].imag

    return result / n**2


# def compute_streamfunction_generic(fft_Jx, fft_Jy):
#     """
#     Computes the 2D FFT of the streamfunction (psi) using normalized/generic grids.
#
#     Parameters:
#     -----------
#     fft_Jx, fft_Jy : ndarray
#         2D complex arrays of the current density Fourier coefficients.
#     """
#     Ny, Nx = fft_Jx.shape
#
#     # Create generic, unit-less frequency indices matching standard FFT layout
#     # (0 to N/2 for positive frequencies, -N/2 to -1 for negative frequencies)
#     kx = np.fft.fftfreq(Nx)
#     ky = np.fft.fftfreq(Ny)
#
#     # Broadcast to 2D grids
#     Kx, Ky = np.meshgrid(kx, ky)
#
#     # Compute denominator
#     K_squared = Kx ** 2 + Ky ** 2
#
#     # Invert the curl: psi = (-i*kx*Jy + i*ky*Jx) / (kx^2 + ky^2)
#     with np.errstate(divide='ignore', invalid='ignore'):
#         fft_psi = (-1j * Kx * fft_Jy + 1j * Ky * fft_Jx) / K_squared
#
#     # Handle the DC component / singular points safely
#     fft_psi[K_squared == 0] = 0.0
#
#     return fft_psi

def compute_streamfunction(fft_Jx_shifted, fft_Jy_shifted):
    """
    Computes the 2D FFT of the streamfunction (psi) from ALREADY SHIFTED
    current density spectra.
    """
    Ny, Nx = fft_Jx_shifted.shape

    # 1. Generate standard frequencies
    kx = np.fft.fftfreq(Nx)
    ky = np.fft.fftfreq(Ny)

    # 2. Shift the frequencies to match the shifted input spectra!
    kx_shifted = np.fft.fftshift(kx)
    ky_shifted = np.fft.fftshift(ky)

    # 3. Create the 2D grids
    Kx, Ky = np.meshgrid(kx_shifted, ky_shifted)

    # 4. Math remains identical
    K_squared = Kx ** 2 + Ky ** 2

    with np.errstate(divide='ignore', invalid='ignore'):
        fft_psi_shifted = (-1j * Kx * fft_Jy_shifted + 1j * Ky * fft_Jx_shifted) / K_squared

    # The DC component is now in the center of the array instead of [0,0]
    fft_psi_shifted[K_squared == 0] = 0.0

    return fft_psi_shifted