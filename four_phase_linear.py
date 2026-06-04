import numpy as np

from analysis import Motor, Phase, Coil
from coils.racetrack import racetrack_coil


def four_phase_linear():
    # coil parameters (in mm). Pole pitch divided by sqrt(2) as they use a different coordinate system
    pole_pitch = 33.3 / np.sqrt(2)
    coil_width = pole_pitch
    bundle_width = pole_pitch/2

    #using the same value as rovers 2015 to minimize force in off-axis
    coil_length = (140.8 + bundle_width - 1 * pole_pitch)

    coil = racetrack_coil(pole_pitch, coil_width, bundle_width, coil_length)
    coil = Coil(j_x=coil.j_x, j_y=coil.j_y, span=coil.span, area=coil.area/2)
    # phases = [
    #     #y forcers
    #     Phase(theta=0, phi=-np.pi/4, z_rotation=False),
    #     Phase(theta=0, phi=np.pi/4, z_rotation=False),
    #
    #     #x forcers
    #     Phase(theta=-np.pi/4, phi=0, z_rotation=True),
    #     Phase(theta=np.pi/4, phi=0, z_rotation=True),
    # ]

    phases = [
        # y forcers
        Phase(theta=0, phi=-2 * np.pi / 3, z_rotation=False),
        Phase(theta=0, phi=0, z_rotation=False),
        Phase(theta=0, phi=2 * np.pi / 3, z_rotation=False),

        # x forcers
        Phase(theta=-2 * np.pi / 3, phi=0, z_rotation=True),
        Phase(theta=0, phi=0, z_rotation=True),
        Phase(theta=2 * np.pi / 3, phi=0, z_rotation=True),
    ]

    return Motor(coil=coil, phases=phases)
