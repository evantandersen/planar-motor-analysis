import numpy as np

from analysis import Motor, Coil, Phase
from coils.racetrack import racetrack_coil


def double_layer_planar_motor():
    # coil parameters (in mm). Pole pitch divided by sqrt(2) as they use a different coordinate system
    pole_pitch = 33.3 / np.sqrt(2)
    coil_width = 31.4
    bundle_width = 12.6
    # in the paper the 140.8 is not the total length of the coil, but the distance to the midpoint of the copper in each endturn
    # here it is the total length of the coil, so it can be computed easily:
    coil_length = 140.8 + bundle_width - 1 * pole_pitch

    coil = racetrack_coil(pole_pitch, coil_width, bundle_width, coil_length)

    # phases = [
    #     #y forcers
    #     Phase(theta=np.pi/2, phi=-2*np.pi/3, z_rotation=False),
    #     Phase(theta=-np.pi/2, phi=2 * np.pi / 3, z_rotation=False),
    #     Phase(theta=-np.pi/2, phi=-6 * np.pi / 3, z_rotation=False),
    #     # Phase(theta=np.pi/2, phi=6 * np.pi / 3, z_rotation=False),
    #     # Phase(theta=np.pi/2, phi=-10 * np.pi / 3, z_rotation=False),
    #     # Phase(theta=-np.pi/2, phi=10 * np.pi / 3, z_rotation=False),
    #
    #     #x forcers
    #     Phase(theta=-2*np.pi/3, phi=np.pi/2, z_rotation=True),
    #     Phase(theta=2 * np.pi / 3, phi=-np.pi/2, z_rotation=True),
    #     Phase(theta=-6 * np.pi / 3, phi=-np.pi/2, z_rotation=True),
    #     # Phase(theta=6 * np.pi / 3, phi=np.pi/2, z_rotation=True),
    #     # Phase(theta=-10 * np.pi / 3, phi=np.pi/2, z_rotation=True),
    #     # Phase(theta=10 * np.pi / 3, phi=-np.pi/2, z_rotation=True),
    # ]

    phases = [
        #y forcers
        Phase(theta=0, phi=-4*np.pi/3, z_rotation=False),
        Phase(theta=0, phi=0, z_rotation=False),
        Phase(theta=0, phi=4*np.pi/3, z_rotation=False),

        #x forcers
        Phase(theta=-4*np.pi/3, phi=0, z_rotation=True),
        Phase(theta=0, phi=0, z_rotation=True),
        Phase(theta=4*np.pi/3, phi=0, z_rotation=True),
    ]


    return Motor(coil=coil, phases=phases)

