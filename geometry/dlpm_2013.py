import numpy as np

from analysis import Motor, Phase
from geometry.racetrack import generate_racetrack_coil

# Design and measurements of the Double Layer Planar Motor
# J.M.M. Rovers; J.W. Jansen; E.A. Lomonova
# 2013

# DOI: 10.1109/IEMDC.2013.6556254

def double_layer_planar_motor():
    # coil parameters (in mm). Pole pitch divided by sqrt(2) as they use a different coordinate system
    pole_pitch = 33.3 / np.sqrt(2)
    coil_width = 31.4
    bundle_width = 12.6
    # in the paper the 140.8 is not the total length of the coil, but the distance to the midpoint of the copper in each endturn
    # here it is the total length of the coil, so it can be computed easily:
    coil_length = 140.8 + bundle_width

    coil = generate_racetrack_coil(pole_pitch, coil_width, bundle_width, coil_length)

    phases = [
        #y forcers
        Phase(theta=0, phi=-2/3, z_rotation=False),
        Phase(theta=0, phi=0, z_rotation=False),
        Phase(theta=0, phi=2/3, z_rotation=False),

        #x forcers
        Phase(theta=-2/3, phi=0, z_rotation=True),
        Phase(theta=0, phi=0, z_rotation=True),
        Phase(theta=2/3, phi=0, z_rotation=True),
    ]

    return Motor(coil=coil, phases=phases)

