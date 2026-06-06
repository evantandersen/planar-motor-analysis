import numpy as np

from analysis import Motor, Phase
from geometry.racetrack import racetrack_coil

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
    coil_length = 140.8 + bundle_width # - 1 * pole_pitch

    x_offset = pole_pitch * (6 + 2/3)
    y_offset = pole_pitch * 4
    offsets = []
    for y_off in [-0.5, 0.5]:
        # for xoff in [-1, 0, 1]:
        #     x_jog = 0
        #     if y_off % 2 == 0:
        #         x_jog = pole_pitch
            offsets.append((0, y_off*y_offset))
    coil = racetrack_coil(pole_pitch, coil_width, bundle_width, coil_length, 4, n_per_pole=50, offsets=offsets)

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

