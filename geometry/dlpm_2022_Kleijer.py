import numpy as np

from analysis import Motor, Phase
from geometry.racetrack import racetrack_coil

# Optimization of Quasi-Halbach Topologies to Maximize the Acceleration of Moving-Magnet Planar Motors
# M. Kleijer; J. W. Jansen; E. A. Lomonova
# 2022

# DOI: 10.1109/ICEM51905.2022.9910902

def dlpm_2022():
    # coil parameters (in mm). Pole pitch divided by sqrt(2) as they use a different coordinate system
    pole_pitch = 33.3 / np.sqrt(2)
    coil_width = 31.4
    bundle_width = 14.6
    # in the paper the 47.1 is not the total length of the coil, but the distance to the midpoint of the copper in each endturn
    # here it is total length of the coil, so we can compute it easily:
    coil_length = 47.1 + bundle_width

    coil = racetrack_coil(pole_pitch, coil_width, bundle_width, coil_length, span=2)
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

