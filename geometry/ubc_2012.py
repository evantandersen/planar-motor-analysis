import numpy as np

from analysis import Coil, Phase, Motor


# 6D direct-drive technology for planar motion stages
# Xiaodong Lu, Irfan-ur-rab Usman
# 2012

# DOI: 10.1016/j.cirp.2012.03.145

N = 512
width = 1/6

def ubc_2012():

    Jx = np.full((int(N*width), N), 1)
    Jy = np.zeros_like(Jx)

    coil = Coil(Jx, Jy, N)

    phases = [
        #y forcers
        Phase(theta=0, phi=-1/3, z_rotation=False),
        Phase(theta=0, phi=0, z_rotation=False),
        Phase(theta=0, phi=1/3, z_rotation=False),

        #x forcers
        Phase(theta=-1/3, phi=0, z_rotation=True),
        Phase(theta=0, phi=0, z_rotation=True),
        Phase(theta=1/3, phi=0, z_rotation=True),
    ]

    return Motor(coil=coil, phases=phases)

