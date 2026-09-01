import numpy as np

from analysis import Coil, Phase, Motor


# 6D direct-drive technology for planar motion stages
# Xiaodong Lu, Irfan-ur-rab Usman
# 2012

# DOI: 10.1016/j.cirp.2012.03.145

N = 512
width = 1/6

def ubc_2012():

    Jx = np.zeros((N, N))
    Jy = np.zeros_like(Jx)

    w = int(N*width)
    half = N // 2
    Jx[:w, :] = 1
    Jx[half:half+w, :] = -1

    mask = np.where(Jx != 0, 1, 0)

    coil = Coil(Jx, Jy, N, mask)

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

