import numpy as np

from analysis import Coil, Phase, Motor

n = 1000
x = np.linspace(0, 1, n, endpoint=False)
y = np.linspace(0, 1, n, endpoint=False)

xv, yv = np.meshgrid(x, y)

npoles = 2
threshold = 0.87

def simple_test_dist():
    p_x = (xv) * npoles * np.pi
    p_y = (yv) * npoles * np.pi

    X = np.cos(p_y)
    Y = np.cos(p_x)
    # Y = np.zeros_like(X) #np.sin(p_x)
    # X = np.where(abs(np.cos(p_y)) > threshold, np.cos(p_y), 0)
    # & (2*np.pi <= p_x) & (p_x < 4*np.pi)
    # Y = np.zeros_like(X) #np.where((abs(np.cos(p_x)) > threshold), np.cos(p_x), 0)
    return X, Y

def ideal_4_phase():
    j_x, j_y = simple_test_dist()
    count = ((j_x != 0) | (j_y != 0)).sum()
    area = count / n**2
    print(f"area is {area}")
    #
    # coil = Coil(j_x=j_x, j_y=j_y, span=int(npoles/2), area=area)

    phases = [
        Phase(theta=0, phi=0, z_rotation=False),
        Phase(theta=0, phi=np.pi, z_rotation=False),
        Phase(theta=np.pi/2, phi=np.pi/2, z_rotation=False),
        Phase(theta=-np.pi/2, phi=np.pi/2, z_rotation=False),
    ]

    p_y = (yv) * npoles * np.pi


    # j_x = np.sin(p_y)
    # j_y = np.zeros_like(j_x)

    coil = Coil(j_x=j_x, j_y=j_y, span=int(npoles/2), area=area)

    # phases = [
    #     Phase(theta=0, phi=0, z_rotation=False),
    #     Phase(theta=0, phi=np.pi/2, z_rotation=False),
    #     Phase(theta=0, phi=0, z_rotation=True),
    #     Phase(theta=np.pi/2, phi=0, z_rotation=True),
    # ]

    # phases = [
    #     Phase(theta=0, phi=0, z_rotation=False),
    #     Phase(theta=0, phi=np.deg2rad(120), z_rotation=False),
    #     Phase(theta=0, phi=np.deg2rad(240), z_rotation=False),
    #
    #     Phase(theta=0, phi=0, z_rotation=True),
    #     Phase(theta=np.deg2rad(120), phi=0, z_rotation=True),
    #     Phase(theta=np.deg2rad(240), phi=0, z_rotation=True),
    # ]


    return Motor(coil=coil, phases=phases)
