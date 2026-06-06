import numpy as np

from analysis import Coil, Phase, Motor

n = 1000
x = np.linspace(0, 1, n, endpoint=False)
y = np.linspace(0, 1, n, endpoint=False)

xv, yv = np.meshgrid(x, y)

npoles = 2

threshold = -0.1#0.84


def threshold_dist():
    p_x = (xv - 0.5) * npoles * np.pi
    p_y = (yv - 0.5) * npoles * np.pi

    X = np.where(abs(np.cos(p_y)) > threshold, np.cos(p_y), 0)
    # Y = np.zeros(X.shape)
    Y = np.where(abs(np.cos(p_x)) > threshold, np.cos(p_x), 0)
    # X = np.zeros_like(Y)

    # norm = np.sqrt(X**2 + Y**2)
    # np.divide(X, norm, out=X, where=norm != 0)
    # np.divide(Y, norm, out=Y, where=norm != 0)
    return X, Y

def concentrated_5_phase():
    j_x, j_y = threshold_dist()

    count = ((j_x != 0) | (j_y != 0)).sum()
    area = count / n**2
    print(f"area = {area:.3f}")
    coil = Coil(j_x=j_x, j_y=j_y, span=int(npoles/2), area=area)
    # phases = [
    #     Phase(theta=np.deg2rad(-180), phi=np.deg2rad(180), z_rotation=False),
    #     Phase(theta=np.deg2rad(-108), phi=np.deg2rad(36), z_rotation=False),
    #     Phase(theta=np.deg2rad(36), phi=np.deg2rad(108), z_rotation=False),
    #     Phase(theta=np.deg2rad(-36), phi=np.deg2rad(-108), z_rotation=False),
    #     Phase(theta=np.deg2rad(108), phi=np.deg2rad(-36), z_rotation=False),
    # ]

    phases = [
        Phase(theta=np.deg2rad(0), phi=np.deg2rad(0), z_rotation=False),
        Phase(theta=np.deg2rad(0), phi=np.deg2rad(120), z_rotation=False),
        Phase(theta=np.deg2rad(0), phi=np.deg2rad(240), z_rotation=False),

        Phase(theta=np.deg2rad(120), phi=np.deg2rad(0), z_rotation=False),
        Phase(theta=np.deg2rad(120), phi=np.deg2rad(120), z_rotation=False),
        Phase(theta=np.deg2rad(120), phi=np.deg2rad(240), z_rotation=False),

        Phase(theta=np.deg2rad(240), phi=np.deg2rad(0), z_rotation=False),
        Phase(theta=np.deg2rad(240), phi=np.deg2rad(120), z_rotation=False),
        Phase(theta=np.deg2rad(240), phi=np.deg2rad(240), z_rotation=False),
    ]


    return Motor(coil=coil, phases=phases)
