import numpy as np

from analysis import Motor, Coil, Phase
from geometry.racetrack import generate_racetrack_coil

# Contactless Planar Actuator With Manipulator: A Motion System Without Cables and Physical
# Contact Between the Mover and the Fixed World
# Jeroen de Boeij; Elena Lomonova; Jorge Duarte
# 2008

# DOI: 10.1109/08IAS.2008.199

def copam_2009():
    # this program uses a coordinate system rotated 45 deg compared to the paper
    pole_pitch = 40 / np.sqrt(2)

    width = 40 * 3/2 #from the paper

    coil = generate_racetrack_coil(
        pole_pitch=pole_pitch,
        coil_width=width,
        bundle_width=width*0.45,    # don't think this was specified in the paper, visual estimate
        coil_length=width,
    )

    # offsets get a little messy due to aforementioned 45 deg coordinate rotation
    off = width/np.sqrt(2)
    phases = []
    for y_off in [-0.5, 0.5]:
        for x_off in [-0.5, 0.5]:
            t = x_off*off + y_off*off
            p = x_off*off - y_off*off
            t = (t/pole_pitch)/2
            p = (p/pole_pitch)/2
            phases.append(Phase(theta=t, phi=p, z_rotation=False))

    return Motor(coil=coil, phases=phases)
