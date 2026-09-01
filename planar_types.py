import numpy as np
from typing import NamedTuple

class Phase(NamedTuple):
    theta: float
    phi: float
    z_rotation: bool

class Coil(NamedTuple):
    j_x: np.ndarray
    j_y: np.ndarray
    scaling: float
    mask: np.ndarray = None

class Motor(NamedTuple):
    coil: Coil
    phases: list[Phase]
