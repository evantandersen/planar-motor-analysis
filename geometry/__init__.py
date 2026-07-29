"""Geometry module for defining motor and coil configurations."""

from geometry.concentrated_5_phase import concentrated_5_phase
from geometry.copam import copam_2009
from geometry.dlpm_2013 import double_layer_planar_motor
from geometry.dlpm_2022_Kleijer import dlpm_2022
from geometry.ideal_distributed import ideal_4_phase, ideal_5_phase
from geometry.ubc_2012 import ubc_2012

__all__ = [
    "concentrated_5_phase",
    "copam_2009",
    "double_layer_planar_motor",
    "dlpm_2022",
    "ideal_4_phase",
    "ideal_5_phase",
    "ubc_2012"
]
