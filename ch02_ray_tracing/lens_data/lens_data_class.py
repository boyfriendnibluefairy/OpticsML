from dataclasses import dataclass
from typing import List


@dataclass
class Surface:
    """
    Single optical surface

    c  : curvature (1/mm)
    t  : thickness to next surface (mm)
    n1 : refractive index before the surface
    n2 : refractive index after the surface
    """
    c: float
    t: float
    n1: float
    n2: float


@dataclass
class LensSystem:
    """
    Ordered list of optical surfaces.
    """
    surfaces: List[Surface]
