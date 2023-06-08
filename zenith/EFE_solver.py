from .initialdata import BowenYork_puncture
from .initialdata import BSSN_puncture


from ngsolve import *


# create the class for the solver: it needs
# 1. the mesh
class EFE_Solver:

    def __init__(self,
                mesh : Mesh,
                bvp_configuration : BVPSolverConfiguration,
                ivp_configuration : IVPSolverConfiguration,
                **kwargs):
        pass

    