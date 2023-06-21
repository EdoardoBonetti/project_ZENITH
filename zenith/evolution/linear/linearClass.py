# in this file we define the einstein bianchi equations

from ngsolve import *
from netgen.csg import *    
import matplotlib.pyplot as plt
from ngsolve.la import EigenValues_Preconditioner
import numpy as np


from netgen.csg import *


from ngsolve.internal import visoptions
from ngsolve.internal import viewoptions

visoptions.scalfunction='u:0'
visoptions.clipsolution = 'scal'
viewoptions.clipping.nx= 0
viewoptions.clipping.ny= 0
viewoptions.clipping.nz= -1
viewoptions.clipping.enable = 1



