# here I test the initial condition for a single black 
# hole with linear momentum

from ngsolve import *
from zenith import *

# create single black hole
mass = CF((1))
pos = CF((0,0,0))
mom = CF((0,0,1))
spin = CF((0,0,0))
BH = BlackHole(mass = mass, pos = pos, mom = mom, spin = spin)
