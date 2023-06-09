from ngsolve.internal import visoptions
from ngsolve.internal import viewoptions


def VisualOptions():
    visoptions.scalfunction='u:0'
    visoptions.clipsolution = 'scal'
    viewoptions.clipping.nx= 0
    viewoptions.clipping.ny= 0
    viewoptions.clipping.nz= -1
    viewoptions.clipping.enable = 1
    return None

    