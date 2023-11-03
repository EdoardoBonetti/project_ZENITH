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

def Spy(m, n = 8, save = False, **kwargs):
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    name = kwargs.get('name', 'spy')
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (4, 4)) , dpi=kwargs.get('dpi', 300) )
    plt.jet()
    colors = mpl.cm.jet(np.linspace(0,1,n))
    for i, color in enumerate(colors):
        plt.spy(m, markersize=0.1, precision=i/n, color = color , alpha = 1 - i/n)

    if save:
        plt.savefig(name + '.png', dpi=kwargs.get('dpi', 300) )
    plt.show()

    