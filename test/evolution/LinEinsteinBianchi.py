
import os
import sys
from sys import path
currentpath = os.getcwd()
# take the current path string and take away the last folder until folder "Project_ZENITH" is found
#while currentpath.split("//")[-1] != "project_ZENITH":
currentpath = "C:/Users/User/OneDrive/Desktop/project_ZENITH/"

# add the path to the sys.path if it does not exist
if currentpath not in sys.path: sys.path.append(currentpath)


from netgen.csg import *

from ngsolve import *
from zenith import *



def main():

    kwargs = {"inverse" : "sparsecholesky",
              "nonassemble" : False, 
              "order" : 1}

    cube = OrthoBrick( Pnt(0,0,0), Pnt(1,1,1) )
    geo = CSGeometry()
    geo.Add (cube)
    mesh = Mesh(geo.GenerateMesh(maxh=0.3) )

    
    eb = LinEinsteinBianchi(mesh, order = 2, **kwargs)

    r = sqrt(x*x+y*y+z*z)
    cf_f = exp(-r*r)
    
    initial_E = cf_init
    initial_B = cf_init

    # set initial values
    print("Setting initial values")
    eb.SetInitialValues(cf_E=initial_E, cf_B=initial_B)
    print("Plotting")
    eb.Plot()
    print("Done")
    input("Press Enter to continue...")
    for i in range(100):
        eb.Step(0.005)
        eb.Plot()


if __name__ == "__main__":
    with TaskManager(): Test()

