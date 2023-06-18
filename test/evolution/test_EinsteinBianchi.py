import sys
from sys import path
import os
currentpath = os.getcwd()
# take the current path string and take away the last folder until folder "Project_ZENITH" is found
while currentpath.split("\\")[-1] != "project_ZENITH":
    currentpath = "\\".join(currentpath.split("\\")[:-1])

# add the path to the sys.path if it does not exist
if currentpath not in sys.path: sys.path.append(currentpath)


from ngsolve import *

from zenith import *
from zenith.utils.CompactObjects import BlackHole
from zenith.utils.Geometries import DefaultMesh

def main():
        
        mesh= Mesh(unit_cube.GenerateMesh(maxh=0.2))
        
        t = 0
        tend = 1
        dt = 0.01

        eb = EinsteinBianchi(mesh, order=2)
        eb.SetInitialCondition()
        
        Draw (eb.gfE, mesh, "E")
        input("Press any key...")
        #Draw (Norm(Norm(eb.gfB)), mesh, "B")
        #Draw (Norm(eb.gfv), mesh, "v")
        eb.TrachEnergy()

        while t < tend:
            t += dt
            eb.TimeStep(dt)
            eb.TrachEnergy()
            Draw (eb.gfE, mesh, "E")
            #Draw (Norm(eb.gfB), mesh, "B")
            #Draw (Norm(eb.gfv), mesh, "v")
            
            # print time in percent and the energies to the 4th decimal place
            print(" t:", int(100*t/tend) , "% " , "E: {:.4f} B: {:.4f} v: {:.4f}".format(eb.energyE[-1], eb.energyB[-1], eb.energyv[-1])
                   , "TraceE: {:.4f} SymB: {:.4f}".format(eb.energyTraceE[-1], eb.energySymB[-1]), end="\r")
        eb.PlotEnergy()

if __name__ == "__main__":
    with TaskManager(): main()