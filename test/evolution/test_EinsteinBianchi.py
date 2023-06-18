import sys
from sys import path
import os
currentpath = os.getcwd()
# take the current path string and take away the last folder until folder "Project_ZENITH" is found
while currentpath.split("\\")[-1] != "project_ZENITH":
    currentpath = "\\".join(currentpath.split("\\")[:-1])

# add the path to the sys.path if it does not exist
if currentpath not in sys.path: sys.path.append(currentpath)


from matplotlib import pyplot as plt

from ngsolve import *

from zenith import *
from zenith.utils.CompactObjects import BlackHole
from zenith.utils.Geometries import DefaultMesh

def main():
        
        #mesh= Mesh(unit_cube.GenerateMesh(maxh=0.3))
        h = 0.3
        mesh = DefaultMesh(h=h, R=1, small_rad=0.6)

        t = 0
        tend = 0.1
        dt = 0.01

        eb = EinsteinBianchi(mesh, order=2, nonassemble=True)
        eb.SetInitialCondition()


        while t < tend:
            t += dt
            eb.TimeStep(dt)
            eb.TrachEnergy()
            print("t : " , int(t*100/tend), "%", end="\r")

        figname = "dt_" + str(dt) + "_tend" + str(tend) + "_h" + str(h)
        eb.PlotEnergy(save=True, figname=figname)
   


if __name__ == "__main__":
    #SetNumThreads(8)
    with TaskManager(): main()