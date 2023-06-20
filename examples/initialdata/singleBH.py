
from ngsolve import *

import sys
from sys import path
sys.path.append("/home/ebonetti/Desktop/project_ZENITH")


from zenith import *
from zenith.utils.CompactObjects import BlackHole
from zenith.utils.Geometries import DefaultMesh
from zenith.initialdata.BowenYork_puncture import BowenYork

def main(argv=None):

    # breate the black hole, can be done with the following command line options:
    mass = 1
    spin =(0.5,0,0)
    pos = (0,0,0)
    mom = (0,0,0)

    for i in range(len(sys.argv)):
        if sys.argv[i] == "-mom":
            mass = float(sys.argv[i+1])
        if sys.argv[i] == "-spin":
            spin = (float(sys.argv[i+1]),float(sys.argv[i+2]),float(sys.argv[i+3]) )
        if sys.argv[i] == "-pos":
            pos = (float(sys.argv[i+1]),float(sys.argv[i+2]),float(sys.argv[i+3]) )
        if sys.argv[i] == "-mom":
            mom = (float(sys.argv[i+1]),float(sys.argv[i+2]),float(sys.argv[i+3]) )

    # create the mesh, can be done with the following command line options:
    h = 0.2
    R = 6
    for i in range(len(sys.argv)):
        if sys.argv[i] == "-h":
            h = int(sys.argv[i+1])
        if sys.argv[i] == "-R":
            R = float(sys.argv[i+1])
    # we can add some kwargs to the mesh
    kwargs = {}
    for i in range(len(sys.argv)):
        if sys.argv[i] == "-grading":
            kwargs["grading"] = float(sys.argv[i+1])
        if sys.argv[i] == "-adapt":
            kwargs["adaption"] = True
        

    blackhole = BlackHole(mass = mass, pos = pos, mom = mom, spin = spin)

    kwargs["blackholes"] = [blackhole]
    mesh = DefaultMesh( h = h, R = R, **kwargs)
    # blackhole.Draw(mesh)
    blackholes = [blackhole]
    # now we want to solve the Bowen York initial data problem


    # h = CF(( 1,3/2,3/2, 3/2,1,3/2, 3/2,3/2,1))
    # kwargs["conformal_metric"] = h
    bowenyork = BowenYork(blackholes, **kwargs)
    
    # not yet solved
    # bowenyork.Draw(mesh)

    # now we solve it
    order = 2
    for i in range(len(sys.argv)):   
        if sys.argv[i] == "-order":
            order = int(sys.argv[i+1])

    kwargs["order"] = order

    h1 = H1(mesh, order = order)
    bowenyork.Solve(h1)
    bowenyork.Draw(mesh)


if __name__ == "__main__":
    
    with TaskManager() : main()