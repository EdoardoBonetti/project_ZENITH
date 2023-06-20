from ngsolve import *
from ngsolve import Einsum

import sys
from sys import path
sys.path.append("/home/ebonetti/Desktop/project_ZENITH")


from zenith import *
from zenith.utils.CompactObjects import BlackHole
from zenith.utils.Geometries import DefaultMesh

def main():
    
    # breate the black hole, can be done with the following command line options:
    mass1 = 1
    spin1 =(0,0,0)
    pos1 = (2,0,0)
    mom1 = (-1,0,0)

    mass2 = 1
    spin2 =(0,0,0)
    pos2 = (-2,0,0)
    mom2 = (1,0,0)

    for i in range(len(sys.argv)):
        if sys.argv[i] == "-mom1":
            mass1 = float(sys.argv[i+1])
        if sys.argv[i] == "-spin1":
            spin1 = (float(sys.argv[i+1]),float(sys.argv[i+2]),float(sys.argv[i+3]))
        if sys.argv[i] == "-pos1":
            pos1 = (float(sys.argv[i+1]),float(sys.argv[i+2]),float(sys.argv[i+3]))
        if sys.argv[i] == "-mom1":
            mom1 = (float(sys.argv[i+1]),float(sys.argv[i+2]),float(sys.argv[i+3]))

        if sys.argv[i] == "-mom2":
            mass2 = float(sys.argv[i+1])
        if sys.argv[i] == "-spin2":
            spin2 = (float(sys.argv[i+1]),float(sys.argv[i+2]),float(sys.argv[i+3]))
        if sys.argv[i] == "-pos2":
            pos2 = (float(sys.argv[i+1]),float(sys.argv[i+2]),float(sys.argv[i+3]))
        if sys.argv[i] == "-mom2":
            mom2 = (float(sys.argv[i+1]),float(sys.argv[i+2]),float(sys.argv[i+3]))


    # create the mesh, can be done with the following command line options:
    h = 0.2
    R = 10
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
        

    blackhole1 = BlackHole(mass = mass1, pos = pos1, mom = mom1, spin = spin1)
    blackhole2 = BlackHole(mass = mass2, pos = pos2, mom = mom2, spin = spin2)

    blackholes = [blackhole1, blackhole2]
    kwargs["blackholes"] = blackholes
    
    mesh = DefaultMesh( h = h, R = R, **kwargs)


    bowenyork = BowenYork(blackholes, **kwargs)
    
    # now we solve it
    order = 2
    for i in range(len(sys.argv)):   
        if sys.argv[i] == "-order":
            order = int(sys.argv[i+1])

    kwargs["order"] = order

    h1 = H1(mesh, order = order)
    bowenyork.Solve(h1)
    #bowenyork.Draw(mesh)

    gf_h = GridFunction(VectorValued(h1,9))
    gf_Aij = GridFunction(VectorValued(h1,9))
    gf_W = GridFunction(h1)
    help(gf_Aij.Einsum)

    kwargs["bonus_intorder"] = 10
    bowenyork.GetSolution(gf_h, gf_Aij, gf_W)

    Draw(gf_h, mesh, "h")
    Draw(gf_Aij, mesh, "Aij")
    Draw(gf_W, mesh, "W")


    #print((gf_Aij.Reshape((3,3))))




    


if __name__ == "__main__":
    
    with TaskManager() : main()