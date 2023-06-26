# create a test scenario for the bowen york  initial data using unittest

import unittest
import numpy as np
import os
import sys

# add the path to the zenith module
dirname = "C:\\Users\\User\\OneDrive\\Desktop\\project_ZENITH\\"
sys.path.append(dirname)

from ngsolve import *
from zenith import *

class TestBowenYorkAng(unittest.TestCase):
    # i need to create a resting black hole, the bowen york initial data will result in zero
    def setUp():
        # set up the bowen york initial data
        mass = 1.0
        pos = (0,0,0)
        mom = (0,0,0)
        ang = (1,0,0)

        r = CF((x**2 + y**2 + z**2)**0.5)

        bh = BlackHole(mass = mass, pos = pos, mom = mom, spin = ang)
        BHs = [bh]
        initialdata = BowenYork(BHs)

        # create the mesh
        mesh = DefaultMesh()
        h1 = H1(mesh, order=2, dirichlet="outer")
        gf_h = GridFunction(MatrixValued(h1,3))
        gf_Aij = GridFunction(MatrixValued(h1,3))
        gf_W = GridFunction(h1)

        initialdata.Solve(h1)
        initialdata.GetSolution(gf_h, gf_Aij, gf_W)

        def Inner(a, b):
            return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
        
        psi20 = -(1+mass/(2*r))**-5 *(5*(mass/(2*r))**3 +4*(mass/(2*r))**4 +  (mass/(2*r))**5 )
        psi22 = -0.1*(1+mass/(2*r))**-5 *((mass/(r))**3 )
        X = (x,y,z)
        P2 = (3*(Inner(ang, X))**2 - 1)/2
        P0 = 1
        psi2 = psi20*P0 + psi22*P2
        psi = 1 + mass/(2*r) + Inner(ang, ang)/mass**4 * psi2
        W = psi**-2

        Draw(gf_W, mesh, "W")
        Draw(W, mesh, "W_exact")
        error = gf_W - W
        Draw(error, mesh, "error")
        # set up the approx sol for 




class TestBowenYorkMom(unittest.TestCase):
    # i need to create a resting black hole, the bowen york initial data will result in zero
    def setUp():
        # set up the bowen york initial data
        mass = 1.0
        pos = (0,0,0)
        mom = (1,0,0)
        ang = (0,0,0)

        r = CF((x**2 + y**2 + z**2)**0.5)

        bh = BlackHole(mass = mass, pos = pos, mom = mom, spin = ang)
        BHs = [bh]
        initialdata = BowenYork(BHs)

        # create the mesh
        mesh = DefaultMesh()
        h1 = H1(mesh, order=2, dirichlet="outer")
        gf_h = GridFunction(MatrixValued(h1,3))
        gf_Aij = GridFunction(MatrixValued(h1,3))
        gf_W = GridFunction(h1)

        initialdata.Solve(h1)
        initialdata.GetSolution(gf_h, gf_Aij, gf_W)

        def Inner(a, b):
            return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
        
        psi20 = -(1+mass/(2*r))**-5 *(5*(mass/(2*r))**3 +4*(mass/(2*r))**4 +  (mass/(2*r))**5 )
        psi22 = -0.1*(1+mass/(2*r))**-5 *((mass/(r))**3 )
        X = (x,y,z)
        P2 = (3*(Inner(ang, X))**2 - 1)/2
        P0 = 1
        psi2 = psi20*P0 + psi22*P2
        psi = 1 + mass/(2*r) + Inner(ang, ang)/mass**4 * psi2
        W = psi**-2

        Draw(gf_W, mesh, "W")
        Draw(W, mesh, "W_exact")
        error = gf_W - W
        Draw(error, mesh, "error")
        # set up the approx sol for 

