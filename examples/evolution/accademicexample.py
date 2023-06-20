# i create an accademic example for the use of BSSN, in particular the formulation uses
# gf_X_c stands for the conformal transformation of gf_X
#   gamma_c : conformal metric , 3x3 symmetric
#   A_c : conformal traceless extrinsic curvature , 3x3 symmetric
#   W : square of inverse of conformal factor , scalar
#   K : trace of extrinsic curvature , scalar
#   Gamma_c : contracted conformal christoffel symbols ,  vector

# lets create the spaces as H1 and VectorValued(H1, ...)

from ngsolve import *

import os
import sys
from sys import path
currentpath = os.getcwd()
# take the current path string and take away the last folder until folder "Project_ZENITH" is found
while currentpath.split("\\")[-1] != "project_ZENITH":
    currentpath = "\\".join(currentpath.split("\\")[:-1])

# add the path to the sys.path if it does not exist
if currentpath not in sys.path: sys.path.append(currentpath)



from zenith import *
# from zenith.utils.CompactObjects import BlackHole
# from zenith.utils.Geometries import DefaultMesh
# from zenith.initialdata.BowenYork_puncture import BowenYork

def main():

    order = 2
    
    mesh = DefaultMesh()
    scalar_space = H1(mesh, order=order) # sclar space 
    vector_space = VectorValued(scalar_space , 3) # vector space
    matrix_space = VectorValued(scalar_space , 9) # symmetric 3x3 matrix space
    tensor_space = VectorValued(scalar_space , 27) # symmetric 3x3x3 tensor space

    # create the gridfunctions
    gf_gamma_c = GridFunction(matrix_space)
    gf_A_c = GridFunction(matrix_space)
    gf_W = GridFunction(scalar_space)   
    gf_K = GridFunction(scalar_space)
    gf_Gamma_c = GridFunction(vector_space)

    # initialize the gridfunctions
    gf_gamma_c.Set(CF((1,0,0, 0,1,0, 0,0,1)))
    r = sqrt(x*x+y*y+z*z)
    h = exp(-r*r)
    f = h.Diff(y)
    g = h.Diff(z)

    gf_A_c.Set(CF((0,0,0, 0,-f,g, 0,g,f)))
    psi = 1 + 1/r + h 
    gf_W.Set(CF(( 1/(psi*psi) )) )
    gf_K.Set(CF(0))
    gf_Gamma_c.Set(CF((0,0,0)))

    # draw all the gridfunctions
    Draw(gf_gamma_c,mesh,"gamma_c")
    Draw(gf_A_c,mesh,"A_c")
    Draw(gf_W,mesh,"W")
    Draw(gf_K,mesh,"K")
    Draw(gf_Gamma_c,mesh,"Gamma_c")

    # initialize the BSSN evolution equation
    # bssn = BSSNPuncture(gf_gamma_c, gf_A_c, gf_W, gf_K, gf_Gamma_c)


if __name__ == "__main__":
    
    with TaskManager(): main()
    


    
    