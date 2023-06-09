# in this file we implement the BSSN formulation of the Einstein equations given an initial condition
# 
#   for the initial conidtion we use the Bowen York solution for two black holes implemented in the file BowenYork.py
#
#   the BSSN-puncture formulation is given in the book a 100 years of relativity by Masaru Shibata



from ngsolve import *
from netgen.csg import *
from ngsolve.solvers import *
# import the Bowen York solution

import os
import sys
sys.path.append(os.getcwd())
from utils.Geometries import *
from utils.CompactObjects import *


# now we define the class for the BSSN-puncture variables

class BSSNPuncture:
    # instead of tilda we use the notation _c for the conformal variables
    # also insetad of gamma we use the notation h for the conformal metric
    def __init__(self, mesh, order = 2):

        self.mesh = mesh
        self.order = order

        # define the finite element spaces  
        self.scalarH1 = H1(self.mesh, order=  self.order)
        self.vecotrH1 = VectorValued(self.scalarH1, 3)
        self.matrixH1 = VectorValued(self.scalarH1, 9) # dimension 3 x 3 symmetric matrix
        self.tensorH1 = VectorValued(self.scalarH1, 27) # dimension 3 x 3 x 3  but last 2 entries are symmetric, in particular indexed is ijk -> i + j + k

        # define the TnT fcs
        gamma , dgamma = self.matrixH1.TnT()
        A, dA = self.matrixH1.TnT()
        W, dW = self.scalarH1.TnT()
        K, dK = self.scalarH1.TnT()
        Gamma, dGamma = self.vecotrH1.TnT()

        # define the grid functions
        self.gf_gamma = GridFunction(self.matrixH1)
        self.gf_A     = GridFunction(self.matrixH1)
        self.gf_W     = GridFunction(self.scalarH1)
        self.gf_K     = GridFunction(self.scalarH1)
        self.gf_Gamma = GridFunction(self.vecotrH1)

        self.gf_ChristoffelII = GridFunction(self.tensorH1)
        self.gf_ContractedChristoffel = GridFunction(self.vecotrH1)
        self.gf_Ricci = GridFunction(self.matrixH1)
        self.gf_RicciScalar = GridFunction(self.scalarH1)

        self.gf_lapse = GridFunction(self.scalarH1)
        self.gf_shift = GridFunction(self.vecotrH1)

        # Ricci We need Rij_c + Rij_W 
        self.Ricci_c = GridFunction(self.matrixH1)
        self.Ricci_W = GridFunction(self.matrixH1)

        # now we can define
        def BilinearFormRicci_c(testspace, trialspace):
            Ricc_c_BLF = BilinearForm(testspace= testspace, trialspace= trialspace)
            div_Gamma = Grad(self.Gamma[0])[0]


    def computeCS2(self):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    self.CS2.vec[i*9 + j*3 + k].Set(self.Gamma[i]* self.h[3*j+k])
    
    def ComputeContractedUpDown(self):
        for i in range(3):
            f = CF((0))
            for k in range(3):
                f += self.Gamma[k]*self.h[3*i+k]
            self.UpperContractedCS2.vec[i].Set(f)


    def ComputeRicci_c(self):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    self.Ricci_c.vec[i*3 +j].Set(Grad(self.CS2[i*9 + j*3 + k].Diff(k) + self.CS2[i*9 + k*3 + j].Diff(k) - self.CS2[k*9 + j*3 + i].Diff(k) - self.CS2[k*9 + i*3 + j].Diff(k)))
    
    def ComputeRicci(self):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    self.Ricci.vec[i*9 + j*3 + k].Set(self.Ricci_c[i*9 + j*3 + k] + self.Ricci_W[i*9 + j*3 + k])

    # meshod to set the initial condition using the Bowen York solution 
    def SetInitialCondition(self, BHs, method = "BowenYork", **kwargs):
        if method == "BowenYork":

            bowenyork = BowenYork(BHs) # set up the problem
            bowenyork.Solve(self.h1)        # solve the pde
            bowenyork.InitialConditions(self.h, self.A, self.W, **kwargs)
            
            self.K.Set(CF((0)))       
            self.Gamma.Set(CF((0,0,0)))




    def SetSlicing(self, method="geodesic"):
        if method == "geodesic":
            self.alpha.Set(CF((1)))
            self.beta.Set(CF((0,0,0)))


    
    def Step(self, dt):
        self.computeCS2()
        self.ComputeRicci()
        self.h.vec.data += -dt * 2 * self.A.vec  
        self.A.vec.data += dt * self.h.vec 
        #self.W.vec.data += dt * self.W.vec
        #self.K.vec.data += dt * self.A.vec


    def Draw(self, mesh):
        Draw(self.h , mesh, "h")
        Draw(self.A , mesh, "A")
        #Draw(self.W, mesh, "W")
        #Draw(self.K, mesh, "K")
        #Draw(self.Gamma, mesh, "Gamma")
        #Draw(self.alpha, mesh, "alpha")
        #Draw(self.beta, mesh, "beta")          

def main():
    
    # create a mesh
    h = 0.23
    R = 6
    order = 2

    # to run better from shell 
    for i in range(len(sys.argv)):
        if sys.argv[i] == "-p":
            order = int(sys.argv[i+1])
        if sys.argv[i] == "-h":
            h = float(sys.argv[i+1])

    kwargs = {"bonus_intorder": 10}

    mesh = DefaultMesh(h=h, R=R)

    # create a list of black holes

    # BH1
    po1 =  CF((0,0,0))
    lin1 = CF((1,0,0))
    ang1 = CF((0,0,0))
    mass1 = CF((1))
    BH1 = BlackHole(mass1, po1, lin1, ang1 )

    # BH2
    #po2 =  CF((-1,0,0))
    #lin2 = CF((1,0,0))
    #ang2 = CF((0,0,0))
    #mass2 = CF((1))
    #BH2 = BlackHole(mass2, lin2, ang2, po2)

    BHs = [BH1]#, BH2]

    ### so far we have created the initial data
    scheme = BSSNPuncture(mesh, order = order)

    # set the initial condition
    scheme.SetInitialCondition(BHs, "BowenYork", **kwargs)
    scheme.CalcChris1()
    # set the slicing condition
    scheme.SetSlicing("geodesic")
    scheme.Draw(mesh)

    dt = 0.1


    input("press any key to start the evolution")
    for i in range(1000):
        scheme.Step(dt)
        if i%10 == 0:
            scheme.Draw(mesh)
        print("step ", i, end="\r")
        


    #scheme.Draw(mesh)
    
if __name__ == '__main__':
    with TaskManager():
        main()
