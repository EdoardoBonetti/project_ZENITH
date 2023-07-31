# in this file we implement the BSSN formulation of the Einstein equations given an initial condition
# 
#   for the initial conidtion we use the Bowen York solution for two black holes implemented in the file BowenYork.py
#
#   the BSSN-puncture formulation is given in the book a 100 years of relativity by Masaru Shibata



from ngsolve import *
from netgen.csg import *
from ngsolve.solvers import *
# import the Bowen York solution

#from utils.Geometries import *
#from utils.CompactObjects import *


# now we define the class for the BSSN-puncture variables




class BSSNPuncture:
    # the __init__ function take as input the 
    def __init__(self, gf_gamma_c , gf_A_c , gf_W, gf_K, gf_Gamma_c, **kwargs):

        self.scl_space = gf_W.space
        self.vec_space = gf_Gamma_c.space
        self.mat_space = gf_gamma_c.space
        self.tns_space = VectorValued(W.space, 27)

        # define the TnT fcs
        gamma , dgamma = self.mat_space.TnT()
        A, dA = self.mat_space.TnT()
        W, dW = self.scl_space.TnT()
        K, dK = self.scl_space.TnT()
        Gamma, dGamma = self.vec_space.TnT()

        # define the gridfunctions
        self.gf_gamma_c = gf_gamma_c
        self.gf_A_c = gf_A_c
        self.gf_W = gf_W
        self.gf_K = gf_K
        self.gf_Gamma_c = gf_Gamma_c

        self.gf_Chris_c = GridFunction(self.tns_space)
        self.gf_Chris_w = GridFunction(self.tns_space)
        self.gf_Ricci = GridFunction(self.mat_space)
        self.gf_Ricciscal_c = GridFunction(self.scl_space)
        self.gf_lapse = GridFunction(self.scl_space)
        self.gf_shift = GridFunction(self.vec_space)

        # Ricci We need Rij_c + Rij_W 
        self.Ricci_c = GridFunction(self.mat_space)
        self.Ricci_W = GridFunction(self.mat_space)

    
    def Step(self, dt):
        self.computeCS2()
        self.ComputeRicci()
        self.h.vec.data += -dt * 2 * self.A.vec  
        self.A.vec.data += dt * self.h.vec 
        #self.W.vec.data += dt * self.W.vec
        #self.K.vec.data += dt * self.A.vec

    def IBilinearForm(fes,F,Fhatn,Ubnd):
        a = BilinearForm(fes, nonassemble=True)
        a += - InnerProduct(F(U),Grad(V)) * dx
        a += InnerProduct(Fhatn(U),V) * dx(element_boundary=True)
        return a


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
