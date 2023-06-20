# in this file we define the einstein bianchi equations

from ngsolve import *
from netgen.csg import *    
import matplotlib.pyplot as plt
from ngsolve.la import EigenValues_Preconditioner
import numpy as np


from netgen.csg import *


from ngsolve.internal import visoptions
from ngsolve.internal import viewoptions

visoptions.scalfunction='u:0'
visoptions.clipsolution = 'scal'
viewoptions.clipping.nx= 0
viewoptions.clipping.ny= 0
viewoptions.clipping.nz= -1
viewoptions.clipping.enable = 1







currentpath = "/home/ebonetti/Desktop/project_ZENITH/"
if currentpath not in sys.path: sys.path.append(currentpath)


n = specialcf.normal(3)

def CurlTHcc2Hcd(E,dH):
    return InnerProduct(curl(E).trans, dH)*dx \
       +InnerProduct(Cross(E*n, n), dH*n)*dx(element_boundary= True)

def DivHcdHd(H,dv):
    return div(H)*dv*dx - H*n*n * dv*n * dx(element_boundary= True)

def Transpose(gfu):
    return CF((   gfu[0,0],gfu[1,0],gfu[2,0],
                  gfu[0,1],gfu[1,1],gfu[2,1],
                  gfu[0,2],gfu[1,2],gfu[2,2]), dims = (3,3))

def Trace(gfu): return CF((gfu[0,0]+gfu[1,1]+gfu[2,2]))

# define the class for the Einstein Bianchi equations
class EinsteinBianchi:
    """
    Creates the linearized Einstein Bianchi equations. 
    input:
    - mesh [Mesh]: the mesh
    - order [int]: the order of the finite element space, default 2
    - dirichlet [string]: the dirichlet boundary condition, default ""
    - condense [bool]: if True, condense the system with Schur complement, default False
    - nonassemble [bool]: if True, do not assemble the RHS matrices, default False
    - iterative [bool]: if True, use iterative solver, default False
    - divfree [bool]: if True, impose a projection into divvergence free space, default False
    - preconditioner [string]: the preconditioner to use, default "direct"
    - print [bool]: if True, print the informations to schreen, default True
    - visualize [bool]: if True, visualize the solution, default False
    - bonus_intorder [int]: the integration order for the bonus terms, default 0

    attributes:
    - all the input attributes
    - all the finite element spaces
    - all the grid functions
    - all the trial and test functions
    - all the bilinear forms
    - 

    This class creates the linearized Einstein Bianchi equations with the variables:
    - Hcc: E electric Weyl field
    - Hcd: B magnetic Weyl field
    - Hd: v auxiliary field (divergence of the magnetic field)

    The equations are:
    - E_t = curl(B) 
    - B_t = -curl(E)
    - div(B) = 0
    - div(E) = 0

    The continuous setting impose E, B to be STD (symmetric traceless divergence-free) tensors,
    dropping the requirement of div(E) = 0. This is done by adding a Lagrange multiplier v
    - E_t = curl(B) 
    - v_t = -div(E)
    - B_t = -curl(E) + grad(v)
    """
    
    def __init__(self, mesh, **kwargs):
        self.mesh = mesh
        self.order = kwargs.get("order", 2)
        self.dirichlet = kwargs.get("dirichlet", "")
        self.condense = kwargs.get("condense", False)
        self.nonassemble = kwargs.get("nonassemble", False)
        self.iterative = kwargs.get("iterative", False)
        self.divfree = kwargs.get("divfree", False)

        # if print then print the parameters
        if kwargs.get("print", True):
            print("\nEinsteinBianchi parameters:")
            print("- order      : ", self.order)
            print("- dirichlet  : ", self.dirichlet)
            print("- condense   : ", self.condense)
            print("- nonassemble: ", self.nonassemble)
            print("- iterative  : ", self.iterative)
            print("- divfree    : ", self.divfree)

        # mesh and finite element spaces
        self.mesh = mesh
        self.fescc = HCurlCurl(self.mesh, order = self.order,  dirichlet = self.dirichlet)
        self.fescd = HCurlDiv(self.mesh, order=self.order,  dirichlet = self.dirichlet)
        self.fesd = HDiv(self.mesh, order=self.order, RT=True,   dirichlet = self.dirichlet)
        if self.divfree:
            self.fescd_d =   self.fescd*self.fesd
        #self.fescd_d = self.fescd*self.fesd

        # define the grid functions
        self.gfE = GridFunction(self.fescc)
        self.gfB = GridFunction(self.fescd)
        self.gfv = GridFunction(self.fesd)
  

        self.gfB.vec[:] = 0.0
        self.gfE.vec[:] = 0.0
        self.gfv.vec[:] = 0.0

        self.energyB = []
        self.energyE = []
        self.energyv = []
        self.energyTraceE = []
        self.energySymB = []


        # define trial and test functions
        self.E, self.dE = self.fescc.TnT()
        if self.divfree:
            (self.B,self.v), (self.dB, self.dv) = self.fescd_d.TnT()
        else :
            self.v, self.dv = self.fesd.TnT()
            self.B, self.dB = self.fescd.TnT()

        # RHS matrices
        self.bfcurlT = BilinearForm(CurlTHcc2Hcd(self.E, self.fescd.TestFunction()), nonassemble= self.nonassemble)
        self.bfdiv = BilinearForm(DivHcdHd(self.B, self.dv), nonassemble= self.nonassemble)
        if self.nonassemble == True:
            self.bfcurl = BilinearForm(CurlTHcc2Hcd(self.dE, self.fescd.TrialFunction()), nonassemble= self.nonassemble)
            self.bfdivT = BilinearForm(DivHcdHd(self.dB, self.v), nonassemble= self.nonassemble)
            self.matcurl = self.bfcurl.mat
            self.matdiv = self.bfdiv.mat
            self.matcurlT = self.bfcurlT.mat
            self.matdivT = self.bfdivT.mat

        if self.nonassemble == False:
            self.bfcurlT.Assemble()
            self.bfdiv.Assemble()
            self.matcurlT = self.bfcurlT.mat
            self.matdiv = self.bfdiv.mat
            self.matcurl = self.bfcurlT.mat.T
            self.matdivT = self.bfdiv.mat.T
            


        # dont know if this is needed
        for e in mesh.edges:
            for dof in self.fescc.GetDofNrs(e):
                self.fescc.couplingtype[dof] = COUPLING_TYPE.WIREBASKET_DOF

        if self.condense and self.iterative :
            # mass matrix for the electric field
            self.massE = BilinearForm(InnerProduct(self.E,self.dE)*dx, condense=True)
            self.preE = Preconditioner(self.massE, "local")#, block=True, blocktype="edgepatch")
            self.massE.Assemble()
            matE = self.massE.mat
            self.massEinvSchur = CGSolver (matE, self.preE, printrates = False)
            Eext = IdentityMatrix()+self.massE.harmonic_extension
            EextT = IdentityMatrix()+self.massE.harmonic_extension_trans
            self.massEinv =  Eext @ self.massEinvSchur @ EextT + self.massE.inner_solve

            # mass matrix for the magnetic field
            self.massB = BilinearForm(InnerProduct(self.B,self.dB)*dx, condense=True)
            self.preB = Preconditioner(self.massB, "bddc", block=True, blocktype="edgepatch")
            self.massB.Assemble()
            self.matB = self.massB.mat
            self.massBinvSchur = CGSolver (self.matB, self.preB, printrates = False)
            ext = IdentityMatrix()+self.massB.harmonic_extension
            extT = IdentityMatrix()+self.massB.harmonic_extension_trans
            self.massBinv =  ext @ self.massBinvSchur @ extT + self.massB.inner_solve

            # mass matrix for the divergence field
            self.massv = BilinearForm(InnerProduct(self.v,self.dv)*dx, condense=True).Assemble()
            self.matv = self.massv.mat
            self.prev = self.matv.CreateSmoother(self.fesd.FreeDofs(True), GS=False)
            self.massvinvSchur = CGSolver (self.matv, self.prev, printrates = False)
            self.Vext = IdentityMatrix()+self.massv.harmonic_extension
            self.VextT = IdentityMatrix()+self.massv.harmonic_extension_trans
            self.massvinv =  self.Vext @ self.massvinvSchur @ self.VextT + self.massv.inner_solve

        elif self.iterative :
            # mass matrix for the electric field
            self.massE = BilinearForm(InnerProduct(self.E,self.dE)*dx)
#            self.preE = Preconditioner(self.massE, "bddc", block=True, blocktype="edgepatch")
            self.preE = Preconditioner(self.massE, "local")#, block=True, blocktype="edgepatch")
            self.massE.Assemble()
            self.matE = self.massE.mat
            self.massEinv = CGSolver (self.matE, self.preE, printrates = False)

            # mass matrix for the magnetic field
            self.massB = BilinearForm(InnerProduct(self.B,self.dB)*dx)
#            self.preB = Preconditioner(self.massB, "bddc", block=True, blocktype="edgepatch")
            self.preB = Preconditioner(self.massB, "local")#, block=True, blocktype="edgepatch") 
            self.massB.Assemble()
            self.matB = self.massB.mat
            self.massBinv = CGSolver (self.matB, self.preB, printrates = False)

            # mass matrix for the divergence field
            self.massv = BilinearForm(InnerProduct(self.v,self.dv)*dx)
            #self.prev = Preconditioner(self.massv, "bddc", block=True, blocktype="edgepatch")
            self.prev = Preconditioner(self.massv, "local")#, block=True, blocktype="edgepatch")
            self.massv.Assemble()
            self.matv = self.massv.mat
            self.massvinv = CGSolver (self.matv, self.prev, printrates = False)

        elif self.divfree :
            print("divfree method")
            self.massE = BilinearForm(InnerProduct(self.E,self.dE)*dx).Assemble()
            self.massH = BilinearForm(self.fescd_d)
            self.massH += InnerProduct(self.B,self.dB)*dx + DivHcdHd(self.B,self.dv) + DivHcdHd(self.dB,self.v) - 1e-3*self.v*self.dv*dx - div(self.v)*div(self.dv)*dx
            self.massH.Assemble()
            self.massEinv = self.massE.mat.Inverse(inverse="sparsecholesky")
            self.massHinv = self.massH.mat.Inverse(inverse="sparsecholesky")
            self.resB = self.fescd_d.restrictions[0]
            self.massBinv = self.resB@self.massHinv@self.resB.T
            self.massB = self.resB @ self.massH.mat @ self.resB.T

        else :
            print("basic method")
            # mass matrix for the electric field
            self.massE = BilinearForm(InnerProduct(self.E,self.dE)*dx).Assemble()
            self.massEinv = self.massE.mat.Inverse(inverse="sparsecholesky")

            # mass matrix for the magnetic field
            self.massB = BilinearForm(InnerProduct(self.B,self.dB)*dx).Assemble()
            self.massBinv = self.massB.mat.Inverse(inverse="sparsecholesky")

            # mass matrix for the divergence field
            self.massv = BilinearForm(InnerProduct(self.v,self.dv)*dx).Assemble()
            self.massvinv = self.massv.mat.Inverse(inverse="sparsecholesky")

    def SetInitialCondition(self, E0 = None, B0 = None,  **kwargs):
        print("Set initial conditions")
        peak = exp(-((x)**2+(y)**2+(z)**2)/0.5**2 )
        if E0 is None:
            E0 =  ((peak, 0,0), (0,0,0), (0,0,-peak) )
        if B0 is None:  
            B0 =  ((0,0,0), (0,0,peak), (0,-peak,0) )
        self.gfE.Set ( E0 , bonus_intorder=kwargs.get("bonus_intorder", 0), dual = kwargs.get("dual", False))     
        self.gfB.Set ( B0 , bonus_intorder=kwargs.get("bonus_intorder", 0) , dual = kwargs.get("dual", False))

    def TimeStep(self, dt, **kwargs):
        # tend = 5 * dt
        if self.divfree :
                self.gfE.vec.data += -dt * self.massEinv@self.matcurl *self.gfB.vec
                self.gfB.vec.data += dt *  self.massBinv@self.matcurlT * self.gfE.vec

        else : #if self.nonassemble :
            self.gfE.vec.data += -dt * self.massEinv@self.matcurl * self.gfB.vec
            self.gfv.vec.data += dt * self.massvinv@self.matdiv * self.gfB.vec
            hv = self.matcurlT * self.gfE.vec - self.matdivT * self.gfv.vec
            self.gfB.vec.data += dt * self.massBinv * hv
        #else :
        #    self.gfE.vec.data += -dt * self.massEinv@self.bfcurlT.mat.T * self.gfB.vec
        #    self.gfv.vec.data += -dt * self.massvinv@self.bfdiv.mat * self.gfB.vec
        #    hv = self.bfcurlT.mat * self.gfE.vec + self.bfdiv.mat.T * self.gfv.vec
        #    self.gfB.vec.data += dt * self.massBinv * hv
        #self.gfB.vec.data += dt * self.massBinvSchur * (self.bfcurlT.mat * self.gfE.vec + self.bfdiv.mat.T * self.gfv.vec)  
  
    def PlotEigenvalues(self):
        print (EigenValues_Preconditioner(self.matE, self.preE).NumPy())
        print (EigenValues_Preconditioner(self.matB, self.preB).NumPy())
        print (EigenValues_Preconditioner(self.matv, self.prev).NumPy())

    def TrachEnergy(self):
        self.energyTraceE.append (Integrate ( Norm (Trace(self.gfE)), self.mesh ))
        self.energySymB.append (Integrate ( Norm (self.gfB.trans -self.gfB), self.mesh ))
        self.energyE.append (sqrt(InnerProduct((self.massE.mat * self.gfE.vec),self.gfE.vec) ) )
        if not self.divfree :
            self.energyv.append (sqrt(InnerProduct((self.massv.mat * self.gfv.vec),self.gfv.vec) ) )
            self.energyB.append (sqrt(InnerProduct((self.massB.mat * self.gfB.vec),self.gfB.vec) ) )
        else :
            self.energyB.append (sqrt(InnerProduct((self.massB * self.gfB.vec),self.gfB.vec) ) )

        # print the energy
        # print ("TraceE: ", self.energyTraceE[-1])
        # print ("SymB: ", self.energySymB[-1])
        # print ("E: ", self.energyE[-1])
        # print ("B: ", self.energyB[-1])
        # if not self.divfree :
        #     print ("v: ", self.energyv[-1])

    def PlotEnergy(self, **kwargs):
        plt.figure(1)
        plt.plot(self.energyE, label='E')
        plt.plot(self.energyB, label='B')
        if not self.divfree :
            plt.plot(self.energyv, label='v')
        # plot sum of (E^2 and B^2)^1/2
        plt.plot([sqrt(self.energyE[i]**2 + self.energyB[i]**2) for i in range(len(self.energyE))], label='E+B')
        plt.plot(self.energyTraceE, label='TraceE')
        plt.plot(self.energySymB, label='SymB')
        plt.legend()
        if kwargs.get("save", False):
            dirname = kwargs.get("dirname", "")
            figname = kwargs.get("figname", "")
            plt.savefig(dirname + figname + ".png" )
        plt.show()

    def Draw(self):
        Draw(self.gfE,self.mesh, "E")
        Draw(self.gfB,self.mesh, "B")
        if not self.divfree :
            Draw(self.gfv, self.mesh, "divB")



def main():

    geo = CSGeometry()
    geo.Add (Sphere (Pnt(0,0,0), 1))

    mesh = Mesh(geo.GenerateMesh(maxh=0.2))

    eb = EinsteinBianchi(mesh, bonus_intorder=10, condense = True)

    #E00 = 0
    #E01 = (-3200*z + 40*(40*z - 20.0)*(40*(y - 0.5)**2 - 1) + 40*(40*z - 20.0)*(40*(z - 0.5)**2 - 1) + 1600.0)*exp(-20*(x - 0.5)**2 - 20*(y - 0.5)**2 - 20*(z - 0.5)**2)
    #E02 = (3200*y - 40*(40*y - 20.0)*(40*(y - 0.5)**2 - 1) - 40*(40*y - 20.0)*(40*(z - 0.5)**2 - 1) - 1600.0)*exp(-20*(x - 0.5)**2 - 20*(y - 0.5)**2 - 20*(z - 0.5)**2)
    #E10 = (-4800*z + (y - 0.5)**2*(64000*z - 32000.0) + 40*(40*z - 20.0)*(40*(z - 0.5)**2 - 1) + 2400.0)*exp(-20*(x - 0.5)**2 - 20*(y - 0.5)**2 - 20*(z - 0.5)**2)
    #E11 = -2*(40*x - 20.0)*(40*y - 20.0)*(40*z - 20.0)*exp(-20*(x - 0.5)**2 - 20*(y - 0.5)**2 - 20*(z - 0.5)**2)
    #E12 = (-1600*x - 40*(40*x - 20.0)*(40*(z - 0.5)**2 - 1) + (64000*x - 32000.0)*(y - 0.5)**2 + 800.0)*exp(-20*(x - 0.5)**2 - 20*(y - 0.5)**2 - 20*(z - 0.5)**2)
    #E20 = (4800*y + (32000.0 - 64000*y)*(z - 0.5)**2 - 40*(40*y - 20.0)*(40*(y - 0.5)**2 - 1) - 2400.0)*exp(-20*(x - 0.5)**2 - 20*(y - 0.5)**2 - 20*(z - 0.5)**2)
    #E21 = (1600*x + (32000.0 - 64000*x)*(z - 0.5)**2 + 40*(40*x - 20.0)*(40*(y - 0.5)**2 - 1) - 800.0)*exp(-20*(x - 0.5)**2 - 20*(y - 0.5)**2 - 20*(z - 0.5)**2)
    #E22 = 2*(40*x - 20.0)*(40*y - 20.0)*(40*z - 20.0)*exp(-20*(x - 0.5)**2 - 20*(y - 0.5)**2 - 20*(z - 0.5)**2)
    E00 = 0
    E01 = 16*sqrt(610)*z*(5*y**2 + 5*z**2 - 1)*exp(-10*x**2 - 10*y**2 - 10*z**2)/61
    E02 = 16*sqrt(610)*y*(-5*y**2 - 5*z**2 + 1)*exp(-10*x**2 - 10*y**2 - 10*z**2)/61
    E10 = 16*sqrt(610)*z*(5*y**2 + 5*z**2 - 1)*exp(-10*x**2 - 10*y**2 - 10*z**2)/61
    E11 = -160*sqrt(610)*x*y*z*exp(-10*x**2 - 10*y**2 - 10*z**2)/61
    E12 = 80*sqrt(610)*x*(y**2 - z**2)*exp(-10*x**2 - 10*y**2 - 10*z**2)/61
    E20 = 16*sqrt(610)*y*(-5*y**2 - 5*z**2 + 1)*exp(-10*x**2 - 10*y**2 - 10*z**2)/61
    E21 = 80*sqrt(610)*x*(y**2 - z**2)*exp(-10*x**2 - 10*y**2 - 10*z**2)/61
    E22 = 160*sqrt(610)*x*y*z*exp(-10*x**2 - 10*y**2 - 10*z**2)/61
    #E00 = 0
    #E01 = (-8.0e-8*z + (2*z - 1.0)*(4.0e-8*(y - 0.5)**2 - 2.0e-8) + (2*z - 1.0)*(4.0e-8*(z - 0.5)**2 - 2.0e-8) + 4.0e-8)*exp(-(x - 0.5)**2 - (y - 0.5)**2 - (z - 0.5)**2)
    #E02 = (8.0e-8*y - (2*y - 1.0)*(4.0e-8*(y - 0.5)**2 - 2.0e-8) - (2*y - 1.0)*(4.0e-8*(z - 0.5)**2 - 2.0e-8) - 4.0e-8)*exp(-(x - 0.5)**2 - (y - 0.5)**2 - (z - 0.5)**2)
    #E10 = (-1.2e-7*z + (2.0e-8*y - 1.0e-8)*(2*y - 1.0)*(2*z - 1.0) + (2*z - 1.0)*(4.0e-8*(z - 0.5)**2 - 2.0e-8) + 6.0e-8)*exp(-(x - 0.5)**2 - (y - 0.5)**2 - (z - 0.5)**2)
    #E11 = -(2*z - 1.0)*((2.0e-8*x - 1.0e-8)*(2*y - 1.0) + (2*x - 1.0)*(2.0e-8*y - 1.0e-8))*exp(-(x - 0.5)**2 - (y - 0.5)**2 - (z - 0.5)**2)
    #E12 = (-4.0e-8*x + (8.0e-8*x - 4.0e-8)*(y - 0.5)**2 - (2*x - 1.0)*(4.0e-8*(z - 0.5)**2 - 2.0e-8) + 2.0e-8)*exp(-(x - 0.5)**2 - (y - 0.5)**2 - (z - 0.5)**2)
    #E20 = (1.2e-7*y + (4.0e-8 - 8.0e-8*y)*(z - 0.5)**2 - (2*y - 1.0)*(4.0e-8*(y - 0.5)**2 - 2.0e-8) - 6.0e-8)*exp(-(x - 0.5)**2 - (y - 0.5)**2 - (z - 0.5)**2)
    #E21 = (4.0e-8*x + (4.0e-8 - 8.0e-8*x)*(z - 0.5)**2 + (2*x - 1.0)*(4.0e-8*(y - 0.5)**2 - 2.0e-8) - 2.0e-8)*exp(-(x - 0.5)**2 - (y - 0.5)**2 - (z - 0.5)**2)
    #E22 = (2*z - 1.0)*((2.0e-8*x - 1.0e-8)*(2*y - 1.0) + (2*x - 1.0)*(2.0e-8*y - 1.0e-8))*exp(-(x - 0.5)**2 - (y - 0.5)**2 - (z - 0.5)**2)
    E = CoefficientFunction( ( (E00,E01,E02), (E10,E11,E12), (E20,E21,E22) ) , dims=(3,3) )
    B = CoefficientFunction( ( (0,0,0), (0,0,0), (0,0,0) ) , dims=(3,3) )
    #B = CoefficientFunction( ( (0,0,- cos(y-t)), (0,0,0), (- cos(y-t),0,0) ) , dims=(3,3) )
    eb.SetInitialCondition(E,B, dual=True)

    eb.Draw()
    input("Press any key...")
    tend = 1
    t = 0
    dt = 0.01
    
    
    while t < tend:
        t += dt
        eb.TimeStep(dt)
        eb.TrachEnergy()
        eb.Draw()
        print("t = ", int(t/tend*100) ,"%" , end='\r')

    eb.PlotEnergy()


    

    

if __name__ == "__main__":
    with TaskManager(): main()

