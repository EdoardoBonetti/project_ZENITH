# in this file we define the einstein bianchi equations

from ngsolve import *
from netgen.csg import *    
import matplotlib.pyplot as plt
from ngsolve.la import EigenValues_Preconditioner
import numpy as np




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
    This class creates the linearized Einstein Bianchi equations with the variables:
    - Hcc: the curl of the electric field
    - Hcd: the curl of the magnetic field
    - Hd: the divergence of the magnetic field
    """
    
    def __init__(self, mesh, **kwargs):
        self.mesh = mesh
        self.order = kwargs.get("order", 2)
        self.dirichlet = kwargs.get("dirichlet", "")
        self.condense = kwargs.get("condense", False)
        self.nonassemble = kwargs.get("nonassemble", False)
        self.iterative = kwargs.get("iterative", False)
        self.divfree = kwargs.get("divfree", False)

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



    def SetInitialCondition(self, **kwargs):
        peak = exp(-((x)**2+(y)**2+(z)**2)/0.5**2 )
        self.gfE.Set ( ((peak, 0,0), (0,0,0), (0,0,-peak) ))     
        #self.gfB.Set ( ((0,0,-peak), (0,0,0), (-peak,0,0) ))

    def TimeStep(self, dt, **kwargs):
        # tend = 5 * dt
        if self.divfree :
                self.gfE.vec.data += -dt * self.massEinv@self.matcurl *self.gfB.vec
                self.gfB.vec.data += dt *  self.massBinv@self.matcurlT * self.gfE.vec

        else : #if self.nonassemble :
            self.gfE.vec.data += -dt * self.massEinv@self.matcurl * self.gfB.vec
            self.gfv.vec.data += -dt * self.massvinv@self.matdiv * self.gfB.vec
            hv = self.matcurlT * self.gfE.vec + self.matdivT * self.gfv.vec
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


