# in this file we define the einstein bianchi equations

from ngsolve import *
from netgen.csg import *    
import matplotlib.pyplot as plt
from ngsolve.la import EigenValues_Preconditioner

import numpy as np

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
        self.order = kwargs.get("order", 2)
        dirichlet = kwargs.get("dirichlet", "")
        condense = kwargs.get("condense", False)
        nonassemble = kwargs.get("nonassemble", False)

        # mesh and finite element spaces
        self.mesh = mesh
        self.fescc = HCurlCurl(mesh, order = self.order,  dirichlet = dirichlet)
        self.fescd = HCurlDiv(mesh, order=self.order,  dirichlet = dirichlet)
        self.fesd = HDiv(mesh, order=self.order, RT=True,   dirichlet = dirichlet)
        
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
        self.v, self.dv = self.fesd.TnT()
        self.B, self.dB = self.fescd.TnT()

        # RHS matrices
        self.bfcurlT = BilinearForm(CurlTHcc2Hcd(self.E, self.fescd.TestFunction()), nonassemble= nonassemble).Assemble()
        self.bfdiv = BilinearForm(DivHcdHd(self.B, self.dv)).Assemble()

        # dont know if this is needed
        #for e in mesh.edges:
        #    for dof in self.fescc.GetDofNrs(e):
        #        self.fescc.couplingtype[dof] = COUPLING_TYPE.WIREBASKET_DOF

        if condense:
            self.massE = BilinearForm(InnerProduct(self.E,self.dE)*dx, condense=True)
            self.preE = Preconditioner(self.massE, "bddc", block=True, blocktype="edgepatch")
            self.massE.Assemble()
            matE = self.massE.mat
            # preE = matE.CreateBlockSmoother(fescc.CreateSmoothingBlocks(blocktype="edgepatch", eliminate_internal=True), GS=False)

            self.massEinvSchur = CGSolver (matE, self.preE)
            Eext = IdentityMatrix()+self.massE.harmonic_extension
            EextT = IdentityMatrix()+self.massE.harmonic_extension_trans
            self.massEinv =  Eext @ self.massEinvSchur @ EextT + self.massE.inner_solve
        else :
            self.massE = BilinearForm(InnerProduct(self.E,self.dE)*dx)
            self.preE = Preconditioner(self.massE, "bddc", block=True, blocktype="edgepatch")
            self.massE.Assemble()
            self.matE = self.massE.mat
            self.massEinv = CGSolver (self.matE, self.preE)

        if condense:
            self.massB = BilinearForm(InnerProduct(self.B,self.dB)*dx, condense=True)
            self.preB = Preconditioner(self.massB, "bddc", block=True, blocktype="edgepatch")
            # preH = matH.CreateSmoother(fescd.FreeDofs(True), GS=False)
            self.massB.Assemble()
            self.matB = self.massB.mat
            self.massBinvSchur = CGSolver (self.matB, self.preB)
            ext = IdentityMatrix()+self.massB.harmonic_extension
            extT = IdentityMatrix()+self.massB.harmonic_extension_trans
            self.massBinv =  ext @ self.massBinvSchur @ extT + self.massB.inner_solve
        else :
            self.massB = BilinearForm(InnerProduct(self.B,self.dB)*dx)
            self.preB = Preconditioner(self.massB, "bddc", block=True, blocktype="edgepatch")
            self.massB.Assemble()
            self.matB = self.massB.mat
            self.massBinv = CGSolver (self.matB, self.preB)
        
        
        if condense:
            self.massv = BilinearForm(InnerProduct(self.v,self.dv)*dx, condense=True).Assemble()
            self.matv = self.massv.mat
            self.prev = self.matv.CreateSmoother(self.fesd.FreeDofs(True), GS=False)

            self.massvinvSchur = CGSolver (self.matv, self.prev)
            self.ext = IdentityMatrix()+self.massv.harmonic_extension
            self.extT = IdentityMatrix()+self.massv.harmonic_extension_trans
            self.massvinv =  self.ext @ self.massvinvSchur @ self.extT + self.massv.inner_solve
        else :  
            self.massv = BilinearForm(InnerProduct(self.v,self.dv)*dx)
            self.prev = Preconditioner(self.massv, "bddc", block=True, blocktype="edgepatch")
            self.massv.Assemble()
            self.matv = self.massv.mat
            self.massvinv = CGSolver (self.matv, self.prev)

    def SetInitialCondition(self, **kwargs):

        peak = exp(-((x-0.5)**2+(y-0.5)**2+(z-0.5)**2)/ 0.2**2 )
        self.gfE.Set ( ((peak, 0,0), (0,0,0), (0,0,-peak) ))     
        self.gfB.Set ( ((0,0,-peak), (0,0,0), (-peak,0,0) ))

    def TimeStep(self, dt, **kwargs):

        # tend = 5 * dt
        #scene = Draw(Norm(gfB), mesh)
  
        self.gfE.vec.data += -dt * self.massEinv@self.bfcurlT.mat.T * self.gfB.vec
        self.gfv.vec.data += -dt * self.massvinv@self.bfdiv.mat * self.gfB.vec
        hv = self.bfcurlT.mat * self.gfE.vec + self.bfdiv.mat.T * self.gfv.vec
        self.gfB.vec.data += dt * self.massBinv * hv
        #self.gfB.vec.data += dt * self.massBinvSchur * (self.bfcurlT.mat * self.gfE.vec + self.bfdiv.mat.T * self.gfv.vec)
        
  
    def PlotEigenvalues(self):
        print (EigenValues_Preconditioner(self.matE, self.preE).NumPy())
        print (EigenValues_Preconditioner(self.matB, self.preB).NumPy())
        print (EigenValues_Preconditioner(self.matv, self.prev).NumPy())

    def TrachEnergy(self):
        self.energyTraceE.append (Integrate ( Norm (Trace(self.gfE)), self.mesh ))
        self.energySymB.append (Integrate ( Norm (self.gfB.trans -self.gfB), self.mesh ))
        self.energyE.append (Integrate ( Norm (self.gfE), self.mesh ))
        self.energyB.append (Integrate ( Norm (self.gfB), self.mesh ))
        self.energyv.append (Integrate ( Norm (self.gfv), self.mesh ))

    def PlotEnergy(self):
        plt.plot(self.energyTraceE, label='TraceE')
        plt.plot(self.energySymB, label='SymB')
        plt.plot(self.energyE, label='E')
        plt.plot(self.energyB, label='B')
        plt.plot(self.energyv, label='v')
        plt.legend()
        plt.show()


def main():
        
        mesh= Mesh(unit_cube.GenerateMesh(maxh=0.2))
        
        t = 0
        tend = 1
        dt = 0.001

        eb = EinsteinBianchi(mesh, condense=True, order=2)
        eb.SetInitialCondition()
        
        Draw (Norm(eb.gfE), mesh, "E")
        #Draw (Norm(Norm(eb.gfB)), mesh, "B")
        #Draw (Norm(eb.gfv), mesh, "v")
        eb.TrachEnergy()

        while t < tend:
            t += dt
            eb.TimeStep(dt)
            eb.TrachEnergy()
            #Draw (Norm(eb.gfE), mesh, "E")
            #Draw (Norm(eb.gfB), mesh, "B")
            #Draw (Norm(eb.gfv), mesh, "v")
            
            # print time in percent and the energies to the 4th decimal place
            print(" t:", int(100*t/tend) , "% " , "E: {:.4f} B: {:.4f} v: {:.4f}".format(eb.energyE[-1], eb.energyB[-1], eb.energyv[-1]), end="\r")
        eb.PlotEnergy()

if __name__ == "__main__":
    with TaskManager(): main()