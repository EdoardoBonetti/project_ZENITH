# in this file we define the einstein bianchi equations

from ngsolve import *
from netgen.csg import *


# define the class for the Einstein Bianchi equations
class LinEinsteinBianchi:
    def __init__(self, mesh, order, **kwargs):
        self.mesh = mesh

        self.Hcc = HCurlCurl(mesh, order = order, dirichlet = "outer")
        self.Hcd = HCurlDiv(mesh, order = order, dirichlet = "outer")

        print("Define mass matrices:", end="\r")
        self.massHcc = BilinearForm(self.Hcc, symmetric = True, nonassemble = kwargs.get("nonassemble", False))
        self.massHcc += InnerProduct(self.Hcc.TrialFunction(), self.Hcc.TestFunction())*dx
        if kwargs.get("nonassemble", False) == False: self.massHcc.Assemble()

        self.massHcd = BilinearForm(self.Hcd, symmetric = True, nonassemble = kwargs.get("nonassemble", False))
        self.massHcd += InnerProduct(self.Hcd.TrialFunction(), self.Hcd.TestFunction())*dx
        if kwargs.get("nonassemble", False) == False: self.massHcd.Assemble()
        print("Define mass matrices: done")

        print("Define inverse mass matrices:", end="\r")
        inverse = kwargs.get("inverse", "sparsecholesky")
        self.invmassHcc = self.massHcc.mat.Inverse(self.Hcc.FreeDofs(), inverse)
        self.invmassHcd = self.massHcd.mat.Inverse(self.Hcd.FreeDofs(), inverse)
        print("Define inverse mass matrices: done")

        print("Define curl matrix:", end="\r") 
        u = self.Hcc.TrialFunction()
        dv = self.Hcd.TestFunction()
        n = specialcf.normal(mesh.dim)
        Cn = CoefficientFunction( (0, n[2], -n[1], -n[2], 0, n[0], n[1], -n[0], 0) , dims=(3,3) )
        Pn = OuterProduct(n,n)
        Qn = Id(3) - Pn        
        def CurlOp(u,v) : return InnerProduct(curl(u),v)*dx - InnerProduct(Pn*u*Cn,Pn*v*Qn)*dx(element_boundary = True) 
        self.curlOp = BilinearForm(trialspace = self.Hcc, testspace = self.Hcd, nonassemble = kwargs.get("nonassemble", False))
        self.curlOp += CurlOp(u, dv)
        if kwargs.get("nonassemble", False) == False: self.curlOp.Assemble()
        print("Define curl matrix: done")

        self.invmasscurl = self.invmassHcc @ self.curlOp.mat.T 

        self.gf_E = GridFunction(self.Hcc)
        self.gf_B = GridFunction(self.Hcd)

    def SetInitialValues(self, cf_E, cf_B):
        self.gf_E.Set(cf_E)
        self.gf_B.Set(cf_B)

    def ApplyCurl(self, gf_out, gf_in, **kwargs):
        gf_out.vec.data = self.invmasscurl * gf_in.vec

    def Plot(self):
        Draw(self.gf_E, self.mesh, "E")
        Draw(self.gf_B, self.mesh, "B")

    def Step(self, dt):
        self.gf_E.vec.data -= dt * self.invmassHcc @ self.curlOp.mat.T * self.gf_B.vec
        self.gf_B.vec.data += dt * self.invmassHcd @ self.curlOp.mat * self.gf_E.vec

def Curl(u):
    if u.dim == 3:
        return CF( (u[1].Diff(z)- u[2].Diff(y), u[2].Diff(x)- u[0].Diff(z), u[0].Diff(y)- u[1].Diff(x)) )
    if u.dim == 9:
        return CF( (Curl(u[0,:]),Curl(u[1,:]),Curl(u[2,:])),dims=(3,3) )

