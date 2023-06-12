# in this file we define the einstein bianchi equations

from ngsolve import *
from netgen.csg import *


# define the class for the Einstein Bianchi equations
class LinEinsteinBianchi:
    def __init__(self, mesh, order, **kwargs):
        
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
        inverse = kwargs.get("inverse", "pardiso")
        self.invmassHcc = self.massHcc.mat.Inverse(self.Hcc.FreeDofs(), inverse)
        self.invmassHcd = self.massHcd.mat.Inverse(self.Hcd.FreeDofs(), inverse)
        print("Define inverse mass matrices: done")

        print("Define stiffness matrices:", end="\r")
        def vol_curl(u,v):
            return InnerProduct(curl(u), v) * dx
        
        
        n = specialcf.normal(mesh.dim)
        Cn = CoefficientFunction( (0, n[2], -n[1], -n[2], 0, n[0], n[1], -n[0], 0) , dims=(3,3) )
        Pn = OuterProduct(n,n)
        Qn = Id(3) - Pn        
        def CurlOp(u,v) : return InnerProduct(curl(u),v)*dx - InnerProduct(Pn*u*Cn,Pn*v*Qn)*dx(element_boundary = True) 



def Curl(u):
    if u.dim == 3:
        return CF( (u[1].Diff(z)- u[2].Diff(y), u[2].Diff(x)- u[0].Diff(z), u[0].Diff(y)- u[1].Diff(x)) )
    if u.dim == 9:
        return CF( (Curl(u[0,:]),Curl(u[1,:]),Curl(u[2,:])),dims=(3,3) )

def main():

    kwargs = {"inverse" : "pardiso",
              "nonassemble" : True, 
              "order" : 2}

    # check if the curl is working




    cube = OrthoBrick( Pnt(0,0,0), Pnt(1,1,1) )

    geo = CSGeometry()
    geo.Add (cube)
    mesh = Mesh(geo.GenerateMesh(maxh=0.1) )


    n = specialcf.normal(mesh.dim)
    Cn = CoefficientFunction( (0, n[2], -n[1], -n[2], 0, n[0], n[1], -n[0], 0) , dims=(3,3) )
    Pn = OuterProduct(n,n)
    Qn = Id(3) - Pn        
    def CurlOp(u,v) : return InnerProduct(curl(u),v)*dx - InnerProduct(Pn*u*Cn,Pn*v*Qn)*dx(element_boundary = True) 
    

    Hcc = HCurlCurl(mesh, order = kwargs.get("order",2)  , dirichlet = "outer")
    Hcd = HCurlDiv(mesh, order = kwargs.get("order",2)  , dirichlet = "outer")

    # test the curl of a Hcd function
    h = exp(-x*x-y*y-z*z)
    cf_init = CF((h,0,0 , 0,-h,0 , 0,0,0), dims=(3,3))
    gf_init = GridFunction(Hcc)
    gf_init.Set(cf_init)

    gf_exact_curl = GridFunction(Hcc)
    gf_exact_curl.Set(Curl(cf_init))

    Draw(Curl(cf_init),mesh,"ExactCurl")
    Draw(gf_exact_curl,mesh,"NumericalCurl")

    print("L2Error", Integrate(sqrt(Norm(Curl(cf_init)-gf_exact_curl)),mesh))




if __name__ == "__main__":
    with TaskManager():
        main()