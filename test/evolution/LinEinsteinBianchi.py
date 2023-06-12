
import os
import sys
from sys import path
currentpath = os.getcwd()
# take the current path string and take away the last folder until folder "Project_ZENITH" is found
while currentpath.split("/")[-1] != "project_ZENITH":
    currentpath = "~/Desktop/project_ZENITH/"

# add the path to the sys.path if it does not exist
if currentpath not in sys.path: sys.path.append(currentpath)


from netgen.csg import *

from ngsolve import *
from zenith import *


def main():
    # create the mesh
    mesh = DefaultMesh(h = 0.2, R = 6)

    print("Create the LinEinsteinBianchi object")

    eb = LinEinsteinBianchi(mesh, order = 2)




def Curl(u):
    if u.dim == 3:
        return CF( (-u[1].Diff(z)+ u[2].Diff(y), -u[2].Diff(x)+ u[0].Diff(z), -u[0].Diff(y)+ u[1].Diff(x)) )
    if u.dim == 9:
        return CF( (Curl(u[0,:]),Curl(u[1,:]),Curl(u[2,:])),dims=(3,3) )

def Test():

    kwargs = {"inverse" : "pardiso",
              "nonassemble" : True, 
              "order" : 1}

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
    

    Hcc = HCurlCurl(mesh, order = kwargs.get("order",2) )
    Hcd = HCurlDiv(mesh, order = kwargs.get("order",2)  )

    # test the curl of a Hcd function
    h = exp(-x*x-y*y-z*z)
    cf_init = CF((h,0,0 , 0,-h,0 , 0,0,0), dims=(3,3))
    gf_init = GridFunction(Hcc)
    gf_init.Set(cf_init)

    gf_exact_curl = GridFunction(Hcd)
    gf_exact_curl.Set(Curl(cf_init))

    Draw(Curl(cf_init),mesh,"ExactCurl")
    Draw(gf_exact_curl,mesh,"NumericalCurl")
    Draw(Curl(cf_init)-gf_exact_curl,mesh,"ErrorCurl")

    print("L2Error", Integrate(sqrt(Norm(Curl(cf_init)-gf_exact_curl)),mesh))

    for i in range(3):
        for j in range(3):
            print("L2Error",i,j , Integrate(Curl(cf_init)[i,j]-gf_exact_curl[i,j],mesh))

    


if __name__ == "__main__":
    with TaskManager(): Test()

