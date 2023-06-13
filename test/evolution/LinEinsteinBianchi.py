
import os
import sys
from sys import path
currentpath = os.getcwd()
# take the current path string and take away the last folder until folder "Project_ZENITH" is found
#while currentpath.split("//")[-1] != "project_ZENITH":
#currentpath = "C:/Users/User/OneDrive/Desktop/project_ZENITH/"
currentpath = "/home/ebonetti/Desktop/project_ZENITH/"
# add the path to the sys.path if it does not exist
if currentpath not in sys.path: sys.path.append(currentpath)


from netgen.csg import *

from ngsolve import *
from zenith import *

def Div(u):
    if u.dim == 3:
        return u[0].Diff(x)+u[1].Diff(y)+u[2].Diff(z)
    if u.dim == 9:
        return CF( (Div(u[0,:]),Div(u[1,:]),Div(u[2,:])),dims=(3,3) )

def Curl(u):
    if u.dim == 3:
        return CF( (u[2].Diff(y)- u[1].Diff(z), -u[2].Diff(x)+ u[0].Diff(z), -u[0].Diff(y)+ u[1].Diff(x)) )
    if u.dim == 9:
        return CF( (Curl(u[0,:]),Curl(u[1,:]),Curl(u[2,:])),dims=(3,3) )
    
def Transpose(u):
    return CF ( (u[0,0],u[1,0],u[2,0],u[0,1],u[1,1],u[2,1],u[0,2],u[1,2],u[2,2]),dims=(3,3) )

def Inc(u):
    return Transpose(Curl( Transpose( Curl( u ) ) ))


def main():

    kwargs = {"inverse" : "sparsecholesky",
              "nonassemble" : False, 
              "order" : 1}

    cube = Sphere(Pnt(0,0,0), 1)
    geo = CSGeometry()
    geo.Add (cube)
    mesh = Mesh(geo.GenerateMesh(maxh=0.4) )

    
    eb = LinEinsteinBianchi(mesh,**kwargs)

    r = sqrt(x*x+y*y+z*z)
    cf_f = CF( (exp(-r*r), 0, 0 , 0, -exp(-r*r), 0, 0, 0, 0), dims=(3,3) )

    cf_init = Inc(cf_f).Compile()

    Draw(cf_init,mesh,"cf_init")

    TestInitialCondition(cf_init, mesh)
    input("Press Enter to continue...")    
    
    initial_E = cf_init
    initial_B = cf_init

    # set initial values
    print("Setting initial values")
    eb.SetInitialValues(cf_E=initial_E, cf_B=initial_B)
    print("Plotting")
    eb.Plot()
    print("Done")
    input("Press Enter to continue...")
    for i in range(100):
        eb.Step(0.005)
        eb.Plot()

def TestInitialCondition(u, mesh):
    # we test if it is divergence free, symmetric and traceless
    print("Testing initial condition")
    #print("Divergence free: ", Integrate(Norm(Div(u).Compile()),mesh))
    #print("Symmetric: ", Integrate(Norm(u-Transpose(u)).Compile(),mesh))
    print("Traceless: ", Integrate(Norm(Trace(u)).Compile(),mesh))


def test(h = 1):
   
    cube = Sphere(Pnt(0,0,0), 2)
    geo = CSGeometry()
    geo.Add (cube)
    mesh = Mesh(geo.GenerateMesh(maxh=h) )
    order = 1
    eb = LinEinsteinBianchi(mesh, order, **{"nonassemble" : False, "inverse" : "sparsecholesky", "dirichlet" : ".*"})

    r = sqrt(x*x+y*y+z*z)
    initial_E = CF( (exp(-r*r), 0, 0 , 0, -exp(-r*r), 0, 0, 0, 0), dims=(3,3) )
    final_B = Curl(initial_E)

    gf_E = GridFunction(eb.Hcc)
    gf_exact_curl_B = GridFunction(eb.Hcd)
    gf_E.Set(initial_E)
    gf_exact_curl_B.Set(final_B)




    gf_numerical_curl_B = GridFunction(eb.Hcd)
    gf_numerical_curl_B.Set(gf_exact_curl_B, BND)
    gf_numerical_curl_B.vec.data = eb.invmassHcd @ eb.curlOp.mat.T* gf_E.vec
    error = gf_exact_curl_B  - final_B
    print("h: " , h , "  error = ", Integrate (sqrt(InnerProduct(error,error)), mesh) )
    # Draw(gf_numerical_curl_B, mesh, "curl_numerical_E")
    # Draw(final_B, mesh, "curl_exact_E")
    # Draw(error, mesh, "error")
    # for i in range(3): 
    #     for j in range(3):
    #         print("error[%d,%d] = %f"%(i,j,Integrate(Norm(error[i,j]),mesh)))
    return Integrate (sqrt(InnerProduct(error,error)), mesh)


if __name__ == "__main__":
    H = []
    with TaskManager():
        for h in [1.5, 0.75, 0.375, 0.1875]:
            H.append(test(h))

    print(H)
    print("Convergence rate: ", [log(H[i-1]/H[i])/log(2) for i in range(1,len(H))])

