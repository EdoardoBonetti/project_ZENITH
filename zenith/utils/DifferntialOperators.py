# here what I want to do is to create the solution for the following problem:
# given a function a and a function b I want to compute c = a*b
from ngsolve import *
import scipy.sparse as sp
import matplotlib.pylab as plt
from netgen.csg import *

n = specialcf.normal(3)
Cn = CF( (0, n[2], -n[1], -n[2], 0, n[0], n[1], -n[0], 0) , dims=(3,3) )
Pn = OuterProduct(n,n)
Qn = Id(3) - Pn     


def TCurHcc2Hcd(E,dH, **kwargs):
    """ 
    calculates the transpose curl of a HCurlCurl function and a HCurlDiv function and results into a [HCurlDiv]^* function
    E : Test function of HCurlCurl
    dH : Trial function of HCurlDiv
    """
    short = kwargs.get("short", False)
    if short:    return InnerProduct(curl(E).trans, dH)*dx +InnerProduct(Cross(E*n, n), dH*n)*dx(element_boundary= True)
    else : return InnerProduct(curl(E).trans, dH)*dx -InnerProduct( (E*Cn).trans, Qn*dH*Pn)*dx(element_boundary= True)

def SymCurlTHcd2Hcc(dE,H, **kwargs):
    """
    calculates the symmetric curl transpose of a [HCurlDiv]^* function and a HCurlCurl function and results into a HCurlDiv function
    dE : Test function of [HCurlDiv]^*
    H : Trial function of HCurlCurl
    """
    return TCurHcc2Hcd(dE,H, **kwargs)

def DivHcdHd( H,dv, **kwargs):
    """
    calculates the divegence of a HCurlDiv function and results into a [HDiv]^* function
    H :  Test function of HCurlDiv
    dv : Trial function of [HDiv]^*
    """
    return div(H)*dv*dx - H*n*n * dv*n * dx(element_boundary= True)

def DevGradHd2Hcd(v,dH, **kwargs):
    """
    calculates the deviatoric gradient of a [HDiv]^* function and results into a HCurlDiv function
    v : Test function of [HDiv]^*
    dH : Trial function of HCurlDiv
    """
    return  DivHcdHd(dH,v)
    
def Curl(u):
    """
    calculates the curl of a vector valued  or  a matrixvalued function 
        u : function
    """
    if u.dim == 3:
        return CF( (u[2].Diff(y)- u[1].Diff(z), -u[2].Diff(x)+ u[0].Diff(z), -u[0].Diff(y)+ u[1].Diff(x)) )
    if u.dim == 9:
        return CF( (Curl(u[0,:]),Curl(u[1,:]),Curl(u[2,:])),dims=(3,3) )

def Grad(u):
    """
    calculates the gradient of a scalar valued function
        u : function
    """
    if u.dim == 3 : return CF( (u.Diff(x), u.Diff(y), u.Diff(z)) )
    if u.dim == 9 : return CF( (Grad(u[0,:]),Grad(u[1,:]),Grad(u[2,:])),dims=(3,3) )

def Div(u):
    """
    calculates the divergence of a vector valued function
        u : function
    """
    if u.dim == 3 : return CF( (u[0].Diff(x)+ u[1].Diff(y)+ u[2].Diff(z)) )
    if u.dim == 9 : return CF( (Div(u[0,:]),Div(u[1,:]),Div(u[2,:])) )

def Dev(u):
    """
    calculates the deviatoric part of a matrix valued function
        u : function
    """
    return CF( (u - 1/3*Trace(u)*Id(3) ), dims=(3,3) )

#def Trace(gf_u):
#    """
#    calculates the trace of a matrix valued function
#        gf_u : function
#    """
#    return CF( (gf_u[0,0]+ gf_u[1,1]+ gf_u[2,2]) )

def Transpose(A):
    """
    calculates the transpose of a matrix valued function
        A : function
    """
    return CF( (A[0,0], A[1,0], A[2,0], A[0,1], A[1,1], A[2,1], A[0,2], A[1,2], A[2,2]), dims=(3,3) )

 
def Test():

    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.2))

    order = 3
    fescc = HCurlCurl(mesh, order=order)

    fescd = HCurlDiv(mesh, order=order)
    fesd = HDiv(mesh, order=order)
    fescd_d = fescd*fesd

    E, dE = fescc.TnT()
    H, dH = fescd.TnT()
    v, dv = fesd.TnT()

    massE = BilinearForm(InnerProduct(E,dE)*dx).Assemble()
    massH = BilinearForm(InnerProduct(H,dH)*dx).Assemble()
    massv = BilinearForm(InnerProduct(v,dv)*dx).Assemble()

    bftcurl = BilinearForm(TCurHcc2Hcd(E, dH)).Assemble()
    bfdiv = BilinearForm(DivHcdHd(H, dv)).Assemble()
    bfdevgrad = BilinearForm(DevGradHd2Hcd(v, dH)).Assemble()
    bfsymcurlt = BilinearForm(SymCurlTHcd2Hcc(dE, H)).Assemble()

    massEinv = massE.mat.Inverse(inverse="sparsecholesky")
    massHinv = massH.mat.Inverse(inverse="sparsecholesky")
    massvinv = massv.mat.Inverse(inverse="sparsecholesky")    

    # test TCurHcc2Hcd
    gfE = GridFunction(fescc)
    gfH = GridFunction(fescd)

    sym1 = CF( ( (y,0,0), (0,0,0), (0,0,0)), dims=(3,3) )
    sym2 = CF( ( (0,0,0), (0,x*(1-x),0), (0,0,0)) , dims=(3,3))

    for sym in [sym1, sym2]:
        gfE.Set ( sym1 )
        gfH.vec.data = massHinv@bftcurl.mat * gfE.vec
        gfH_true = GridFunction(fescd)
        gfH_true.Set ( Transpose(Curl(sym1 )) )
        print ("TCurHcc2Hcd: ", sqrt (Integrate (InnerProduct(gfH-gfH_true, gfH-gfH_true), mesh)))
        Draw (gfH- gfH_true, mesh, "error")
        input("TCurHcc2Hcd")

    # test SymCurlTHcd2Hcc
    gfH = GridFunction(fescc)
    gfE = GridFunction(fescd)

    tracefree1 = CF( ( (0,0,x*(1-x)), (0,0,0), (x*(1-x),0,0)), dims=(3,3) )
    tracefree2 = CF( ( (0,z,0), (z,0,0), (0,0,0)), dims=(3,3) )

    for tracefree in [tracefree1, tracefree2]:
        gfH.Set ( tracefree )
        gfE.vec.data = massEinv@ bftcurl.mat.T * gfH.vec
        gfE_true = GridFunction(fescd)
        gfE_true.Set ( (Transpose(Curl(tracefree)) + Curl(tracefree))/2  ) 
        print ("SymCurlTHcd2Hcc: ", sqrt (Integrate (InnerProduct(gfE-gfE_true, gfE-gfE_true), mesh)))
        Draw (gfE- gfE_true, mesh, "error")
        input("SymCurlTHcd2Hcc")

    # test DivHcdHd
    gfH = GridFunction(fescd)
    gfV = GridFunction(fesd)

    div1 = CF( ( (x*y,0,0), (0,x*y,0), (0,0,z*z)) , dims=(3,3) )
    div2 = CF( ( (0,0,x*z), (0,0,0), (0,0,0)), dims=(3,3) )

    for div in [div1, div2]:
        gfH.Set ( div )
        gfV.vec.data = massvinv@ bfdiv.mat * gfH.vec
        gfV_true = GridFunction(fesd)
        gfV_true.Set ( Div(div) )
        print ("DivHcdHd: ", sqrt (Integrate (InnerProduct(gfV-gfV_true, gfV-gfV_true), mesh)))
        Draw (gfV- gfV_true, mesh, "error")
        input("DivHcdHd")


    # test DevGradHd2Hcd
    gfV = GridFunction(fesd)
    gfH = GridFunction(fescd)

    devgrad1 = CF((x*y,0,0))
    devgrad2 = CF((0,0,x*z) )

    for devgrad in [devgrad1, devgrad2]:
        gfV.Set ( devgrad )
        gfH.vec.data = massHinv@ bfdevgrad.mat * gfV.vec
        gfH_true = GridFunction(fescd)
        gfH_true.Set ( (Grad(devgrad) - Id(3)*Div(devgrad)/3) )
        print ("DevGradHd2Hcd: ", sqrt (Integrate (InnerProduct(gfH-gfH_true, gfH-gfH_true), mesh)))
        Draw (gfH- gfH_true, mesh, "error")
        input("DevGradHd2Hcd")




def main():
    Test()

if __name__ == "__main__":
    with TaskManager(): main()