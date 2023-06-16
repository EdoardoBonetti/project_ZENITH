from ngsolve import *
from netgen.csg import *

import scipy.sparse as sp
import matplotlib.pylab as plt

from ngsolve.internal import visoptions
from ngsolve.internal import viewoptions


def VisualOptions():
    visoptions.scalfunction='u:0'
    visoptions.clipsolution = 'scal'
    viewoptions.clipping.nx= 0
    viewoptions.clipping.ny= 0
    viewoptions.clipping.nz= -1
    viewoptions.clipping.enable = 1
    return None

def Trace(gf_u):
    return CF((gf_u[0,0]+ gf_u[1,1]+ gf_u[2,2]))

def Transpose(A):
    return CF( (A[0,0], A[1,0], A[2,0], A[0,1], A[1,1], A[2,1], A[0,2], A[1,2], A[2,2]), dims=(3,3) )

def TestCorrectness(gf_u, mesh, divergence = True):
    print("Trace of gf_u:")
    print(Integrate(Norm(Trace(gf_u)), mesh))
    
    print("Sym of gf_u:")
    print(Integrate(Norm(gf_u - Transpose(gf_u) ), mesh))

    if divergence == "Hdc":
        print("Div of gf_u:")
        print(Integrate(Norm(div(gf_u)), mesh))

def Curl(u):
    if u.dim == 3:
        return CF( (u[2].Diff(y)- u[1].Diff(z), -u[2].Diff(x)+ u[0].Diff(z), -u[0].Diff(y)+ u[1].Diff(x)) )
    if u.dim == 9:
        return CF( (Curl(u[0,:]),Curl(u[1,:]),Curl(u[2,:])),dims=(3,3) )


def Plot(a):
    
    A = sp.csr_matrix(a.mat.CSR())
    plt.spy(A)
    plt.show()


def main(h = 0.3, **kwargs):
    order = kwargs.get("order",1)   
    omega = kwargs.get("omega",1)

    innersphere = Sphere(Pnt(0,0,0) , 1).maxh(h)
    outersphere = Sphere(Pnt(0,0,0) , 10)
    geo = CSGeometry()
    geo.Add (innersphere)
    geo.Add (outersphere-innersphere)
    mesh = Mesh(geo.GenerateMesh(maxh=20*h) )




    Hcc = HCurlCurl(mesh, order=order)#, dirichlet=".*")
    Hdc = HCurlDiv(mesh, order=order) #, dirichlet=".*")
    Hc = HDiv(mesh, order=order,RT = False )#, dirichlet=".*")

    f = exp(-x*x-y*y-z*z)

    B_0 = CF( ( f, 0, 0 , 0, 0, 0, 0, 0, -f) , dims=(3,3) ) 
    #E_0 = CF( ( 0, 0, 0 , 0, 0, 0, 0, 0, 0) , dims=(3,3) ) 
    E_0 = CF( ( 0, 0, f , 0, 0, 0, f, 0, 0) , dims=(3,3) )


    fes = Hcc * Hdc * Hc
    (u, v, p) = fes.TrialFunction()
    (du, dv, dp) = fes.TestFunction()

    n = specialcf.normal(3)
    Cn = CoefficientFunction( (0, n[2], -n[1], -n[2], 0, n[0], n[1], -n[0], 0) , dims=(3,3) )
    Pn = OuterProduct(n,n)
    Qn = Id(3) - Pn

    Hccdofs = fes.Range(0)
    emb_Hcc = Embedding(fes.ndof, Hccdofs)
    u, du = Hcc.TnT()
    massHcc = BilinearForm(Hcc)
    massHcc += InnerProduct(u,du)*dx
    massHcc.Assemble()
    invmassHcc = massHcc.mat.Inverse( inverse="sparsecholesky")
    invHcc = emb_Hcc @ invmassHcc @ emb_Hcc.T

    Hdcdofs = fes.Range(1)
    emb_Hdc = Embedding(fes.ndof, Hdcdofs)
    v, dv = Hdc.TnT()
    massHdc = BilinearForm(Hdc)
    massHdc += InnerProduct(v,dv)*dx
    massHdc.Assemble()
    invmassHdc = massHdc.mat.Inverse( inverse="sparsecholesky")
    invHdc = emb_Hdc @ invmassHdc @ emb_Hdc.T
    
    Hdofs = fes.Range(2)
    emb_H = Embedding(fes.ndof, Hdofs)
    p, dp = Hc.TnT()
    massH = BilinearForm(Hc)
    massH += InnerProduct(p,dp)*dx
    massH.Assemble()
    invmassH = massH.mat.Inverse( inverse="sparsecholesky")
    invH = emb_H @ invmassH @ emb_H.T

    print("curl operator Hcd -> Hcc: from E to B:")
    u ,du= Hcc.TnT()
    v, dv = Hdc.TnT()
    b = BilinearForm(testspace =Hcc, trialspace=Hdc, nonassemble = True)
    b +=  InnerProduct(curl(du),(v) )*dx +InnerProduct(Cn*du*Pn, v )*dx(element_boundary= True)
    B = emb_Hcc @ b.mat @ emb_Hdc.T

    print("curl operator Hcc -> Hcd:")
    BT = emb_Hdc @ b.mat.T @ emb_Hcc.T

    print("div operator Hcd -> Hc:")
    
    u ,du= Hc.TnT()
    v, dv = Hdc.TnT()
    d = BilinearForm(testspace =Hc, trialspace=Hdc, nonassemble = True)
    d +=  InnerProduct(du,div(v) )*dx 
    d += - InnerProduct(du*n, (v*n)*n )*dx(element_boundary= True)
    D = emb_H @ d.mat @ emb_Hdc.T
    
    print("div operator Hc -> Hcd:")
    DT = emb_Hdc @ d.mat.T @ emb_H.T


    gf = GridFunction(fes)
    gf.components[0].Set(E_0, dual = True , bonus_intorder=10)
    gf.components[1].Set(B_0,  dual = True, bonus_intorder=10)

    Draw(gf.components[0], mesh, "E")

    Eneries ={  "Energy_E" : [] ,
                "Energy_B" : [] ,
                "Energy_Trace_E" : [] ,
                "Energy_Sym_B" : [] ,
                "Energy_Div_B" : [] ,
             }
    
    def CalcEnergy(u, Enegies, mesh):
        Eneries["Energy_E"].append( Integrate(Norm(u.components[0]), mesh) )
        Eneries["Energy_B"].append( Integrate(Norm(u.components[1]), mesh) )
        Eneries["Energy_Trace_E"].append( Integrate(Norm(Trace(u.components[0])), mesh) )
        Eneries["Energy_Sym_B"].append( Integrate(Norm(u.components[1] - Transpose(u.components[1]) ), mesh) )
        Eneries["Energy_Div_B"].append( Integrate(Norm(u.components[2]), mesh) )
    
    CalcEnergy(gf, Eneries, mesh)

    dt = 0.02
    for i in range(100):
        gf.vec.data += dt * invH @ D * gf.vec
        gf.vec.data += -dt * invHcc @ B * gf.vec    
        gf.vec.data +=  dt * invHdc @ (BT - DT)  * gf.vec
        #gf.vec.data +=  dt * invHdc @ (BT)  * gf.vec
        

        Draw(gf.components[0], mesh, "E")
        Draw(gf.components[1], mesh, "B")
        Draw(gf.components[2], mesh, "p")
        CalcEnergy(gf, Eneries, mesh)
        print("time : "+ str(i*dt) , end="\r")

    Draw(Norm(gf.components[0]), mesh, "E")
    Draw(Norm(gf.components[1]), mesh, "B")

    plt.plot(Eneries["Energy_E"], label = "Energy_E")
    plt.plot(Eneries["Energy_B"], label = "Energy_B")
    plt.plot(Eneries["Energy_Trace_E"], label = "Energy_Trace_E")
    plt.plot(Eneries["Energy_Sym_B"], label = "Energy_Sym_B")
    plt.plot(Eneries["Energy_Div_B"], label = "Energy_Div_B")
    plt.legend()
    plt.show()


    




if __name__ == '__main__':
    VisualOptions()
    with TaskManager():
        #curlfromHccToHdc(h = 0.2, order=1, omega=1)
        #curlfromHdcToHcc(h = 0.2, order=0, omega=1, divergence = False)
        main(h = 0.6, order=1, omega=1)