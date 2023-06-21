from ngsolve import *
from netgen.occ import *
from ngsolve.internal import SnapShot
from netgen import gui
import ngsolve.internal

shape = Box((-1,-1,-1), (1,1,1)) # create a box
geo = OCCGeometry(shape) # create a geometry from the box
mesh = Mesh(geo.GenerateMesh(maxh=0.4)) # create a mesh from the geometry
#Draw (mesh); # draw the mesh

order = 2
Hcc = HCurlCurl(mesh, order=order ,  dirichlet=".*") # for the gamma variable
Hdd = HDivDiv(mesh, order=order ) # , orderinner=order+1) fr t
# fesalpha = H1(mesh, order=order+1) # , dirichlet=".*")
Hc = HCurl(mesh, order=order) # for the div of a Hdd field

k , dk= Hdd.TnT() 
p, dp = Hc.TnT()
g, dg = Hcc.TnT()

n = specialcf.normal(3)
t = specialcf.tangential(3, True)
bbndtang  = specialcf.EdgeFaceTangentialVectors(3)
tef1 = bbndtang[:,0]
tef2 = bbndtang[:,1]
nu1 = Cross(tef1,t)
nu2 = Cross(tef2,t)

def IncOp(g, dg):
    return InnerProduct(curl(g), curl(dg).trans)*dx \
        + (curl(g)*n) * Cross (dg*n, n) * dx(element_vb=BND) \
        + (curl(dg)*n) * Cross (g*n, n) * dx(element_vb=BND) \
        + (g[nu1,t]*dg[t,tef1]-g[nu2,t]*dg[t,tef2])*dx(element_vb=BBND)

# print ( (gamma*nu1*t)*(dgamma*t*tef1))

def J(g): return g - 0.5*Trace(g)*Id(3)

def Constraint(k, p):
    return  div( k)*p * dx - InnerProduct(k*n - k[n,n]*n , p - (p*n)*n ) * dx(element_boundary = True)

def Curl(u):
    if u.dim == 3:
        return CF( (u[1].Diff(z)- u[2].Diff(y), u[2].Diff(x)- u[0].Diff(z), u[0].Diff(y)- u[1].Diff(x)) )
    if u.dim == 9:
        return CF( (Curl(u[0,:]),Curl(u[1,:]),Curl(u[2,:])),dims=(3,3) )

def Transpose(u):
    return CF( (u[0,0] , u[1,0] , u[2,0] , u[0,1] , u[1,1] , u[2,1] , u[0,2] , u[1,2] , u[2,2] ),dims=(3,3) )

def Symetric(u): return u + Transpose(u) 



MINC = BilinearForm(IncOp(g,dg))

Mcc = BilinearForm(InnerProduct(g, dg)*dx)
Mdd = BilinearForm(InnerProduct(k, dk)*dx)

Mccdd = BilinearForm(trialspace=Hcc, testspace=Hdd)
Mccdd += InnerProduct(g, dk)*dx

MJdd = BilinearForm(trialspace=Hdd, testspace=Hdd)
MJdd += InnerProduct(k, dk)*dx-0.5*Trace(k)*Trace(dk)*dx

with TaskManager():
    MINC.Assemble()
    Mcc.Assemble()
    Mdd.Assemble()
    Mccdd.Assemble()
    MJdd.Assemble()


fes = Hdd*Hc
k, p = fes.TrialFunction()
dk, dp = fes.TestFunction()

PROJ = BilinearForm(fes)
PROJ += InnerProduct(k, dk)*dx
PROJ += Constraint(k, dp)
PROJ += Constraint(dk, p)
PROJ += - 1e-6*(p*dp)*dx # I don't know why this is needed, 

MASS = BilinearForm(fes)
MASS += InnerProduct(k, dk)*dx
with TaskManager():
    PROJ.Assemble() 
    MASS.Assemble()


# 3 minutes for a cube of size 2 and a mesh of size 0.2 and order 3
with TaskManager():
    inv_PROJ = PROJ.mat.Inverse(inverse="sparsecholesky", freedofs=fes.FreeDofs())
    inv_Mcc = Mcc.mat.Inverse(inverse="sparsecholesky", freedofs=fes.FreeDofs())
    inv_Mdd = Mdd.mat.Inverse(inverse="sparsecholesky", freedofs=fes.FreeDofs())



gf_g = GridFunction(Hcc)
gf_inc_g = GridFunction(Hcc)

peak = exp(-10*(x*x+y*y+z*z))/1000
cf_gamma = CF((peak,0,0, 0,0,0,  0,0,0), dims=(3,3))
cf_symcurl_gamma = Symetric(Curl(cf_gamma))

#print(" total magnitude")
#Draw(InnerProduct(cf_symcurl_gamma,cf_symcurl_gamma), mesh, "SCGamma", clipping={"x": -1, "y": 0, "z": 0})
#print(" (x,x) component")
#Draw(InnerProduct(cf_symcurl_gamma[0,0],cf_symcurl_gamma[0,0]), mesh, "SCGamma", clipping={"x": -1, "y": 0, "z": 0})
#print(" (x,y) component")
#Draw(InnerProduct(cf_symcurl_gamma[0,1],cf_symcurl_gamma[0,1]), mesh, "SCGamma", clipping={"x": -1, "y": 0, "z": 0})
#print(" (x,z) component")
#Draw(InnerProduct(cf_symcurl_gamma[0,2],cf_symcurl_gamma[0,2]), mesh, "SCGamma", clipping={"x": -1, "y": 0, "z": 0})
#print(" (y,y) component")
#Draw(InnerProduct(cf_symcurl_gamma[1,1],cf_symcurl_gamma[1,1]), mesh, "SCGamma", clipping={"x": -1, "y": 0, "z": 0})
#print(" (y,z) component")
#Draw(InnerProduct(cf_symcurl_gamma[1,2],cf_symcurl_gamma[1,2]), mesh, "SCGamma", clipping={"x": -1, "y": 0, "z": 0})
#print(" (z,z) component")
#Draw(InnerProduct(cf_symcurl_gamma[2,2],cf_symcurl_gamma[2,2]), mesh, "SCGamma", clipping={"x": -1, "y": 0, "z": 0})




gf_g.Set( cf_symcurl_gamma, dual=True, bonus_intorder=12)

gf_k = GridFunction(Hdd)
gf_J_k = GridFunction(Hdd)


gf_proj = GridFunction(fes)
gf_proj_k , gf_proj_p = gf_proj.components


print ("gamma\n")
scene1 = Draw (sqrt(InnerProduct(gf_g,gf_g)), mesh, name = "gamma")#, draw_surf=False, clipping={"x": -1, "y": 0, "z": 0})
#print ("inc gamma\n")
#scene2 = Draw (InnerProduct(gf_inc_g,gf_inc_g), mesh, draw_surf=False, clipping={"x": -1, "y": 0, "z": 0})
#print ("kappa \n")
#scene3 = Draw (InnerProduct(gf_k, gf_k), mesh, draw_surf=False, clipping={"x": -1, "y": 0, "z": 0})
#print ("J kappa\n")
#scene4 = Draw (InnerProduct(gf_J_k, gf_J_k), mesh, draw_surf=False, clipping={"x": -1, "y": 0, "z": 0})
#print ("div kappa\n")
#scene5 = Draw (InnerProduct(div(gf_k), div(gf_k)), mesh, draw_surf=False, clipping={"x": -1, "y": 0, "z": 0})
Redraw(True)

Energy = []
dt = 0.1e-2
final_time = 0.1
with TaskManager():
  for i in range(round(final_time/dt)):
    NORM = Norm(gf_g.vec)
    Energy.append(NORM)
    if NORM > 0.01:
      break

    print ("t = ", i*dt, "norm = ",NORM)

    gf_J_k.vec.data = inv_Mdd @ MJdd.mat * gf_k.vec
    gf_g.vec.data += dt*inv_Mcc @ Mccdd.mat.T * gf_J_k.vec

    gf_inc_g.vec.data = inv_Mcc @ MINC.mat * gf_g.vec
    gf_k.vec.data -= dt*inv_Mdd @ Mccdd.mat * gf_inc_g.vec
    
    gf_proj_k.vec.data = gf_k.vec
    gf_proj.vec.data = inv_PROJ @ MASS.mat * gf_proj.vec
    gf_k.vec.data = gf_proj_k.vec

    #scene1.Redraw()
    SnapShot("IMAGES/image"+str(i))


    #if i % 10 == 0:
    #    scene1.Redraw()
    #    #scene2.Redraw()
    #    #scene3.Redraw()
    #    #scene4.Redraw()
    #    scene5.Redraw()


#print(" total magnitude")
#Draw(InnerProduct(gf_g,gf_g), mesh, "SCGamma", clipping={"x": -1, "y": 0, "z": 0})
#print(" (x,x) component")
#Draw(InnerProduct(gf_g[0,0],gf_g[0,0]), mesh, "SCGamma", clipping={"x": -1, "y": 0, "z": 0})
#print(" (x,y) component")
#Draw(InnerProduct(gf_g[0,1],gf_g[0,1]), mesh, "SCGamma", clipping={"x": -1, "y": 0, "z": 0})
#print(" (x,z) component")
#Draw(InnerProduct(gf_g[0,2],gf_g[0,2]), mesh, "SCGamma", clipping={"x": -1, "y": 0, "z": 0})
#print(" (y,y) component")
#Draw(InnerProduct(gf_g[1,1],gf_g[1,1]), mesh, "SCGamma", clipping={"x": -1, "y": 0, "z": 0})
#print(" (y,z) component")
#Draw(InnerProduct(gf_g[1,2],gf_g[1,2]), mesh, "SCGamma", clipping={"x": -1, "y": 0, "z": 0})
#print(" (z,z) component")
#Draw(InnerProduct(gf_g[2,2],gf_g[2,2]), mesh, "SCGamma", clipping={"x": -1, "y": 0, "z": 0})

import matplotlib.pyplot as plt
import numpy as np

TIME = np.linspace(0,final_time,len(Energy))
plt.plot(TIME,Energy)
