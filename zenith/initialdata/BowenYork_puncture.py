# in this file I will create the initial condition for the black hole inital value problem using puncture method
#
#       
#    here we quire the initial data to be:
# 1) conformally flat (i.e. the metric should be the Minkowski metric times a conformal factor)
# 2) maximal (i.e. the trace of the extrinsic curvature should be zero)
# 3) asymptotically flat (i.e. the metric should be the Minkowski metric at infinity)

from ngsolve import *
from netgen.csg import *
from ngsolve.solvers import *

from ngsolve.internal import visoptions
from ngsolve.internal import viewoptions

visoptions.scalfunction='u:0'
visoptions.clipsolution = 'scal'
viewoptions.clipping.nx= 0
viewoptions.clipping.ny= 0
viewoptions.clipping.nz= -1
viewoptions.clipping.enable = 1


sys.path.append("C:\\Users\\User\\OneDrive\\Desktop\\project_ZENITH")
from zenith.utils.Geometries import *
from zenith.utils.CompactObjects import *

# exec(open("Geometries.py").read())
# from Geometries import *


class BowenYork:
    """
    This class creates the initial data for the Bowen York initial value problem

    Parameters
    ----------
    BHs : list of BlackHole objects
    conformal_metric : ngsolve.CoefficientFunction
        the conformal metric, if not specified it is set to the Minkowski metric

    Global 
    ----------
    Xi : ngsolve.CoefficientFunction
        the conformal factor
    Aij : ngsolve.CoefficientFunction
        the extrinsic curvature
    b : ngsolve.CoefficientFunction
        the trace of the extrinsic curvature
    W : ngsolve.CoefficientFunction
        the conformal factor of the extrinsic curvature

    Attributes
    ----------
    BHs : list of BlackHole objects
    h : ngsolve.CoefficientFunction
        the conformal metric
    """

    def __init__(self, blackholes,  **kwargs):

        # list of black holes
        self.blackholes = blackholes

        if "conformal_metric" in kwargs:
            self.h = kwargs["conformal_metric"]
        else:
            self.h = CF((1,0,0, 0,1,0, 0,0,1), dims = (3,3))
    
        # unbouded initial function
        cf_invXi = CF((0))
        for bh in blackholes: cf_invXi += bh.mass/(2*bh.r)
        self.Xi = 1/cf_invXi
        self.Aij = CF((0,0,0, 0,0,0, 0,0,0), dims = (3,3)) # Total extrinsic curvature
        for bh in blackholes: self.Aij += bh.Aij
        self.b = -InnerProduct(self.Aij, self.Aij)*self.Xi**7 /8
        self.W = CF((0)) # not yet computed, is computed in the Solve() function

        #self.gf_u = CF((0))


    def Solve(self, FES,  **kwargs):
        inverse = kwargs.get("inverse", "sparsecholesky")
        #order = kwargs.get("order", 1)
        
        #FES = H1(mesh, order=order, dirichlet="outer")
        # in the space the outer boundary must be called "outer" and needs to ne Neumann
        u , v = FES.TnT()
        a = BilinearForm(FES)
        a += grad(u) * grad(v)*dx
        a += self.b*v/((1 + self.Xi * (u+1) )**7) *dx
        a += (u)/CF(sqrt(x*x + y*y + z*z)) * v*ds("outer")
        gf_u = GridFunction(FES)
        Newton(a,gf_u,freedofs=gf_u.space.FreeDofs(),maxit=100,maxerr=1e-15,inverse=inverse ,dampfactor=0.7,printing=True)
        mesh = FES.mesh
        Draw(gf_u, mesh, "u")
        
        self.W =  CF( (self.Xi *self.Xi *(CF((1)) + self.Xi * (gf_u+CF((1)) ) )**(-2)) )

    def GetSolution(self, gf_h, gf_Aij, gf_W, **kwargs):

        # if bonus_intorder is not given, we set it to 2
        bonus_intorder = kwargs.get("bonus_intorder", 2)

        #gf_Aij.Set(self.Aij* self.W* self.W , bonus_intorder = bonus_intorder)
        gf_Aij.Set(self.Aij, bonus_intorder = bonus_intorder)
        gf_h.Set(self.h, bonus_intorder = bonus_intorder)
        gf_W.Set(self.W, bonus_intorder = bonus_intorder)
        
    
    def Draw(self, mesh):
        Draw(self.Aij* self.W* self.W , mesh, "Aij_c")
        Draw(self.b, mesh, "b")
        Draw(self.Xi, mesh, "Xi")
        Draw(self.W, mesh, "W")



        
def Spinning(mesh, spin = (0,0,0),mass = 1,  **kwargs):
    s0 = spin[0]
    s1 = spin[1]
    s2 = spin[2]

    r = sqrt(x*x + y*y + z*z)
    J = sqrt(s0*s0 + s1*s1 + s2*s2)
    X = (x,y,z)
    costheta = (s0*X[0] + s1*X[1] + s2*X[2])/(r * sqrt(s0*s0 + s1*s1 + s2*s2) )
    AjiAij = 18*(s0*s0 + s1*s1 + s2*s2)* (1-costheta**2)/(r*r*r*r*r*r)
    psi20 = (1 + mass / (2*r) )**-5 * mass *((mass /2*r)**4 + 5 * (mass /2*r)**3 + 10 * (mass /2*r)**2 + 10 * (mass /2*r) + 5)/(16 *r)
    psi22 = 1/20*(1 + mass / (2*r) )**-5 * (mass /2*r)**2 *( 
        84 * (mass /2*r)**5
        + 368 * (mass /2*r)**4
        + 658 * (mass /2*r)**3
        + 539 * (mass /2*r)**2
        + 192* (mass /2*r)
        + 15
    ) + 21/ 5 *(mass / 2*r)**3 * log(mass /(2*r *(1 + mass / 2*r)))

    P0 = 1 
    P2 = ( 3* costheta* costheta - 1)/2
    psi2 = psi20*P0 + psi22*P2 
    psi = J*J/(mass**4 )*psi2 +1 + mass/(2*r)  
    W = 1/psi**2
    #Draw(W, mesh, "psi")
    u = J*J/(mass**4 )*psi2
    Draw(u, mesh, "u")

def Momenting(mesh, mom = (0,0,0),mass = 1,  **kwargs):
    s0 = mom[0]
    s1 = mom[1]
    s2 = mom[2]

    r = sqrt(x*x + y*y + z*z)
    P = sqrt(s0*s0 + s1*s1 + s2*s2)
    X = (x,y,z)
    costheta = (s0*X[0] + s1*X[1] + s2*X[2])/(r * sqrt(s0*s0 + s1*s1 + s2*s2) )
    AjiAij = 18*(s0*s0 + s1*s1 + s2*s2)* (1-costheta**2)/(r*r*r*r*r*r)
    psi20 = (1 + mass / (2*r) )**-5 * mass *((mass /2*r)**4 + 5 * (mass /2*r)**3 + 10 * (mass /2*r)**2 + 10 * (mass /2*r) + 5)/(16 *r)
    psi22 = 1/20*(1 + mass / (2*r) )**-5 * (mass /2*r)**2 *( 
        84 * (mass /2*r)**5
        + 368 * (mass /2*r)**4
        + 658 * (mass /2*r)**3
        + 539 * (mass /2*r)**2
        + 192* (mass /2*r)
        + 15
    ) + 21/ 5 *(mass / 2*r)**3 * log(mass /(2*r *(1 + mass / 2*r)))

    P0 = 1 
    P2 = ( 3* costheta* costheta - 1)/2
    psi2 = psi20*P0 + psi22*P2 
    psi = P*P/(mass**4 )*psi2 +1 + mass/(2*r)  
    W = 1/psi**2
    Draw(W, mesh, "psi")


def main():
    
    # create a mesh
    h = 0.1
    r = 0.75
    H = 0.5
    R = 7
    order = 2
    kwargs = {"bonus_intorder": 10, "inverse": "pardiso", "order": order, "mesh_order": 1}

    # BH1
    po1 =  (1,0,0)
    lin1 = (0,1,0)
    ang1 = (0,0,1)
    mass1 =1
    BH1 = BlackHole(mass1, po1, lin1, ang1 )

    # BH2
    po2 =  (-1,0,0)
    lin2 = (0,-1,0)
    ang2 = (0,0,1)
    mass2 =1
    BH2 = BlackHole(mass2, po2, lin2, ang2 )

    BHs = [BH1, BH2]
    mesh = MeshBlackHoles(BHs, h=h, R=R, H = H, r= r,  curve_order = 2)
    
    by = BowenYork(BHs)
    h1 = H1(mesh, order=order, dirichlet="outer")
    
    by.Solve(FES = h1, **kwargs)

    gf_h = GridFunction(MatrixValued(h1,3))
    gf_Aij = GridFunction(MatrixValued(h1,3))
    gf_W = GridFunction(h1)

    by.GetSolution(gf_h, gf_Aij, gf_W, **kwargs)

    by.Draw(mesh)


   

        



if __name__ == "__main__":
    with TaskManager():
        main()


