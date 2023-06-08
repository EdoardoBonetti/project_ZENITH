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


sys.path.append(os.getcwd())
from zenith.utils.Geometries import *

# exec(open("Geometries.py").read())
# from Geometries import *


class BlackHole:
    def __init__(self, mass = CF((1)),  pos= CF((0,0,0)), mom = CF((0,0,1)), spin= CF((0,0,0)) ):
        self.mass = mass
        self.pos = pos
        self.spin = spin
        self.mom = mom

        # here the tensors of the traceless extrinsic curvature given by the linear momentum and the spin
        self.r = CF((sqrt((x-self.pos[0])**2 + (y-self.pos[1])**2 + (z-self.pos[2])**2)))
        self.n = CF(((x-self.pos[0])/self.r, (y-self.pos[1])/self.r, (z-self.pos[2])/self.r))
        Pn = OuterProduct(self.mom, self.n)
        nn = OuterProduct(self.n, self.n)
        def Transpose(A): return CF(( A[0,0],A[1,0],A[2,0] , A[0,1],A[1,1],A[2,1] , A[0,2],A[1,2],A[2,2] ), dims = (3,3))
        def Trace(A): return CF((A[0,0] + A[1,1] + A[2,2]))
        
        AijP = 3*(Pn + Transpose(Pn) - (Id(3) - nn)*Trace(Pn))/(2*self.r**2)
        AijS = 3*( OuterProduct(Cross(self.n, self.spin), self.n) + OuterProduct(self.n, Cross(self.n, self.spin)) )/(2*self.r**3)
        self.Aij = AijP + AijS
    

class BowenYork:

    def __init__(self, BHs, **kwargs):

        # list of black holes
        self.BHs = BHs

        # conformal metric
        # if in kwargs there is a conformal metric, we use it, otherwise we set it to the flat metric
        
        if "conformal_metric" in kwargs:
            self.h = kwargs["conformal_metric"]
        else:
            self.h = CF((1,0,0, 0,1,0, 0,0,1), dims = (3,3))
    
        # unbouded initial function
        cf_invXi = CF((0))
        for BH in BHs: cf_invXi += BH.mass/(2*BH.r)
        self.Xi = 1/cf_invXi
        
        self.Aij = CF((0,0,0, 0,0,0, 0,0,0), dims = (3,3)) # Total extrinsic curvature
        for BH in BHs:    self.Aij += BH.Aij
        
        self.b = -InnerProduct(self.Aij, self.Aij)*self.Xi**7 /8
        
        self.W = CF((0)) # not yet computed, is computed in the Solve() function
        
    def Solve(self ,FES ,  inverse = "pardiso"):
        # in the space the outer boundary must be called "outer" and needs to ne Neumann
        u , v = FES.TnT()
        a = BilinearForm(FES)
        a += grad(u) * grad(v)*dx
        a += self.b*v/((1 + self.Xi * (u+1) )**7) *dx
        a += (u)/CF(sqrt(x*x + y*y + z*z)) * v*ds("outer")
        gf_u = GridFunction(FES)
        Newton(a,gf_u,freedofs=gf_u.space.FreeDofs(),maxit=100,maxerr=1e-15,inverse=inverse ,dampfactor=0.7,printing=True)
        self.W =  CF( (self.Xi *self.Xi *(1 + self.Xi * (gf_u+1) )**(-2)) )

    def InitialConditions(self, gf_h, gf_Aij, gf_W, **kwargs):

        # if bonus_intorder is not given, we set it to 2
        bonus_intorder = kwargs.get("bonus_intorder", 2)

        gf_Aij.Set(self.Aij* self.W* self.W , bonus_intorder = bonus_intorder)
        gf_h.Set(self.h, bonus_intorder = bonus_intorder)
        gf_W.Set(self.W, bonus_intorder = bonus_intorder)
        
    
    def Draw(self, mesh):
        Draw(self.Aij, mesh, "Aij")
        Draw(self.b, mesh, "b")
        Draw(self.Xi, mesh, "Xi")
        Draw(self.W, mesh, "W")

def main():
    
    # create a mesh
    h = 0.2
    R = 6
    order = 3
    kwargs = {"bonus_intorder": 10}

    mesh = DefaultMesh(h=h, R=R)
    h1 = H1(mesh, order = order)

    # create a list of black holes

    # BH1
    po1 =  CF((0,0,0))
    lin1 = CF((1,0,0))
    ang1 = CF((0,0,0))
    mass1 = CF((1))
    BH1 = BlackHole(mass1, po1, lin1, ang1 )

    # BH2
    #po2 =  CF((-1,0,0))
    #lin2 = CF((1,0,0))
    #ang2 = CF((0,0,0))
    #mass2 = CF((1))
    #BH2 = BlackHole(mass2, lin2, ang2, po2)

    # list: BHs
    BHs = [BH1]#,BH2]

    # solve the problem
    bowenyork = BowenYork(BHs) # set up the problem
    bowenyork.Solve(h1)        # solve the pde
    #bowenyork.Draw(mesh)

    # create the gridfunctions to store the initial data
    gf_h = GridFunction(MatrixValued(h1,3))
    gf_Aij = GridFunction(MatrixValued(h1,3))
    gf_W = GridFunction(h1)

    # compute the initial data
    bowenyork.InitialConditions(gf_h, gf_Aij, gf_W, **kwargs)

    # plot the initial data


if __name__ == "__main__":
    with TaskManager():
        main()
