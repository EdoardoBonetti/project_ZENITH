# here we define all the compact objects that we want to use in the simulation such as neutron stars, black holes, etc.

from ngsolve import *


class BlackHole:
    def __init__(
                    self,
                    mass : float = 1,
                    pos  : tuple[float, ...] = (0,0,0) ,
                    mom  : tuple[float, ...] = (0,0,0) ,
                    spin : tuple[float, ...] = (0,0,0)
                
                )->None:
    
        """
        mass: [float] the mass of the black hole
        pos: [tuple[float, ...]] the position of the black hole
        mom: [tuple[float, ...]] the linear momentum of the black hole
        spin: [tuple[float, ...]] the spin of the black hole
        """


        self.fl_mass = mass
        self.fl_pos  = pos
        self.fl_spin = spin
        self.fl_mom  = mom

        self.mass = CF((mass))
        self.pos  = CF((pos))
        self.spin = CF((spin))
        self.mom  = CF((mom))



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


    def Draw(self, mesh):
        Draw(self.Aij, mesh, "Covariant-extrinsic-curvature")

    
