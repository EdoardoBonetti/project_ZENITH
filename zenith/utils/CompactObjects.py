# here we define all the compact objects that we want to use in the simulation such as neutron stars, black holes, etc.

from ngsolve import *
from netgen.csg import *



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

    


def MeshBlackHoles(
        blackholes : list[BlackHole],
        h : float =0.15, 
        r : float = 1,
        H : float =10,
        R : float =20, 
        **kwargs : dict 
        ) -> Mesh:
    
    """
    blackholes [list[BlackHole]] : list of black holes
    h [float] : mesh size of inner spheres
    r [float] : inner radious
    H [float] : mesh size of outer sphere
    R [float] : outer radious

    kwargs : 
        grading [float] : the grading of the mesh
        
    return [Mesh] : the mesh
    """

    grading = kwargs.get("grading", 0.9)

    geo = CSGeometry()
    totmass = 0
    dist = 0

    x_cm , y_cm, z_cm = 0,0,0
    for bh in blackholes:
            totmass += bh.fl_mass
            x_cm += bh.fl_mass*bh.fl_pos[0]
            y_cm += bh.fl_mass*bh.fl_pos[1]
            z_cm += bh.fl_mass*bh.fl_pos[2]
            dist = max(dist, sqrt((bh.fl_pos[0]-x_cm)**2 + (bh.fl_pos[1]-y_cm)**2 + (bh.fl_pos[2]-z_cm)**2))

    x_cm /= totmass
    y_cm /= totmass
    z_cm /= totmass

    sphere_inner = Sphere(Pnt(blackholes[0].fl_pos[0],blackholes[0].fl_pos[1],blackholes[0].fl_pos[2]) ,r*blackholes[0].fl_mass).maxh(h)
    for bh in blackholes[1:len(blackholes)]:
        sphere_inner += Sphere(Pnt(bh.fl_pos[0],bh.fl_pos[1],bh.fl_pos[2]),r*bh.fl_mass).maxh(h)
    sphere_inner.bc("inner")
    geo.Add(sphere_inner)    
   
    sphere_outer = Sphere(Pnt(x_cm,y_cm,z_cm),R)
    sphere_outer.bc("outer")
    geo.Add(sphere_outer- sphere_inner)

    mesh = Mesh(geo.GenerateMesh(maxh=H, grading=grading))
    mesh.Curve(kwargs.get("curve_order",1))

    return mesh

if __name__ == "__main__":
    # test the DefaultMesh function
    mass1 = 2
    pos1 = (-1,0,0)
    spin1 = (0,0,0)
    mom1 = (0,0,0)

    mass2 = 2
    pos2 = (1,0,0)
    spin2 = (0,0,0)
    mom2 = (0,0,0)

    bh1 = BlackHole(mass = mass1, pos = pos1, spin = spin1, mom = mom1)
    bh2 = BlackHole(mass = mass2, pos = pos2, spin = spin2, mom = mom2)

    blackholes = [bh1,bh2]

    mesh = MeshBlackHoles(blackholes)

    f = GridFunction(H1(mesh,order=0))
    f.Set((x))
    Draw(f, mesh, "f")

    
