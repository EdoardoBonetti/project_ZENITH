
from ngsolve import *
from netgen.csg import *
from ngsolve.solvers import *

#from BowenYork import BlackHole
# 
# from ngsolve.internal import visoptions
# from ngsolve.internal import viewoptions
# 
# visoptions.scalfunction='u:0'
# visoptions.clipsolution = 'scal'
# viewoptions.clipping.nx= 0
# viewoptions.clipping.ny= 0
# viewoptions.clipping.nz= -1
# viewoptions.clipping.enable = 1




def DefaultMesh(
        h : float =0.15, 
        r : float = 1,
        H : float =10,
        R : float =20, 
        **kwargs : dict 
        ) -> Mesh:
    
    """
    h [float] : mesh size of inner sphere
    r [float] : inner radious
    H [float] : mesh size of outer sphere
    R [float] : outer radious
    kwargs : 
        grading [float] : the grading of the mesh   


    """

    #print (kwargs)
    grading = kwargs.get("grading", 0.9)
    
        # create the mesh in whick the inner sphere contains all the black holes
    geo = CSGeometry()
    totmass = 0
    dist = 0

    x_cm , y_cm, z_cm = 0,0,0
    sphere_inner = Sphere(Pnt(0,0,0), r).maxh(h)
    sphere_inner.bc("inner")
    geo.Add(sphere_inner)
    sphere_outer = Sphere(Pnt(x_cm,y_cm,z_cm),R)
    sphere_outer.bc("outer")
    geo.Add(sphere_outer- sphere_inner)
    
    mesh = Mesh(geo.GenerateMesh(maxh=H, grading=grading))
    mesh.Curve(kwargs.get("curve_order",1))

    return mesh


def AdaptMesh(
        h : float =0.15, 
        r : float = 1,
        H : float =10,
        R : float =20, 
        **kwargs : dict 
        ) -> Mesh:
    
    """
    h [float] : mesh size of inner spheres
    r [float] : inner radious
    H [float] : mesh size of outer sphere
    R [float] : outer radious

    kwargs : 
        grading [float] : the grading of the mesh
        blackholes [list] : a list of BlackHole objects
        adaption [bool] : if True, the mesh is adapted to each black hole
        
    returns a mesh for multiple blackholes:
        if 'adaption'== False : the mesh is the union of two spheres, 
                                the inner one is the inner sphere with boundary "inner",
                                the outer one is the outer sphere emulating the infinity with boundary "outer" 
        if 'adaption'== True : the mesh is the union of multiple spheres (one for each black hole),
                                the inner one is the inner sphere with boundary "inner",
                                the outer one is the outer sphere emulating the infinity with boundary "outer"
    """

    #print (kwargs)

    if "grading" in kwargs:
        grading = kwargs["grading"]
    else:
        grading = 0.9

    if "blackholes" in kwargs:
        BHs = kwargs["blackholes"]

        # create the mesh in whick the inner sphere contains all the black holes
        geo = CSGeometry()
        totmass = 0
        dist = 0

        x_cm , y_cm, z_cm = 0,0,0
        for BH in BHs:
            totmass += BH.fl_mass
            x_cm += BH.fl_mass*BH.fl_pos[0]
            y_cm += BH.fl_mass*BH.fl_pos[1]
            z_cm += BH.fl_mass*BH.fl_pos[2]
            dist = max(dist, sqrt((BH.fl_pos[0]-x_cm)**2 + (BH.fl_pos[1]-y_cm)**2 + (BH.fl_pos[2]-z_cm)**2))
        x_cm /= totmass
        y_cm /= totmass
        z_cm /= totmass

        if "adaption" in kwargs and kwargs["adaption"] == True:
            # create the inner sphere for each black hole

            sphere_inner = Sphere(Pnt(BHs[0].fl_pos[0],BHs[0].fl_pos[1],BHs[0].fl_pos[2]),3/2*BHs[0].fl_mass).maxh(h)
            for BH in BHs[1:len(BHs)]:
                sphere_inner += Sphere(Pnt(BH.fl_pos[0],BH.fl_pos[1],BH.fl_pos[2]),3/2*BH.fl_mass).maxh(h)
            sphere_inner.bc("inner")
            geo.Add(sphere_inner)    

        else:
            sphere_inner = Sphere(Pnt(x_cm,y_cm,z_cm),3/2*dist).maxh(h)
            sphere_inner.bc("inner")
            geo.Add(sphere_inner)
        
        sphere_outer = Sphere(Pnt(x_cm,y_cm,z_cm),R)
        sphere_outer.bc("outer")
        geo.Add(sphere_outer- sphere_inner)


        mesh = Mesh(geo.GenerateMesh(maxh=h*(R+2)/2, grading=grading))
        mesh.Curve(1)

        return mesh

    else:
        # create a mesh
        geo = CSGeometry()
        sphere_inner = Sphere(Pnt(0,0,0), r ).maxh(h)
        sphere_inner.bc("inner")
        geo.Add(sphere_inner)
        sphere_outer = Sphere(Pnt(0,0,0), R )
        sphere_outer.bc("outer")
        geo.Add(sphere_outer- sphere_inner)
        mesh = Mesh(geo.GenerateMesh(maxh=h*R/2, grading=grading))
        mesh.Curve(1)

        return mesh


if __name__ == "__main__":
    # test the DefaultMesh function
    mesh = DefaultMesh(grading =1)
    #Draw(mesh)
    f = GridFunction(H1(mesh,order=0))
    f.Set((x))
    Draw(f, mesh, "f")
