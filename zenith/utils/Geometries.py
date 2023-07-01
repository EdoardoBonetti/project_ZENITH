
from ngsolve import *
from netgen.csg import *
from ngsolve.solvers import *

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
        curve_order [int] : the order of the mesh curves
    """

    grading = kwargs.get("grading", 0.9)
    geo = CSGeometry()

    sphere_inner = Sphere(Pnt(0,0,0), r).maxh(h)
    sphere_inner.bc("inner")
    geo.Add(sphere_inner)
    sphere_outer = Sphere(Pnt(0,0,0),R)
    sphere_outer.bc("outer")
    geo.Add(sphere_outer- sphere_inner)
    
    mesh = Mesh(geo.GenerateMesh(maxh=H, grading=grading))
    mesh.Curve(kwargs.get("curve_order",1))

    return mesh
