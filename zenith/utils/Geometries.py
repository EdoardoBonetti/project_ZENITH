
from ngsolve import *
from netgen.csg import *
from ngsolve.solvers import *

#from BowenYork import BlackHole

from ngsolve.internal import visoptions
from ngsolve.internal import viewoptions

visoptions.scalfunction='u:0'
visoptions.clipsolution = 'scal'
viewoptions.clipping.nx= 0
viewoptions.clipping.ny= 0
viewoptions.clipping.nz= -1
viewoptions.clipping.enable = 1



def DefaultMesh(h=0.15, R=20, **kwargs):
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
            totmass += BH.mass
            x_cm += BH.mass*BH.pos[0]
            y_cm += BH.mass*BH.pos[1]
            z_cm += BH.mass*BH.pos[2]
            dist = max(dist, sqrt((BH.pos[0]-x_cm)**2 + (BH.pos[1]-y_cm)**2 + (BH.pos[2]-z_cm)**2))
        x_cm /= totmass
        y_cm /= totmass
        z_cm /= totmass

        sphere_inner = Sphere(Pnt(x_cm,y_cm,z_cm),dist).maxh(h)
        sphere_inner.bc("inner")
        geo.Add(sphere_inner)
        
        sphere_outer = Sphere(Pnt(x_cm,y_cm,z_cm),R)
        sphere_outer.bc("outer")
        geo.Add(sphere_outer- sphere_inner)
        mesh = Mesh(geo.GenerateMesh(maxh=h*R/2, grading=grading))
        mesh.Curve(1)
            

    else:
        # create a mesh
        geo = CSGeometry()
        sphere_inner = Sphere(Pnt(0,0,0),1).maxh(h)
        sphere_inner.bc("inner")
        geo.Add(sphere_inner)
        sphere_outer = Sphere(Pnt(0,0,0),R)
        sphere_outer.bc("outer")
        geo.Add(sphere_outer- sphere_inner)
        mesh = Mesh(geo.GenerateMesh(maxh=h*R/2, grading=grading))
        mesh.Curve(1)

        return mesh


