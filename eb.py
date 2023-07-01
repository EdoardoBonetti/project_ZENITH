from netgen.csg import *

from ngsolve import *
import matplotlib.pyplot as plt
import numpy as np

from ngsolve.internal import visoptions
from ngsolve.internal import viewoptions

visoptions.scalfunction='u:0'
visoptions.clipsolution = 'scal'
viewoptions.clipping.nx= 0
viewoptions.clipping.ny= 0
viewoptions.clipping.nz= -1
viewoptions.clipping.enable = 1

class BlackHole:
    def __init__(
                    self,
                    mass : float = 1,
                    pos   = (0,0,0) ,
                    mom   = (0,0,0) ,
                    spin  = (0,0,0)
                
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




def Save(name):
    data = np.loadtxt(name+".txt", delimiter=";", skiprows=1)
    plt.plot(data[:,0], data[:,1], label="E")
    plt.plot(data[:,0], data[:,2], label="B")
    plt.plot(data[:,0], data[:,3], label="trace")
    plt.plot(data[:,0], data[:,4], label="sym")
    plt.plot(data[:,0], data[:,5], label="v")
    # sum 
    plt.plot(data[:,0], [sqrt(data[i,1]**2 + data[i,2]**2+ data[i,5]**2 ) for i in range(len(data[:,0]))], label="E^2 + B^2 + v^2")
    plt.legend()
    # save the plot in a png file with high resolution
    plt.savefig(name+".png", dpi=300)


def Eval(h = 0.2, H=0.1, order=1 , t = 0 , tend = 1, dt = 0.001):
    # create  a black hole at the origin
    mass = 1
    bh = BlackHole(mass=mass)
    
    bh_list = [bh]
    mesh = DefaultMesh(h=h, R=1, r=0.5, H =H, grading=0.9)
    
    #Draw(mesh, clipping='z')
    
    n = specialcf.normal(3)
    
    def CurlTHcc2Hcd(E,dH):
        return InnerProduct(curl(E).trans, dH)*dx \
           +InnerProduct(Cross(E*n, n), dH*n)*dx(element_boundary= True)
    def DivHcdHd(B,dv):
        return div(B)*dv*dx - B*n*n * dv*n * dx(element_boundary= True)
    #order = 1
    
    fescc = HCurlCurl(mesh, order=order)
    fescd = HCurlDiv(mesh, order=order)
    fesd = HDiv(mesh, order=order, RT=True)
    
    E, dE = fescc.TnT()
    v, dv = fesd.TnT()
    B, dB = fescd.TnT()
    
    bfcurlT = BilinearForm(CurlTHcc2Hcd(E, dB)).Assemble()
    bfdiv = BilinearForm(DivHcdHd(B, dv)).Assemble()
    
    with TaskManager():
        massE = BilinearForm(InnerProduct(E,dE)*dx, condense=True).Assemble()
        # preE = Preconditioner(massE, "bddc", block=True, blocktype="edgepatch")
        matE = massE.mat
        preE = matE.CreateSmoother(fescd.FreeDofs(True), GS=False)
        massE.Assemble()
        matE = massE.mat
        
        massEinvSchur = CGSolver (matE, preE, printrates = False)
        ext = IdentityMatrix()+massE.harmonic_extension
        extT = IdentityMatrix()+massE.harmonic_extension_trans
        massEinv =  ext @ massEinvSchur @ extT + massE.inner_solve
        
    with TaskManager():
        
        massB = BilinearForm(InnerProduct(B,dB)*dx, condense=True).Assemble()
        #preB = Preconditioner(massB, "bddc", block=True, blocktype="edgepatch")
        matB = massB.mat
        preB = matB.CreateSmoother(fescc.FreeDofs(True), GS=False)
        #massB.Assemble()
        #massB.Assemble()
        #matB = massB.mat    
    
        # preH = matH.CreateSmoother(fescd.FreeDofs(True), GS=False)
    
        massBinvSchur = CGSolver (matB, preB, printrates = False)
        ext = IdentityMatrix()+massB.harmonic_extension
        extT = IdentityMatrix()+massB.harmonic_extension_trans
        massBinv =  ext @ massBinvSchur @ extT + massB.inner_solve
        
    with TaskManager():
        massv = BilinearForm(InnerProduct(v,dv)*dx, condense=True).Assemble()
        matv = massv.mat
        prev = matv.CreateSmoother(fesd.FreeDofs(True), GS=False)
        
        massvinvSchur = CGSolver (matv, prev, printrates = False)
        ext = IdentityMatrix()+massv.harmonic_extension
        extT = IdentityMatrix()+massv.harmonic_extension_trans
        massvinv =  ext @ massvinvSchur @ extT + massv.inner_solve
        
    gfE = GridFunction(fescc)
    gfB = GridFunction(fescd)
    gfv = GridFunction(fesd)
    
    E00 = 0
    E01 = 16*sqrt(610)*z*(5*y**2 + 5*z**2 - 1)*exp(-10*x**2 - 10*y**2 - 10*z**2)/61
    E02 = 16*sqrt(610)*y*(-5*y**2 - 5*z**2 + 1)*exp(-10*x**2 - 10*y**2 - 10*z**2)/61
    E10 = 16*sqrt(610)*z*(5*y**2 + 5*z**2 - 1)*exp(-10*x**2 - 10*y**2 - 10*z**2)/61
    E11 = -160*sqrt(610)*x*y*z*exp(-10*x**2 - 10*y**2 - 10*z**2)/61
    E12 = 80*sqrt(610)*x*(y**2 - z**2)*exp(-10*x**2 - 10*y**2 - 10*z**2)/61
    E20 = 16*sqrt(610)*y*(-5*y**2 - 5*z**2 + 1)*exp(-10*x**2 - 10*y**2 - 10*z**2)/61
    E21 = 80*sqrt(610)*x*(y**2 - z**2)*exp(-10*x**2 - 10*y**2 - 10*z**2)/61
    E22 = 160*sqrt(610)*x*y*z*exp(-10*x**2 - 10*y**2 - 10*z**2)/61
    E = CoefficientFunction( (E00, E01, E02, E10, E11, E12, E20, E21, E22) , dims=(3,3) )
    
    #Draw(Norm(E), mesh, clipping ={ "z":-1})
    gfE.Set( E, bonus_intorder=10)
    t = 0
    # tend = 5 * dt
    #scene = Draw(Norm(gfB), mesh, clipping={"z":-1})
    energyE = []
    energyB = []
    energytrace = []
    energySym = []
    energyv = []
    name = "tend"+str(tend)+"dt"+str(dt)+"h"+str(h)+"order"+str(order)
    # if there is a txt with name 'energy.txt' in the folder, it will be deleted
    if os.path.exists(name+".txt"):
        os.remove(name+".txt")

    scene = Draw(Norm(gfE), mesh, "E")
    input("press")
    while t < tend:
    
            
            hv = bfcurlT.mat * gfE.vec + bfdiv.mat.T * gfv.vec
            gfB.vec.data += dt * massBinv * hv
            gfv.vec.data += -dt * massvinv@bfdiv.mat * gfB.vec
            gfE.vec.data += -dt * massEinv@bfcurlT.mat.T * gfB.vec
            #scene.Redraw()
    
            energyE.append (sqrt(InnerProduct (gfE.vec, massE.mat * gfE.vec)))
            energyB.append (sqrt(InnerProduct (gfB.vec, massB.mat * gfB.vec)))   
            energytrace.append (Integrate ( Norm (Trace(gfE)), mesh ))
            energySym.append (Integrate ( Norm (gfB-(gfB.trans))/2, mesh ))
            energyv.append (sqrt(InnerProduct (gfv.vec, massv.mat * gfv.vec)))
            t += dt
            with open(name+".txt", "a") as myfile: 
                myfile.write(str(t) + ";" + str(energyE[-1]) + ";" + str(energyB[-1]) + ";" +  str(energyv[-1]) + ";" + str(energytrace[-1]) + ";" + str(energySym[-1]) + "\n")
            print ("t: ", round(t/tend*100, 4), "%"+" E", round(energyE[-1],4), " B", round(energyB[-1],4), " trace", round(energytrace[-1],4), " sym", round(energySym[-1],4), " v", round(energyv[-1],4), end="\r")
            
    #t =0
    #with open(name+".txt", "a") as myfile:
    #    for i in range(len(energyE)):
    #        t += dt 
    #        myfile.write(str(t) + ";" + str(energyE[i]) + ";" + str(energyB[i]) + ";" + str(energytrace[i]) + ";" + str(energySym[i]) + ";" + str(energyv[i]) + "\n")
    
    #Save(name)

if "__main__" == __name__:
    with TaskManager(): 
        Eval(  h = 0.1, order = 1, tend= 10, dt = 0.1)


