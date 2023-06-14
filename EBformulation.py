from ngsolve import *
from netgen.csg import *
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

def curlfromHccToHdc(h=0.4, **kwargs):
    order = kwargs.get("order",1)   
    omega = kwargs.get("omega",1)
    print("bug 1")
    cube = OrthoBrick(Pnt(0,0,0), Pnt(pi,pi,pi))
    geo = CSGeometry()
    geo.Add (cube)
    mesh = Mesh(geo.GenerateMesh(maxh=h) )

    Hcc = HCurlCurl(mesh, order=order)#, dirichlet=".*")
    Hdc = HCurlDiv(mesh, order=order)#, dirichlet=".*")
    Hc = HCurl(mesh, order=order)#, dirichlet=".*")

    t = Parameter(0)
    f = CF( (   -sin(omega*(y-t)),0,0, 
                0,0,0,
                0,0,sin(omega*(y-t))) , dims=(3,3))
    g = omega*CF( (0,0,cos(omega*(y-t)), 0,0,0, cos(omega*(y-t)),0,0) , dims=(3,3))

    # we need to prove that curl(f) = g in a discrete sense

    gf = GridFunction(Hcc)
    gf.Set(f)

    u, du = Hcc.TnT()
    v, dv = Hdc.TnT()
    p, dp = Hc.TnT()

    n = specialcf.normal(3)
    Cn = CoefficientFunction( (0, n[2], -n[1], -n[2], 0, n[0], n[1], -n[0], 0) , dims=(3,3) )
    Pn = OuterProduct(n,n)
    Qn = Id(3) - Pn     
    OpCurl = BilinearForm(testspace=Hdc, trialspace=Hcc)

    OpCurl += InnerProduct(curl(u),dv )*dx +InnerProduct(Cn*u*Pn, dv )*dx(element_boundary= True)
    OpCurl.Assemble()

    mass = BilinearForm(Hdc*Hc , symmetric=True, condense=True)
    mass += InnerProduct(v,dv)*dx
    #mass.Assemble()

    #OpDiv = BilinearForm( testspace=Hc, trialspace=Hdc)
    mass += InnerProduct(div(v),dp)*dx + InnerProduct(n*dp, (v*n)*n)*dx(element_boundary=True)
    #OpDiv.Assemble()
    mass += InnerProduct(div(dv),p)*dx + InnerProduct(n*p, (dv*n)*n)*dx(element_boundary=True)
    mass.Assemble()

    inv = mass.mat.Inverse(freedofs = (Hdc*Hc).FreeDofs(coupling = True), inverse="sparsecholesky")

    gf_NUMERICAL_curl = GridFunction(Hdc)
    gf_NUMERICAL_curl. Set ( g , definedon=mesh.Boundaries(".*") )
    gf_NUMERICAL_curl.vec.data = inv @ OpCurl.mat * gf.vec

    gf_EXACT_curl = GridFunction(Hdc)
    gf_EXACT_curl.Set(g )
    error = gf_EXACT_curl - gf_NUMERICAL_curl

    #Draw(f,mesh,"f")
    #Draw(g,mesh,"g")
    Draw(gf_NUMERICAL_curl ,mesh,"numerical_curl")
    Draw(gf_EXACT_curl     ,mesh,"exact_curl")
    Draw(error             ,mesh,"error")
    print("error = ", sqrt(Integrate(InnerProduct(error,error),mesh)))

    TestCorrectness(gf_NUMERICAL_curl, mesh)

def curlfromHdcToHcc(h=0.4, **kwargs):
    order = kwargs.get("order",1)   
    omega = kwargs.get("omega",1)

    cube = OrthoBrick(Pnt(0,0,0), Pnt(pi,pi,pi))
    geo = CSGeometry()
    geo.Add (cube)
    mesh = Mesh(geo.GenerateMesh(maxh=h) )

    Hcc = HCurlCurl(mesh, order=order)#, dirichlet=".*")
    Hdc = HCurlDiv(mesh, order=order)#, dirichlet=".*")


    t = Parameter(0)
    f = CF( (   -sin(omega*(y-t)),0,0, 
                0,0,0,
                0,0,sin(omega*(y-t))) , dims=(3,3))
    g = omega*CF( (0,0,cos(omega*(y-t)), 0,0,0, cos(omega*(y-t)),0,0) , dims=(3,3))

    # we need to prove that curl(f) = g in a discrete sense

    gf = GridFunction(Hdc)
    gf.Set(f)

    u, du = Hcc.TnT()
    v, dv = Hdc.TnT()

    n = specialcf.normal(3)
    Cn = CoefficientFunction( (0, n[2], -n[1], -n[2], 0, n[0], n[1], -n[0], 0) , dims=(3,3) )
    Pn = OuterProduct(n,n)
    Qn = Id(3) - Pn     
    OpCurl = BilinearForm(testspace=Hdc, trialspace=Hcc)

    OpCurl += InnerProduct(curl(u),dv )*dx +InnerProduct(Cn*u*Pn, dv )*dx(element_boundary= True)
    OpCurl.Assemble()

    mass = BilinearForm(Hcc)
    mass += InnerProduct(u,du)*dx
    mass.Assemble()

    inv = mass.mat.Inverse(Hcc.FreeDofs(), inverse="sparsecholesky")

    gf_NUMERICAL_curl = GridFunction(Hcc)
    gf_NUMERICAL_curl. Set ( g , definedon=mesh.Boundaries(".*") )
    gf_NUMERICAL_curl.vec.data = inv @ OpCurl.mat.T * gf.vec

    gf_EXACT_curl = GridFunction(Hcc)
    gf_EXACT_curl.Set(g )
    error = gf_EXACT_curl - gf_NUMERICAL_curl

    #Draw(f,mesh,"f")
    #Draw(g,mesh,"g")
    Draw(Norm(gf_NUMERICAL_curl),mesh,"numerical_curl")
    Draw(Norm(gf_EXACT_curl),mesh,"exact_curl")
    Draw(Norm(error),mesh,"error")
    print("error = ", sqrt(Integrate(InnerProduct(error,error),mesh)))




if __name__ == '__main__':
    VisualOptions()
    with TaskManager():
        curlfromHccToHdc(h = 0.2, order=1, omega=1)
        #curlfromHdcToHcc(h = 0.2, order=0, omega=1, divergence = False)