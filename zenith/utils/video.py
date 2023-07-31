import asyncio
from ffmpeg import FFmpeg

ffmpeg = FFmpeg().option('y').input(
    'rtsp://example.com/cam',
    # Specify file options using kwargs
    rtsp_transport='tcp',
    rtsp_flags='prefer_tcp',
).output(
    'output.ts',
    # Use a dictionary when an option name contains special characters
    {'codec:v': 'copy'},
    f='mpegts',
)

@ffmpeg.on('start')
def on_start(arguments):
    print('Arguments:', arguments)

@ffmpeg.on('stderr')
def on_stderr(line):
    print('stderr:', line)

@ffmpeg.on('progress')
def on_progress(progress):
    print(progress)

@ffmpeg.on('progress')
def time_to_terminate(progress):
    # Gracefully terminate when more than 200 frames are processed
    if progress.frame > 200:
        ffmpeg.terminate()

@ffmpeg.on('completed')
def on_completed():
    print('Completed')

@ffmpeg.on('terminated')
def on_terminated():
    print('Terminated')

@ffmpeg.on('error')
def on_error(code):
    print('Error:', code)

asyncio.run(ffmpeg.execute())

###################

from ngsolve import *
from ngsolve.internal import SnapShot
from netgen import gui

# viscosity
nu = 0.001

# timestepping parameters
tau = 0.001
tend = 0.01

from netgen.geom2d import SplineGeometry
geo = SplineGeometry()
geo.AddRectangle( (0, 0), (2, 0.41), bcs = ("wall", "outlet", "wall", "inlet"))
geo.AddCircle ( (0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl", maxh=0.02)
mesh = Mesh( geo.GenerateMesh(maxh=0.07))

mesh.Curve(3)

V = VectorH1(mesh,order=3, dirichlet="wall|cyl|inlet")
Q = H1(mesh,order=2)

X = V*Q

u,p = X.TrialFunction()
v,q = X.TestFunction()

stokes = nu*InnerProduct(grad(u), grad(v))+div(u)*q+div(v)*p - 1e-10*p*q
a = BilinearForm(X, symmetric=True)
a += stokes*dx
a.Assemble()

# nothing here ...
f = LinearForm(X)   
f.Assemble()

# gridfunction for the solution
gfu = GridFunction(X)

# parabolic inflow at inlet:
uin = CoefficientFunction( (1.5*4*y*(0.41-y)/(0.41*0.41), 0) )
gfu.components[0].Set(uin, definedon=mesh.Boundaries("inlet"))

# solve Stokes problem for initial conditions:
inv_stokes = a.mat.Inverse(X.FreeDofs(), inverse='pardiso')

res = f.vec.CreateVector()
res.data = f.vec - a.mat*gfu.vec
gfu.vec.data += inv_stokes * res


# matrix for implicit Euler 
mstar = BilinearForm(X, symmetric=True)
mstar += (u*v + tau*stokes)*dx
mstar.Assemble()
inv = mstar.mat.Inverse(X.FreeDofs(), inverse="sparsecholesky")

# the non-linear term 
conv = BilinearForm(X, nonassemble = True)
conv += (grad(u) * u) * v * dx

# for visualization
Draw (Norm(gfu.components[0]), mesh, "velocity", sd=3)

# implicit Euler/explicit Euler splitting method:
t = 0
i = 0
Redraw(True)
from ngsolve.internal import SnapShot
input("adjust window size, press enter to start")
#with TaskManager():
while t < tend:
        print("t=", t)#, end="\r")

        conv.Apply (gfu.vec, res)
        res.data += a.mat*gfu.vec
        gfu.vec.data -= tau * inv * res    

        t = t + tau
        Redraw(True)
        #SnapShot(f"image_{i:04}")
        SnapShot("image_"+str(i))
        Redraw(True)
        i+=1

input("press any key to stop")