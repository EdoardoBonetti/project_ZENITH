from cmath import sqrt


def SymplecticEulerStep(tau,u,Mu,v,Mv ):
    u.vec.data += tau*Mv.mat*v.vec
    v.vec.data += tau*Mu.mat*u.vec
    return u,v

def YoshidaIntegratorStep(tau,u,Mu,v,Mv):
    # https://en.wikipedia.org/wiki/Leapfrog_integration#cite_note-Yoshida1990-6
    # book yoshida symplectic
    w0 = - 2**-(1/3)/(2-2**-(1/3))
    w1 = 1/(2-2**-(1/3))

    c= [w0/2 , (w0+w1)/2  , (w0+w1)/2 , w0/2 ]
    d= [w1,w0,w1]

    u.vec.data += c[0]*tau*Mv.mat*v.vec
    v.vec.data += d[0]*tau*Mu.mat*u.vec

    u.vec.data += c[1]*tau*Mv.mat*v.vec
    v.vec.data += d[1]*tau*Mu.mat*u.vec

    u.vec.data += c[2]*tau*Mv.mat*v.vec
    v.vec.data += d[2]*tau*Mu.mat*u.vec 

    u.vec.data += c[3]*tau*Mv.mat*v.vec
    return u,v


def LeapFrogStep(tau,u,Mu,v,Mv, dual = False):
    # SE1
    u.vec.data += tau/2*Mv.mat*v.vec
    v.vec.data += tau/2*Mu.mat*u.vec
    # SE2
    v.vec.data += tau/2*Mu.mat*u.vec
    u.vec.data += tau/2*Mv.mat*v.vec

    return u,v

# the aim of this part is to deifne the a gauss method for the function of the form 
# :
# 
# now we define a Runge-Kutta implicit method with butcher tableau: c|a/b
def RungeKuttaStep(tau,u,Mu,v,Mv,c,a,b):
    pass
