# here what I want to do is to create the solution for the following problem:
# given a function a and a function b I want to compute c = a*b
from ngsolve import *
import scipy.sparse as sp
import matplotlib.pylab as plt
from netgen.csg import *




class MultiplicationOperator:
    def __init__(self, gf_f_space, gf_g_space, gf_h_space, **kwargs):

        self.space_a = gf_f_space
        self.space_b = gf_g_space
        self.space_c = gf_h_space
        #print("space_a", self.space_a)
        #print("space_b", self.space_b)

        self.space_ab = self.space_a * self.space_b
        print("space_ab", self.space_ab)


        (a , b) , (da , db) = self.space_ab.TnT()
        c , dc = self.space_c.TnT()
#
        self.mass = BilinearForm(self.space_c)
        self.mass += InnerProduct(c,dc)*dx
        self.mass.Assemble()

#
        inverse = kwargs.get("inverse", "pardiso")
        self.invmass = self.mass.mat.Inverse(self.space_c.FreeDofs(), inverse)

        multBLF = BilinearForm(trialspace = self.space_ab, testspace = self.space_c, nonassemble = True)
        multBLF += a*b*dc*dx
        self.multBLF = multBLF

        self.MultOp = self.invmass @ multBLF.mat

        self.res = GridFunction(self.space_ab)



    def Mult(self, a, b, c):
        self.res.components[0].vec.data =a.vec
        self.res.components[1].vec.data =b.vec

        # self.MultOp.Apply(c.vec, self.res)
        c.vec.data = self.MultOp * self.res.vec

#    def Plot(self):
#        # plot the matrix MultOp
#        self.multBLF.Assemble()
#        A = sp.csr_matrix(self.multBLF.mat.CSR())
#        plt.rcParams['figure.figsize'] = (4,4)
#        plt.spy(A)
#        plt.show()

    def Draw(self, mesh):

        Draw(self.res.components[0], mesh, "res")






def main():


    geo = CSGeometry()
    geo.Add ( OrthoBrick( Pnt(0,0,0), Pnt(1,1,1) ) )
    mesh = Mesh(geo.GenerateMesh(maxh=0.3))

    
    # def gridfunction(mesh, order = 1):
    f = CoefficientFunction( (x-0.5)*(y-0.5) )
    g = CoefficientFunction( (x-0.5)*(y-0.5) )

    h1 = H1(mesh, order = 1)
    gf_f = GridFunction(h1)
    gf_f.Set( f )
    gf_g = GridFunction(h1)
    gf_g.Set( g )
    
    gf_h = GridFunction(h1) 
    cf_h = CF(f*g)

    mult = MultiplicationOperator(gf_f.space, gf_g.space, gf_h.space)

    mult.Mult(gf_f, gf_g, gf_h)

    mult.Draw(mesh)

    Draw(gf_h, mesh, "gf_h")

    print("L2 error = ", sqrt(Integrate(Norm(gf_h - cf_h), mesh)))

    Draw(gf_h - cf_h, mesh, "error")

if __name__ == "__main__":
    with TaskManager(): main()