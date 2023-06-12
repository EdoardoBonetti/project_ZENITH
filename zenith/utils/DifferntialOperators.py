# here what I want to do is to create the solution for the following problem:
# given a function a and a function b I want to compute c = a*b
from ngsolve import *
import scipy.sparse as sp
import matplotlib.pylab as plt
from netgen.csg import *




class MultiplicationOperator:
    def __init__(self, mesh, order = 1, **kwargs):
        
        h1 = H1(mesh, order = order)


        space_a = h1
        space_b = h1

        space_ab = space_a * space_b

        space_c = h1

        (a , b) , (da , db) = space_ab.TnT()
        c , dc = space_c.TnT()

        mass = BilinearForm(space_c)
        mass += InnerProduct(c,dc)*dx
        mass.Assemble()

        inverse = kwargs.get("inverse", "pardiso")
        self.mass = mass
        self.invmass = mass.mat.Inverse(space_c.FreeDofs(), inverse)

        multBLF = BilinearForm(trialspace = space_ab, testspace = space_c)#, nonassemble = True)
        multBLF += a*a*dc*dx
        self.multBLF = multBLF

        #self.MultOp = self.invmass @ multBLF.mat

        self.res = GridFunction(space_ab)

    def Mult(self, a, b, c):
        self.res.components[0].vec.data =a.vec
        self.res.components[1].vec.data =b.vec
        self.MultOp.Apply(c.vec, self.res)

    def Plot(self):
        # plot the matrix MultOp
        self.multBLF.Assemble()
        A = sp.csr_matrix(self.multBLF.mat.CSR())
        plt.rcParams['figure.figsize'] = (4,4)
        plt.spy(A)
        plt.show()






def main():


    geo = CSGeometry()
    geo.Add ( OrthoBrick( Pnt(0,0,0), Pnt(1,1,1) ) )
    mesh = Mesh(geo.GenerateMesh(maxh=0.3))

    mult = MultiplicationOperator(mesh)

    mult.Plot()


if __name__ == "__main__":
    with TaskManager(): main()