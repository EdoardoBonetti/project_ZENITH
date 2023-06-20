import unittest
from ngsolve import *
from zenith import *

def Div(A):
    return CoefficientFunction( (A[0,0].Diff(x) + A[1,0].Diff(y) + A[2,0].Diff(z), A[0,1].Diff(x) + A[1,1].Diff(y) + A[2,1].Diff(z), A[0,2].Diff(x) + A[1,2].Diff(y) + A[2,2].Diff(z) ) )

class TestStringMethods(unittest.TestCase):

    def test_basic(self):
            
        with TaskManager():
            mesh = Mesh(unit_cube.GenerateMesh(maxh=0.4))
            self.eb = EinsteinBianchi(mesh)
            t = Parameter(0)
            E = CoefficientFunction( ( (sin(y-t),0,0), (0,0,0), (0,0,-sin(y-t)) ) , dims=(3,3) )
            B = CoefficientFunction( ( (0,0,- cos(y-t)), (0,0,0), (- cos(y-t),0,0) ) , dims=(3,3) )

            self.eb.SetInitialCondition(E,B)


        return self.assertTrue(True)
    

    def test_divfreeness(self):
        u00 = -16*x*y*z*exp(-x**2 - y**2 - z**2)
        u01 =8*z*(x**2 + z**2 - 2)*exp(-x**2 - y**2 - z**2)
        u02 = 8*y*(x**2 - z**2)*exp(-x**2 - y**2 - z**2)
        u10 = 8*z*(x**2 + z**2 - 2)*exp(-x**2 - y**2 - z**2)
        u11 = 0
        u12 = -8*x*(x**2 + z**2 - 2)*exp(-x**2 - y**2 - z**2)
        u20 = 8*y*(x**2 - z**2)*exp(-x**2 - y**2 - z**2)
        u21 = -8*x*(x**2 + z**2 - 2)*exp(-x**2 - y**2 - z**2)
        u22 = 16*x*y*z*exp(-x**2 - y**2 - z**2)
        IC = CoefficientFunction( ( (u00,u01,u02), (u10,u11,u12), (u20,u21,u22) ) , dims=(3,3) )
        mesh = Mesh(unit_cube.GenerateMesh(maxh=0.1))

        return self.assertEqual(Integrate(Norm(Div(IC)), mesh) ,0)
if __name__ == '__main__':
    unittest.main()