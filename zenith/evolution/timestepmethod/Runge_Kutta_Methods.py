import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

# create a function that given a butcher tableau tells you if it is explicit = True or implicit = False
def isExplicit(a):
    for i in np.arange(len(a)):
        for j in np.arange(i,len(a)):
            if a[i,j] != 0:
                return False
    return True

def vec(M): # matbe it need to be transposed before
    #return np.reshape(M.transpose(),(len(M)*len(M[0]),1))
    return M.flatten("F")

def vecInverse(v , row): 
    # returs the transpose of matrix col x row 
    return np.reshape(v,(int(len(v)/row),row)).transpose()


# create a function that given a butcher tableau c|A/b and a vector of initial conditions, returns the solution of the ODE calculated with the Runge-Kutta method
def LinearRungeKutta(M , tau = None ,t_0 = 0,  t_final = None , butcher_tableau = np.array([np.array([0]),np.array([0]),np.array([1])]), y_0  = None, steps = None , draw = False , sol = None):
    """
    :param tau: the time step
    :param t_final: the final time
    :param butcher_tableau: the butcher tableau written in the form c|A/b
    :param y_0: the initial conditions for the ode
    :return: the solution of the ODE
    """
    if len(butcher_tableau) == 3:
        c,a,b = butcher_tableau
    if len(butcher_tableau) == 4:
        c,a,b ,b_star = butcher_tableau
    if tau == None and t_final == None and steps == None :
        raise ValueError("the following combinations are possible : \
        [tau,t_final]\
        [steps,tau]\
        [steps,t_final]\
        ")
    if tau == None :
        tau = float(t_final - t_0)/float(steps)
    elif t_final == None:
        t_final = t_0 + steps*tau
    elif steps == None:
        steps = (t_final - t_0)/tau

    y = [y_0]
    #print("initial position : \n" ,y[-1])

    m = len(a)
    #print("dimension of butcher tableau \n: " ,m)

    d = len(M)
    #print("dimension of the ODE : \n" ,d)

    # RHS matrix
    RHS_matrix = np.identity(m*d)-tau*np.kron(a,M)

    # Iteration Matrix
    iter_matrix = inv(RHS_matrix)

    for i in range(int(steps)):
        vec_y_1mT = np.kron(np.ones(m), M.dot(y[-1]))
        k =  vecInverse( iter_matrix.dot(vec_y_1mT) , d) 
        y.append(y[-1]+ tau* k.dot(b))
    return y

# create a function that returns the Heun iteration for the ODE 
def SimpleHeunSolver(M,steps , tau, y_0):
    y = [y_0]
    for i in range(steps):
        y_1 = y[-1] + tau*M.dot(y[-1])
        y.append(y[-1] + tau*(M.dot(y[-1]) + M.dot(y_1))/2)
    return y

########################################################

#Butcher tableau for Crank-Nicolson method
def CrankNicolson():
    c = np.array([1 , 1])
    a = np.array([[0.5 , 0] , [0 , 0.5]])
    b = np.array([0.5 , 0.5])
    return [c,a,b]
def Heun():
    c = np.array([0 , 1])
    a = np.array([[0 , 0] , [1 , 0]])
    b = np.array([1/2, 1/2])
    return [c,a,b]

def RK4():
    c = np.array([0 , 1/2 , 1/2 , 1])
    a = np.array([[0 , 0 , 0 , 0] , [1/2 , 0 , 0 , 0] , [0 , 1/2 , 0 , 0] , [0 , 0 , 1 , 0]])
    b = np.array([1/6, 1/3, 1/3, 1/6])
    return [c,a,b]
def GaussLegendre4():
    c = np.array([1/2 - 3**(1/2)/6 , 1/2 + 3**(1/2)/6])
    a = np.array([[1/4 , 1/4  - 3**(1/2)/6] , [1/4  - 3**(1/2)/6 , 1/4]])
    b = np.array([0.5 , 0.5])
    b_star = np.array([1/2 + 3**(1/2)/6, 1/2 - 3**(1/2)/6])
    return [c,a,b, b_star]

tau = 1/5
y_0  = [1,1 , 1]
draw = True
steps = 10

M_f = np.array([[1.0 , 0 , 0 ] , [0 , -1.0 , 0] , [0 , 0 , 0]])


y1 = LinearRungeKutta(M = M_f , tau = tau ,t_0 = 0 , butcher_tableau = CrankNicolson() , y_0  = y_0, steps = steps , draw = True )
y2 = SimpleHeunSolver(M = M_f,steps =steps , tau= tau, y_0=y_0)
#y2 = RungeKutta(M = M_f , tau = tau ,t_0 = 0 , butcher_tableau = RK4() , y_0  = y_0, steps = 500 , draw = True )
#y3 = RungeKutta(M = M_f , tau = tau ,t_0 = 0 , butcher_tableau = GaussLegendre4() , y_0  = y_0, steps = 500 , draw = True )


#########################################################
def dist(A, B):
    return np.sqrt(np.sum((np.array(A) - np.array(B) )*(np.array(A) - np.array(B) )))
# is it a good sol?

Sol_X =[]
Sol_Y =[]
Sol_Z =[]

T = []

X1 = []
Y1 = []
Z1 = []

X2 = []
Y2 = []
Z2 = []

#X3 = []
#Y3 = []
#Z3 = []

for i in range(len(y1)):
    T.append(i*tau)
    X1.append(y1[i][0])
    Y1.append(y1[i][1])
    Z1.append(y1[i][2])
    X2.append(y2[i][0])
    Y2.append(y2[i][1])
    Z2.append(y2[i][2])
#    X3.append(y3[i][0])
#    Y3.append(y3[i][1])
#    Z3.append(y3[i][2])
    
    Sol_X.append(np.exp(i*tau))
    Sol_Y.append(np.exp(-i*tau))
    Sol_Z.append(1)

print("general method" , dist(X1, Sol_X))
print("distance simple heun" , dist(X2, Sol_X))
#print("distance  exact sol" , dist(X3, Sol_X))
plt.plot(T,X1)
plt.plot(T,X2)
#plt.plot(T,X3)
plt.plot(T,Sol_X)
plt.legend(["Generic","SimpleHeun","Exact"])
plt.title("X")
plt.show()

print("general method" , dist(Y1, Sol_Y))
print("distance simple heun" , dist(Y2, Sol_Y))
#print("distance  exact sol" , dist(Y3, Sol_Y))
plt.plot(T,Y1)
plt.plot(T,Y2)
#plt.plot(T,Y3)
plt.plot(T,Sol_Y)
plt.legend(["Generic","SimpleHeun","Exact"])
plt.title("Y")
plt.show()

print("general method" , dist(Z1, Sol_Z))
print("distance simple heun " , dist(Z2, Sol_Z))
#print("distance  exact sol" , dist(Z3, Sol_Z))
plt.plot(T,Z1)
plt.plot(T,Z2)
#plt.plot(T,Z3)
plt.plot(T,Sol_Z)
plt.legend(["Generic","SimpleHeun","Exact"])
plt.title("Z")
plt.show()


plt.plot(X1,Y1)
plt.plot(X2,Y2)
#plt.plot(X3,Y3)
plt.plot(Sol_X,Sol_Y)
plt.legend(["Generic","SimpleHeun","Exact"])
plt.title("(X,Y)")
plt.show()


# is it a good sol?
print("distance general method",dist(np.array([Sol_X, Sol_Y ,Sol_Z]),np.array([X1, Y1 ,Z1]) ))
print("distance  simple heun ",dist(np.array([Sol_X, Sol_Y ,Sol_Z]),np.array([X2, Y2 ,Z2]) ))
##########################################