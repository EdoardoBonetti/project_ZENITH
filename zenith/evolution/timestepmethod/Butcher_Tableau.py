# from the page https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods,  
# we take all the Butcher tableaus and we store it under the following form:
# c = np.array([...])
# a = np.array([[...] , ... , [...]])
# b = np.array([...])
# butcher_tableau_name = np.array([c],[a],[b]),


import numpy as np


####################
# Explicit methods #
####################

# butcher tableau for foreward Euler
if "forewardeuler" :
    c = np.array([0])
    a = np.array([[0]])
    b = np.array([1])
    butcher_tableau_foreward = [c,a,b]

# butcher tableau for Explicit midpoint method
if "Explicit Midpoint" :
    c = np.array([0 , 1/2])
    a = np.array([[0 , 0] , [1/2 , 0]])
    b = np.array([0, 1])
    butcher_tableau_explicit_midpoint = [c,a,b]

# butcher tableau for Heun's method
if "Heun3" :
    c = np.array([0 , 1])
    a = np.array([[0 , 0] , [1 , 0]])
    b = np.array([1/2, 1/2])
    butcher_tableau_heun2 = [c,a,b]

# butcher tableau for Ralston's method
if "Ralston" :
    c = np.array([0 , 2/3 ])
    a = np.array([[0 , 0] , [2/3 , 0] ])
    b = np.array([1/4, 3/4])
    butcher_tableau_ralston = [c,a,b]

# butcher tableau for Generic second-order method, 
# we need to previously define alpha
if "GenericSecondOrder" :
    c = np.array([0  , alpha])
    a = np.array([[0 , 0] , [alpha , 0] ])
    b = np.array([1-1/(2*alpha),1/(2*alpha)])
    butcher_tableau_generic_second_order = [c,a,b]

# butcher tableau for Kutta's third-order method
if "RK3":
    c = np.array([0 , 1/2 , 1])
    a = np.array([[0 , 0 , 0] , [1/2 , 0 , 0] , [-1 , 2 , 0]])
    b = np.array([1/6, 2/3, 1/6])
    butcher_tableau_rk3 = [c,a,b]

# butcher tableau for Generic third-order method
if "GenericThirdOrder" :
    if alpha == 0 or alpha == 2/3 or alpha == 1 :
        print("alpha must be different from 0, 2/3 and 1")
    c = np.array([0 , alpha , 1])
    a = np.array([[0 , 0 , 0] , [alpha , 0 , 0] , [1+ (1-alpha)/(alpha*(3*alpha-2)),-(1-alpha)/(alpha*(3*alpha-2) )  , 0]])
    b = np.array([1/2 - 1/(6*alpha), 1/(6*alpha*(1-alpha)),   (2-3*alpha)/(6*alpha*(1-alpha))])
    butcher_tableau_generic_third_order = [c,a,b]

# butcher tableau for Heun's third-order method
if "Heun3" :
    c = np.array([0 , 1/3 , 2/3])
    a = np.array([[0 , 0 , 0] , [1/3, 0 , 0] , [0, 2/3 , 0]])
    b = np.array([1/4, 0 , 3/4])
    butcher_tableau_heun3 = [c,a,b]

# butcher tableau for Van der Houwen's/Wray third-order method
if "Wray3":
    c = np.array([0 , 8/15 , 2/3])
    a = np.array([[0 , 0 , 0] , [8/15, 0 , 0] , [1/4, 5/12 , 0]])
    b = np.array([1/4, 0 , 3/4])
    butcher_tableau_wray3 = [c,a,b]

# butcher tableau for Ralston's third-order method
if "Raltson3":
    c = np.array([0 , 1/2 , 3/4])
    a = np.array([[0 , 0 , 0] , [1/2, 0 , 0] , [0, 3/4 , 0]])
    b = np.array([1/9, 1/3, 4/9])
    butcher_tableau_ralston3 = [c,a,b]

# butcher tableau for Third-order Strong Stability Preserving Runge-Kutta (SSPRK3)
if "SSPRK3":
    c = np.array([0 , 1 , 1/2])
    a = np.array([[0 , 0 , 0] , [1, 0 , 0] , [1/4, 1/4 , 0]])
    b = np.array([1/6, 1/6 , 2/3])
    butcher_tableau_ssprk3 = [c,a,b]

# butcher tableau for Classic fourth-order method
if "RK4":
    c = np.array([0 , 1/2 , 1/2 , 1])
    a = np.array([[0 , 0 , 0 , 0] , [1/2 , 0 , 0 , 0] , [0 , 1/2 , 0 , 0] , [0 , 0 , 1 , 0]])
    b = np.array([1/6, 1/3, 1/3, 1/6])
    butcher_tableau_rk4 = [c,a,b]

# butcher tableau for 3/8-rule fourth-order method
if "3/8-rule":
    c = np.array([0 , 1/3 , 2/3 , 1])
    a = np.array([[0 , 0 , 0, 0] , [1/3, 0 , 0, 0] , [-1/3, 1 , 0] , [1,-1,1,0]])
    b = np.array([1/8, 3/8, 3/8, 1/8])
    butcher_tableau_3_8_rule = [c,a,b]

# butcher tableau for Ralston's fourth-order method
#if "Raltson4":
#    c = np.array([0 , 0.4 ,0. , 1])
#    a = np.array([[0 , 0 , 0 , 0] , [1/2, 0 , 0 , 0] , [0, 3/4 , 0 , 0] , [0, 0 , 1 , 0]])
#    b = np.array([1/9, 1/3, 4/9, 1/9])
#    butcher_tableau_ralston4 = [c,a,b]

####################
# Embedded methods #
####################

# butcher tableau for Embedded for Heun–Euler Method
if "HeumEuler":
    c = np.array([0 , 1])
    a = np.array([[0 , 0] , [1 , 0]])
    b = np.array([1/2, 1/2])
    b_star = np.array([1, 0])
    butcher_tableau_heum_euler = [c,a,b ,b_star]

# butcher tableau for Fehlberg RK1(2)
if "Fehlberg":
    c = np.array([0 , 1 , 1])
    a = np.array([[0 , 0, 0] , [1/2 , 0 , 0] , [1/256 , 255/256 ,0 ]   ])
    b = np.array([1/512, 255/256 , 1/512])
    b_star = np.array([1/256, 255/256 , 0])
    butcher_tableau_Fehlberg = [c,a,b ,b_star]

# butcher tableau for Fehlberg RK2(3)
#if "Fehlberg2":
#    c = np.array([0 , 1 , 1 , 1])
#    a = np.array([[0 , 0 , 0] , [1/2 , 0 , 0] , [1/4 , 1/4 , 0] , [1/6 , 1/3 , 1/6]])
#    b = np.array([1/6, 1/3, 1/3, 1/6])
#    b_star = np.array([1/6, 1/3, 1/3, 1/6])
#    butcher_tableau_Fehlberg2 = [c,a,b ,b_star]

# butcher tableau for Bogacki–Shampine
if "BogackiShampine":
    c = np.array([0 , 1/2 , 3/4 , 1])
    a = np.array([[0 , 0 , 0 , 0] , [1/2 , 0 , 0 , 0] , [0 , 3/4 ,0 ,  0] , [2/9 , 1/3 , 4/9 , 0]])
    b = np.array([2/9	,1/3,	4/9	,0])
    b_star = np.array([	7/24, 1/4,	1/3	,1/8])
    butcher_tableau_BogackiShampine = [c,a,b ,b_star]

## butcher tableau for Cash-Karp RK2(4)
#if "CashKarp":
#    c = np.array([0 , 1/2 , 1/2 , 1 , 1])
#    a = np.array([[0 , 0 , 0 , 0 , 0] , [1/2 , 0 , 0 , 0 , 0] , [0 , 1/2 , 0 , 0 , 0] , [0 , 0 , 1 , 0 , 0] , [0 , 0 , 0 , 1 , 0]])
#    b = np.array([1/6, 1/3, 1/3, 1/6, 0])
#    b_star = np.array([1/6, 1/3, 1/3, 1/6, 1/6])
#    butcher_tableau_CashKarp = [c,a,b ,b_star]

# butcher tableau for The Runge–Kutta–Fehlberg method
if "RKF":
    c = np.array([0 , 1/4 , 3/8 , 12/13 , 1 , 1/2])
    a = np.array([[0 , 0 , 0 , 0 , 0 , 0] , [1/4 , 0 , 0 , 0 , 0 , 0] , [0 , 3/32 , 0 , 0 , 0 , 0] , [0 , 9/32 , 0 , 0 , 0 , 0] , [0 , 1932/2197 , -7200/2197 , 7296/2197 , 0 , 0] , [0 , 439/216 , 8 , 3680/513 , -845/4104 , 0 , 0]])
    b = np.array([16/135 , 0 , 6656/12825 , 28561/56430 , -9/50 , 2/55])
    b_star = np.array([-25/216 , 0 , 1408/2565 , -2197/4104 , 1/5 , 0])
    butcher_tableau_RKF = [c,a,b ,b_star]

# butcher tableau for The Cash-Karp method
if "CashKarp":
    c = np.array([0 , 1/5 , 3/10 , 3/5 , 1 , 7/8])
    a = np.array([[0          , 0      , 0         ,0     ,0],[1/5        , 0      , 0         ,0     ,0],[3/40       , 9/40   , 0         ,0     ,0],[3/10       ,-9/10   , 6/5, 0    , 0],[-11/54     ,5/2     , -70/27    ,	35/27, 0, 0],[1631/55296 ,175/512 , 575/13824	,44275/110592, 	253/4096, 0]])
    b = np.array([37/378 , 0 , 250/621 , 125/594 , 0 , 512/1771])
    b_star = np.array([2825/27648	,0	,18575/48384,	13525/55296	,277/14336,	1/4])
    butcher_tableau_CashKarp = [c,a,b ,b_star]

# butcher tableau for the Dormand–Prince method is:
if "DormandPrince":
    c = np.array([0 , 1/5 , 3/10 , 4/5 , 8/9 , 1 , 1])
    a = np.array([[0 , 0 , 0 , 0 , 0 , 0 , 0] , [1/5 , 0 , 0 , 0 , 0 , 0 , 0] , [3/40 , 9/40 , 0 , 0 , 0 , 0 , 0] , [44/45 , -56/15 , 32/9 , 0 , 0 , 0 , 0] , [19372/6561 , -25360/2187 , 64448/6561 , -212/729 , 0 , 0 , 0] , [9017/3168 , -355/33 , 46732/5247 , 49/176 , -5103/18656 , 0 , 0] , [35/384 , 0 , 35/384 , 0 , 35/384 , 0 , 0] , [0 , 0 , 0 , 0 , 0 , 1 , 0]])
    b_star = np.array([5179/57600 , 0 , 7571/16695 , 393/640 , -92097/339200 , 187/2100 , 1/40])
    b = np.array([35/384 , 0 , 0 , 0 , 0 , 0 , 0])

###########################
#     Implicit Method     #
###########################
# we give all the possible implicit methods given in https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods

#Butcher tableau for Backward Euler
if "BackwardEuler":
    c = np.array([1])
    a = np.array([[1]])
    b = np.array([1])
    butcher_tableau_BackwardEuler = [c,a,b ]

#Butcher tableau for Implicid Midpoint rule
if "ImplicitMidpoint":
    c = np.array([0.5])
    a = np.array([[0.5]])
    b = np.array([1])
    butcher_tableau_ImplicitMidpoint = [c,a,b]

#Butcher tableau for Crank-Nicolson method
if "CrankNicolson":
    c = np.array([1 , 1])
    a = np.array([[0.5 , 0] , [0 , 0.5]])
    b = np.array([0.5 , 0.5])
    butcher_tableau_CrankNicolson = [c,a,b]

#Butcher tableau for Gauss–Legendre methods
if "GaussLegendre4":
    c = np.array([1/2 - 3**(1/2)/6 , 1/2 + 3**(1/2)/6])
    a = np.array([[1/4 , 1/4  - 3**(1/2)/6] , [1/4  - 3**(1/2)/6 , 1/4]])
    b = np.array([0.5 , 0.5])
    b_star = np.array([1/2 + 3**(1/2)/6, 1/2 - 3**(1/2)/6])
    butcher_tableau_GaussLegendre = [c,a,b, b_star]

### stopped just before Gauss Legendre 6th order method###



