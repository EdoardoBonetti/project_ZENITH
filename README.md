# Zenith solver
This is a solver plugin for NGSolve which can be used to solve the EinsteinFieldEquation

# Founding 
We thank the Austrian Science Fund (FWF) for funding via the stand alone project F 6511N-N36 .

Developed at:

- TU Wien, Institute for Analysis and Scientific Computing


# Installation

## Prerequisites
- NGSolve (https://ngsolve.org/)
- Python 3.6 or higher
- numpy
- scipy
- matplotlib

## Installation
1. Clone this repository
2. Add the path to the repository to your PYTHONPATH environment variable
3. Add the path to the repository to your PATH environment variable

# Usage
The solver can be used as a plugin for NGSolve. The solver can be used in the following way:

```python
from ngsolve import *
from zenith import *
```
in the following we will show how to use the solver to solve the EinsteinFieldEquation


# Documentation

## ZenithSolver
The ZenithSolver class is the main class of the solver. It can be used to solve the compressible Navier-Stokes equations. The solver can be used in the following way:

[1] Create an initial conofiguration of physical objects

```python
from zenith import BlackHole

bh1 = BlackHole(mass=1, position=(-1,0,0), momentum=(0,0,1), spin=(0,0,0))

bh2 = BlackHole(mass=1, position=(1,0,0), momentum=(0,0,1), spin=(0,0,0))

bh_list = [bh1, bh2]
```

[2] Solve the boundary value problem, i.e. the initial data problem:

```python
BVPSolver = BVP.BowenYork(bh_list)
BVPSolver.Solve()
```

[3] Solve the initial value problem, i.e. the evolution problem.

```python
IVPSolver = IVP.BSSN_puncture( initial_data = BVPSolver.InitialData )

for i in range(10): 
    IVPSolver.Step(0.1)
```



## Initial data: BVP  
Contains the different formulations to find initial data.

- BVP.BowenYork: 
    Bowen-York initial data for a set of black holes. Each black hole must have (mass, position, momentum, spin) as parameters.

NOTE: In the future different initial data formulations will be added.

## Evolution: IVP 
This class contains the different formulations to solve the evolution equations.

- BSSN_puncture: 
    BSSN formulation with puncture gauge conditions.

NOTE: In the future different evolution formulations will be added, such as ccz4, etc.

