import numpy as np
import scipy as sp
from conv_diff_objects import *


class convDiffModel():
    def __init__():
        # Initialize model
        self.grid = gridObj(1, 10, 1)
        self.dt

    def run(self):
        #Schemes and terms
        self.create_terms()

        #Matrix
        self.create_local_matrix()

        #Boundary conditions
        self.apply_boundary_conditions()

        #Initial conditions
        self.set_initial_conditions()

        #Solve
        self.solve()

        #Error / Checks
        self.check_errors()

    def create_grid(self, dim, res):
        grid = gridObj(dim, res)
        self.grid = grid

    def create_terms(self):
        pass

    def create_local_matrix(self):
        pass

    def apply_boundary_conditions(self):
        pass

    def set_initial_conditions(self):
        pass

    def solve(self):
        pass

    def create_analytical_solution(self):
        pass

    def check_errors(self):
        pass

        

#- create all terms
#Create term matrix
#
#- create total matrix from all terms
#
#- create vector from matrix to solve time implicitly
#
#- solve
#
#- calculate error
#
#
#CHECKS
#- Test with analytical solution
#- Conservation ( total and at fluxes)
#- Consistency
#- Stability (CFL)
