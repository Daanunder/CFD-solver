import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
import pandas as pd
import json


class pressurePoissonModel():
    """
    Solver for the pressure poisson in multiple dimensions
    
    Generally solves:
    1. dv/dt = -1/rho*grad(P) + v*div(v) + k*laplace(v)
    2. 1/rho*laplace(P) = - div( div(v)*v )

    """
    def __init__(self, modelName):
        self.get_basic_settings(modelName)

        # Init grid
        self.grid = gridObj(
                self.settings.get("dimensions"),
                self.settings.get("resolution"),
                self.settings.get("domain"),
                self.settings.get("velocity_faces"),
                self.settings.get("pressure_faces")
                )
        
        # Do initial checks
        self.check_initial_stability()

        # Set boundary conditions
        self.boundaryConditions = boundaryConditions(self)
        
        # Set numerical schemes
        self.gradPres = gradientPressureSchemes(self).explicit
        self.advVel = advectionSchemes(self).explicitCentral
        self.diffVel = diffusionSchemes(self).explicit
        self.laplacePres = laplacianPressureSchemes(self).explicit
        self.divMomentum = divMomentumSchemes(self).explicit

        # Visualtisation
        self.analyticalSolution = analyticalSolution(self)
        self.plotter = visualisationObj(self)

        # Setup matrix (BC are applied here)
        self.create_eom_matrices()
        self.create_poisson_matrices()
        self.create_pressure_correction_matrices()
        #self.test_poisson()

        # Set initial conditions
        self.timesteps = np.arange(0, self.t_end+self.dt, self.dt)
        self.current_timestep_index = 0
        self.vstar_result = np.zeros((self.grid.vector_size, self.timesteps.shape[0]))
        self.vcorr_result = np.zeros((self.grid.vector_size, self.timesteps.shape[0]))
        self.vfinal_result = np.zeros((self.grid.vector_size, self.timesteps.shape[0]))
        self.pressure_result = np.zeros((self.grid.nr_of_cells, self.timesteps.shape[0]))
        self.initialConditions = initialConditions(self).sinusoid(self.initial_amplitudes, self.initial_wavenumbers)

    def get_basic_settings(self, modelName):
        with open('working_models.json') as json_file:
            self.settings = json.load(json_file)[modelName]

        self.log = self.settings.get("log") == 1

        self.name = modelName
        self.initial_amplitudes = np.array(self.settings.get("initial_amplitudes"))
        self.initial_wavenumbers = np.array(self.settings.get("initial_wavenumbers"))
        self.fixedFaceValues = np.array(self.settings.get("fixedFaceValues"))
        self.wall_velocities = np.array(self.settings.get("wall_velocities"))
        
        self.t_end = self.settings.get("t_end")
        self.dt = self.settings.get("dt")
        self.u = np.array(self.settings.get("u"))
        self.kappa = np.array(self.settings.get("kappa"))
        self.viscosity = self.settings.get("viscosity")
        self.density = self.settings.get("density")

    def check_initial_stability(self):
        print("Stability numbers")

        self.CFL = self.dt / self.grid.ds * self.u
        print("CFL: ", self.CFL)
        if any(self.CFL > 1):
            print(f"WARNING: CFL CONDITION NOT MET. Spatial resolution: {self.grid.ds}, dt:{self.dt}, u:{self.u}")

        self.PECLET = self.grid.ds * self.u / self.kappa
        print("PECLET: ", self.PECLET)
        if any(self.PECLET > 2):
            print(f"WARNING: PECLET NUMBER > 2. Spatial resolution: {self.grid.ds}, u:{self.u}, kappa:{self.kappa}")

        self.gamma = self.kappa*self.initial_wavenumbers**2 + self.u*self.initial_wavenumbers
        print("gamma: ", self.gamma)
        if any(1/self.gamma < self.dt):
            print(f"WARNING: Analytical stability not ensured. k:{self.initial_wavenumbers}, dt:{self.dt}, u:{self.u}, kappa:{self.kappa}")

        self.G1 = self.grid.ds**2 / 2 / self.kappa
        print("G1: ", self.G1)
        if any(self.G1 < self.dt):
            print(f"WARNING: Numerical stability G1 not ensured. dt:{self.dt}, ds:{self.grid.ds}, kappa:{self.kappa}")

        self.G2 = 2 * self.kappa / self.u**2
        print("G2: ", self.G2)
        if any(self.G2 < self.dt):
            print(f"WARNING: Numerical stability G2 not ensured. dt:{self.dt}, u:{self.u}, kappa:{self.kappa}")

    def create_eom_matrices(self):
        """
        Set up momentum matrices; For example, for 2D there are two velocity components thus these momentum equations are merged in a single matrix equation. Where the matrix rank (0, N) for the momentum equation of a cell in a certain dimension is the cell ID*nr_of_dims + dimension. Where the total number of matrix rows equals N = self.grid.vector_size =  nr_cells * nr_dimensions.

        Depending on the schemes used terms either end up in LHS or RHS, with the exception of the non linear advection terms which are always explicit and stored in RHS_adv. The RHS_const is available for possible fixed boundary conditions, sources or sinks.

        | LHS  ..  ..  |   | u0_n+1 |     | RHS .. ..  |   | u0_n |    | RHS_const |   | RHS_adv |
        |  ..  ..  ..  |   | v0_n+1 |     | ..  .. ..  |   | v0_n |    | ..        |   | ..      |
        |  ..  ..  ..  |   | u1_n+1 |     | ..  .. ..  |   | u1_n |    | ..        |   | ..      |
        |  ..  ..  ..  | . | v1_n+1 |  =  | ..  .. ..  | . | v1_n | +  | ..        | + | ..      |
        |  ..  ..  ..  |   | ..     |     | ..  .. ..  |   | ..   |    | ..        |   | ..      |
        |  ..  ..  ..  |   | ..     |     | ..  .. ..  |   | ..   |    | ..        |   | ..      |
        |  ..  ..  ..  |   | uN_n+1 |     | ..  .. ..  |   | uN_n |    | ..        |   | ..      |
        |  ..  ..  LHS |   | vN_n+1 |     | ..  .. RHS |   | vN_n |    | RHS_const |   | RHS_adv |
        """
        # Initialize momentum matrices 
        self.EOM_LHS = np.zeros((self.grid.vector_size, self.grid.vector_size))
        self.EOM_RHS = np.zeros((self.grid.vector_size, self.grid.vector_size))
        self.EOM_RHS_adv = np.zeros(self.grid.vector_size)
        self.EOM_RHS_const = np.zeros(self.grid.vector_size)
        
        for ID, c in self.grid.cells.items():
            for d in range(self.grid.dim):
                rank = ID*self.grid.dim + d
                diff = self.diffVel(d, c)
                self.EOM_LHS[rank] = diff[0]
                self.EOM_RHS[rank] = diff[1]
                self.EOM_RHS_const[rank] = diff[2]
            
    def create_poisson_matrices(self):
        self.POIS_LHS = np.zeros((self.grid.nr_of_cells, self.grid.nr_of_cells))
        self.POIS_RHS = np.zeros((self.grid.nr_of_cells, self.grid.vector_size))
        self.POIS_RHS_const = np.zeros(self.grid.nr_of_cells)
        
        for ID, c in self.grid.cells.items():
            lhs = np.zeros(self.grid.nr_of_cells)
            rhs = np.zeros(self.grid.vector_size)

            # For current cell, add contributions of all dimensions 
            for d in range(self.grid.dim):
                lhs += self.laplacePres(c, d)
                rhs += self.divMomentum(c, d)
            
            self.POIS_LHS[ID] = lhs
            self.POIS_RHS[ID] = rhs

        # avoid singular matrix
        #p_ref_vector = np.zeros(self.grid.nr_of_cells)
        #p_ref_vector[0] = 1
        #self.POIS_LHS[0] = p_ref_vector

        #p_ref_value = 0
        #self.POIS_RHS_const[0] = p_ref_value

        #self.POIS_RHS[0] = np.zeros(self.grid.vector_size)
    
    def create_pressure_correction_matrices(self):
        self.CORR_RHS = np.zeros((self.grid.vector_size, self.grid.nr_of_cells))
        for ID, c in self.grid.cells.items():
            for d in range(self.grid.dim):
                rank = c.ID*self.grid.dim + d
                self.CORR_RHS[rank] = self.gradPres(c, d)

    def test_poisson(self):
        self.Q_vector = np.zeros(self.grid.vector_size)
        for ID, c in self.grid.cells.items():
            self.Q_vector[ID] = self.analyticalSolution.test_equation(c.coord)

        # P matrix is defined by the LHS of the poisson matrices
        P = self.POIS_LHS

        prescribed_pressure_cell_id = 0
        avoid_singular = np.zeros(self.grid.vector_size)
        avoid_singular[prescribed_pressure_cell_id] = 1
        P[prescribed_pressure_cell_id] = avoid_singular
        self.Q_vector[prescribed_pressure_cell_id] = self.analyticalSolution.analytical_p(self.grid.cells.get(prescribed_pressure_cell_id).coord)

        # Solve Px = Q where x are the pressures in the cells
        self.pres_res = np.linalg.solve(P, self.Q_vector)
        
        # check difference with P_analytical
        self.analytical_p = np.zeros(self.grid.vector_size)
        for ID, c in self.grid.cells.items():
            self.analytical_p[ID] = self.analyticalSolution.analytical_p(c.coord)

        self.poisson_error = np.abs(self.analytical_p - self.pres_res)
        self.total_poisson_error = np.sum(self.poisson_error)
        
        if self.log:
            print(P)
            fig,(ax1,ax2) = plt.subplots(2, 1)
            ax1.plot(self.analytical_p, label="analytical")
            ax1.plot(self.pres_res, label="approx.")
            ax1.legend()
            ax2.plot(self.poisson_error, label="Error")
            ax2.legend()

            fig.suptitle(f"Total error: {self.total_poisson_error}")
            plt.savefig("./results/1D-pressure-poisson-test.png")
            plt.cla()
            plt.close("all")
    
    def solve_eom(self):
        """
        EOM_LHS * v_star = EOM_RHS * vn + EOM_RHS_const + EOM_RHS_adv
        """
        print("Solving EOM, time: ", self.current_timestep_index) 
        vn = self.vfinal_result[:,self.current_timestep_index]
        # Update non linear advection terms in RHS_adv
        for ID,c in self.grid.cells.items():
            for d in range(self.grid.dim):
                rank = ID*self.grid.dim + d
                self.EOM_RHS_adv[rank] = self.advVel(d, c)
            
        # Explicit part
        RHS = (np.identity(self.grid.vector_size) + self.EOM_RHS) * self.dt/self.grid.cellVolume
        b_const = self.EOM_RHS_const + self.EOM_RHS_adv
        b = np.dot(RHS, vn) + b_const*self.dt/self.grid.cellVolume

        # Implicit part
        A = np.identity(self.grid.vector_size) - self.EOM_LHS*self.dt/self.grid.cellVolume

        # solve A, b to obtain vstar
        v_star = np.linalg.solve(A, b)
        self.vstar_result[0:, self.current_timestep_index] = v_star

    def solve_pressure_poisson(self):
        print("Solving Poisson, time: ", self.current_timestep_index) 
        vstar = self.vstar_result[0:, self.current_timestep_index]
        b = np.dot(self.POIS_RHS, vstar) + self.POIS_RHS_const
        A = self.POIS_LHS 
        self.pressure_result[0:, self.current_timestep_index] = np.linalg.solve(A,b)

    def calc_velocity_correction(self):
        self.vcorr_result[0:, self.current_timestep_index] = np.dot(self.CORR_RHS, self.pressure_result[0:, self.current_timestep_index])
    
    def correct_velocities(self):
        self.vfinal_result[0:, self.current_timestep_index+1] = self.vstar_result[0:, self.current_timestep_index] + self.vcorr_result[0:, self.current_timestep_index] 

    def solve(self):
        # Set t0
        self.current_timestep_index = 0
        for ti in range(1, self.timesteps.shape[0]):

            # Solve EOM: vstar = EOM(u,v)
            self.solve_eom()

            # Calculate new pressures pn+1 = Poisson(vn+1)
            self.solve_pressure_poisson()
            
            # Calculate velocity correction: vc = - dt/rho * grad(pc)
            self.calc_velocity_correction()

            # Correct velocities: vn+1 = vn+1 + vc
            self.correct_velocities()

            # update time index
            self.current_timestep_index = ti


    def getLocalVector(self, ddim, vdim, cell):
        V_i_rank = cell.ID*self.grid.dim + vdim

        negbound = False
        posbound = False
        V_neg_id, V_pos_id = cell.neighbors[ddim]
        V_neg = self.grid.locations.get(V_neg_id)
        if V_neg == None:
            negbound = True
            V_neg_rank = V_neg_id
        else:
            V_neg_rank = V_neg*self.grid.dim + vdim

        V_pos = self.grid.locations.get(V_pos_id)
        if V_pos == None:
            posbound = True
            V_pos_rank = V_pos_id
        else:
            V_pos_rank = V_pos*self.grid.dim + vdim
        
        return [V_neg_rank, V_i_rank, V_pos_rank], negbound, posbound

class gridObj(object):
    """Grid definition of a CFD program"""

    def __init__(self, dim, res, domain, velocity_faces, pressure_faces):
        """
        :dim: int
            number of dimensions 
        :res: array_like
            number of cells in each direction, should be of size 'dim'
        :domain: array_like
            size of the domain in each direction, should be of size 'dim'
        :velocity_faces: array of size [dim, 2]
            Velocity boundary condition types for each face of the domain, for both the positive- and negative normal direction.
        :pressure_faces: array of size [dim, 2]
            Pressure boundary condition types for each face of the domain, for both the positive- and negative normal direction.
        """
        self.dim = dim
        self.res = np.array(res)
        self.domain = np.array(domain)
        self.ds = self.domain/self.res
        self.cellVolume = np.prod(self.ds)
        self.nr_of_cells = np.prod(self.res)
        self.vector_size = self.nr_of_cells*self.dim
        self.span_size = np.zeros(self.dim)
        self._setup_span_size()
        self.cells = dict() 
        self.locations = dict()
        self.velocity_faces = velocity_faces
        self.pressure_faces = pressure_faces
        self.cellFaces = dict()
        
        ## SETUP FUNCS
        self.initialize_cells()
        
    def _setup_span_size(self):
        prev_prod = 1
        for i,r in enumerate(self.res):
            prod = prev_prod*r
            self.span_size[i] = prod 
            prev_prod = prod

    def initialize_cells(self):
        for i in range(self.nr_of_cells):
            loc = (i % self.span_size) / (self.span_size / self.res)
            loc = loc.astype(int)
            coord = self.ds * loc + self.ds/2
            
            neighbors = []
            for n,l in enumerate(loc):
                negside_neighbor = [int(t) for t in loc] 
                if l - 1 >= 0:
                    negside_neighbor[n] = int(l - 1) 
                    nsn_str = ".".join([str(k) for k in negside_neighbor])
                else:
                    negside_neighbor[n] = "-"
                    nsn_str = ".".join([str(k) for k in negside_neighbor])
                    self.cellFaces[nsn_str] = {"velocity":self.velocity_faces[n][0],"pressure":self.pressure_faces[n][0]}

                posside_neighbor = [int(t) for t in loc]
                if l + 1 < self.res[n]:
                    posside_neighbor[n] = int(l + 1)
                    psn_str = ".".join([str(k) for k in posside_neighbor])
                else:
                    posside_neighbor[n] = "+"
                    psn_str = ".".join([str(k) for k in posside_neighbor])
                    self.cellFaces[psn_str] = {"velocity":self.velocity_faces[n][1],"pressure":self.pressure_faces[n][1]}

                nb = [nsn_str, psn_str]
                neighbors.append(nb)

            self.cells[i] = cellObj(i, loc, coord, neighbors)
            self.locations[".".join([str(int(l)) for l in loc])] = i

class cellObj(object):
    """Cell object containing location and neighbors"""

    def __init__(self, ID, loc, coord, neighbors):
        """Object containing grid cell information.

        :ID: int
            Unique ID of the grid cell
        :loc: array_like
            Integer location of the grid cell in all model dimensions
        :coord: array_like
            Absolute coordinate of the grid cell in all model dimensions
        :neighbors:
            Neighboring cells or boundaries

        """
        self.ID = ID
        self.loc = loc
        self.dim = len(loc)
        self.neighbors = neighbors
        self.coord = coord

class boundaryConditions(object):
    """
    Provide boundary condition functions that return the matrix row for the equation of the given cell 

    Should return [lhs, rhs, rhs_const] of size model.vector_size 

    Depends on divConv, divDiff, divDt

    """
    def __init__(self, model):
        self.available_bc = {
                "poisson":
                {
                    "velocity":
                        {
                        "fixedWall": self.fixedWallBC,
                        "noslipWall": self.noslipWallBC,
                        },
                    "pressure":
                        {
                        "zerogradient": self.zeroGradientBC,
                        "fixed": self.fixedPressureBC,
                        }
                    },
                "momentum":
                {
                    "velocity":
                        {
                        "fixedWall": self.fixedWallBC,
                        "noslipWall": self.noslipWallBC,
                        },
                    "pressure":
                        {
                        "zerogradient": self.zeroGradientBC,
                        "fixed": self.fixedPressureBC,
                        }
                }
            }
        self.model = model
        
    def get_contribution(self, cell_face, bc_type, variable, equation, **kwargs):
        return self.available_bc.get(equation).get(variable).get(bc_type)(cell_face, **kwargs)
    
    # Velocity 
    def fixedWallBC(self, cell_face, **kwargs):
        return 0

    def noslipWallBC(self, cell_face, **kwargs):
        raise Exception("THIS BC TYPE IS NOT IMPLEMENTED")
    
    # Pressures
    def zeroGradientBC(self, cell_face, **kwargs):
        lhs = np.zeros(self.model.grid.vector_size)
        loc = cell_face.split(".")
        for d,l in enumerate(loc):
            term_contribution = self.model.laplacePres(d)
            if l == "-":
                cell_loc = cell_face.replace("-", "0")
                cell_id = self.model.grid.locations[cell_loc]
                lhs[cell_id] = term_contribution[0][2]
            elif l == "+":
                cell_loc = cell_face.replace("+", str(self.model.grid.res[d]-1))
                cell_id = self.model.grid.locations[cell_loc]
                lhs[cell_id] = term_contribution[0][0]
        return lhs

    def fixedPressureBC(self, cell_face, **kwargs):
        raise Exception("THIS BC TYPE IS NOT IMPLEMENTED")

    def getWallVelocity(self, cell_face, dimension):
        # TODO: update to work with 3D
        if "-" in cell_face:
            side = 0
        else:
            side = 1

        wall_velocity = self.model.wall_velocities[dimension][side]
        print("Wall velocity: ", "Face: ", cell_face, "Dim: ", dimension, "V: ", wall_velocity)
        return wall_velocity

class initialConditions(object):
    def __init__(self, model):
        self.model = model

    def sinusoid(self, A, k, p=None):
        """
        :A: array_like of size self.model.grid.dim
            Amplitudes in each direction
        :k: array_like of size self.model.grid.dim
            Wave number in each direction
        :p: [optional] array_like of size self.model.grid.dim
            Phase shift in each direction
            
        Returns a periodic initial condition of sinusoid shape with amplitude A, wavenumber k and phase p. For a single direction: sin(kx*2pi/L+p)
        """
        if p == None:
            p = np.zeros(self.model.grid.dim)

        for ID, c in self.model.grid.cells.items():
            #if ID > 700:
                #break
            sin_array = A*np.sin(k*c.coord/self.model.grid.domain*np.pi*2 + p)
            #sinprod = np.prod(A*np.sin(k*c.coord/self.model.grid.domain*np.pi*2 + p))
            self.model.vfinal_result[ID][0] = np.prod(sin_array)

class advectionSchemes(object):
    """
    Provide advection discretization functions based on the given dimension. Returns the coefficients for the local matrix based on temporal and spatial discretization scheme.

    Attention! Only explicit scheme available for now, thus only returning a single float for EOM_RHS_adv

    :Returns: float that represents explicit contribution of normal and shear advection terms
    """
    def __init__(self, model):
        self.model = model
        self.positive_coefficient_matrix =      [[0, 0, 0],
                                                [0, 1/4, 1/4],
                                                [0, 1/4, 1/4]],

        self.negative_coefficient_matrix =      [[-1/4, -1/4, 0],
                                                [-1/4, -1/4, 0],
                                                [0, 0, 0]],

    def explicitCentral(self, dim, cell):
        # compute normal and shear advection contributions based on V_n
        # get outer product of current dimension vector with all dimension vectors
        # for example, in 3D: vxv, vxu, vxw 
        # vdim = velocity dimension
        # ddim = directional dimension
        
        advectiveChange = 0
        for ddim in range(self.model.grid.dim):
            vdim = dim
            u_vector, negbound, posbound = self.model.getLocalVector(ddim, vdim, cell)
            u_vector, negbound, posbound = self.getVectorValuesFromVector(u_vector, negbound, posbound)
            vdim = ddim
            v_vector, negbound, posbound = self.model.getLocalVector(ddim, vdim, cell)
            v_vector, negbound, posbound = self.getVectorValuesFromVector(v_vector, negbound, posbound)
            outer_prod = np.outer(u_vector, v_vector)
            #print(outer_prod)
            if not negbound:
                advectiveChange += np.sum(outer_prod * self.negative_coefficient_matrix) * self.model.grid.cellVolume/self.model.grid.ds[ddim]
            if not posbound:
                advectiveChange += np.sum(outer_prod * self.positive_coefficient_matrix) * self.model.grid.cellVolume/self.model.grid.ds[ddim]

        return advectiveChange 
        
    def getVectorValuesFromVector(self, vector, negbound, posbound):
        # TODO: possibly implement boundary conditions
        V_neg_rank, V_i_rank, V_pos_rank = vector
        if negbound: 
            V_vector = self.model.vfinal_result[:, self.model.current_timestep_index][[V_i_rank, V_pos_rank]]
            V_vector = np.append([0], V_vector)
        elif posbound: 
            V_vector = self.model.vfinal_result[:, self.model.current_timestep_index][[V_neg_rank, V_i_rank]]
            V_vector = np.append(V_vector, [0])
        else:
            V_vector = self.model.vfinal_result[:, self.model.current_timestep_index][[V_neg_rank, V_i_rank, V_pos_rank]]
        return V_vector, negbound, posbound

class diffusionSchemes(object):
    """
    Provide diffusion discretization functions based on the given dimension. Returns the terms for the local matrix based on temporal and spatial discretization scheme.

    :Returns: [lhs, rhs, rhs_const]: arrays of size self.model.vector_size
    """
    def __init__(self, model):
        self.model = model

        # Normal vector  = [i-1, i, i+1]
        self.normal_negative_coefficient_matrix = np.array([1, -1, 0])
        self.normal_positive_coefficient_matrix = np.array([0, -1, 1])
        self.shear_negative_coefficient_matrix = np.array([-1, 0, 1])
        self.shear_positive_coefficient_matrix = np.array([1, 0, -1])

    def explicit(self, dim, c):
        #print("CELL", c.ID)
        lhs = np.zeros(self.model.grid.vector_size)
        rhs = np.zeros(self.model.grid.vector_size)
        # TODO: possibly add rhs const with bc
        rhs_const = 0
        
        selector = np.s_[:]
        
        ## NORMAL SHEAR COMPONENT
        u_vector_i, negbound_normal, posbound_normal = self.model.getLocalVector(dim, dim, c)
        if negbound_normal:
            # implement BC du/dx i-1/2 - Always zero? - Only with accelerations or deformations non zero
            selector = np.s_[1:]
        elif posbound_normal:
            # implement BC du/dx i+1/2 - Always zero? - Only with accelerations or deformations non zero
            selector = np.s_[:-1]
        
        u_vector_i = u_vector_i[selector]
        
        normal_distance = self.model.grid.ds[dim]
        if not negbound_normal:
            rhs[u_vector_i] += self.normal_negative_coefficient_matrix[selector]*2 / normal_distance

        if not posbound_normal:
            rhs[u_vector_i] += self.normal_positive_coefficient_matrix[selector]*2 / normal_distance 
    
        ## DIAGONAL SHEAR COMPONENT
        for d in range(self.model.grid.dim):
            # Txy|j+1/2 = mu * (du/dy|j+1/2 +  dv/dx|j+1/2)
            # Txy|j-1/2 = mu * (du/dy|j-1/2 +  dv/dx|j-1/2)
            if d == dim:
                continue

            cross_selector = np.s_[:]

            negbound_shear = False
            posbound_shear = False
            
            u_vector_j, _, _ = self.model.getLocalVector(d, dim, c)

            v_vector_i, _, _ = self.model.getLocalVector(dim, d, c)
            v_vector_i = v_vector_i[selector]

            nsn, psn =  c.neighbors[d]

            nsn_id = self.model.grid.locations.get(nsn)
            psn_id = self.model.grid.locations.get(psn)
            if nsn_id == None:
                nsn_cell_face = nsn
                cross_selector = np.s_[1:]
            elif psn_id == None:
                psn_cell_face = psn
                cross_selector = np.s_[:-1]

            u_vector_j = u_vector_j[cross_selector]

            if not nsn_id == None:
                nsnc = self.model.grid.cells.get(nsn_id)
                v_vector_jm1, negbound_temp, posbound_temp = self.model.getLocalVector(dim, d, nsnc)
                if not negbound_temp == negbound_normal or not posbound_temp == posbound_normal:
                    raise Exception("Something is wrong 1")
                if negbound_normal:
                    # TODO: Apply BC in approximation of gradient 1.5 \Delta y
                    cross_distance = self.model.grid.ds[d]*1.5
                elif posbound_normal:
                    # TODO: Apply BC in approximation of gradient 1.5 \Delta y
                    cross_distance = self.model.grid.ds[d]*1.5
                else:
                    cross_distance = self.model.grid.ds[d]*2

                rhs[u_vector_j] += self.normal_negative_coefficient_matrix[cross_selector] / self.model.grid.ds[dim]
                v_vector_jm1 = v_vector_jm1[selector]
                rhs[v_vector_jm1] += self.shear_negative_coefficient_matrix[selector] *  1/2 / cross_distance 
                rhs[v_vector_i] += self.shear_negative_coefficient_matrix[selector] *  1/2 / cross_distance

            else:
                #du/dy|j-1/2 = (uj - Uwall)/0.5y 
                #dv/dx|j-1/2 = 0
                cell_rank = c.ID*self.model.grid.dim + dim
                rhs[cell_rank] += 1 / (1/2*self.model.grid.ds[d])
                rhs_const += -self.model.boundaryConditions.getWallVelocity(nsn, d) / (1/2*self.model.grid.ds[d])

            if not psn_id == None:
                psnc = self.model.grid.cells.get(psn_id)
                v_vector_jp1, negbound_temp, posbound_temp = self.model.getLocalVector(dim, d, psnc)
                if not negbound_temp == negbound_normal or not posbound_temp == posbound_normal:
                    raise Exception("Something is wrong 2")
                if negbound_normal:
                    #Apply BC in approximation of gradient 1.5 \Delta y
                    cross_distance = self.model.grid.ds[d]*1.5
                elif posbound_normal:
                    #Apply BC in approximation of gradient 1.5 \Delta y
                    cross_distance = self.model.grid.ds[d]*1.5
                else:
                    cross_distance = self.model.grid.ds[d]*2

                rhs[u_vector_j] += self.normal_positive_coefficient_matrix[cross_selector] / self.model.grid.ds[dim]
                v_vector_jp1 = v_vector_jp1[selector]
                rhs[v_vector_jp1] += self.shear_positive_coefficient_matrix[selector] *  1/2 / cross_distance
                rhs[v_vector_i] += self.shear_positive_coefficient_matrix[selector] *  1/2 / cross_distance

            else:
                #du/dy|j+1/2 = (Uwall - uj)/0.5y 
                #dv/dx|j-1/2 = 0
                cell_rank = c.ID*self.model.grid.dim + dim
                rhs[cell_rank] += -1 / (1/2*self.model.grid.ds[d])
                rhs_const += self.model.boundaryConditions.getWallVelocity(nsn, d) / (1/2*self.model.grid.ds[d])
        
        
        lhs = lhs * self.model.viscosity * self.model.grid.cellVolume/self.model.grid.ds[dim]
        rhs = rhs * self.model.viscosity * self.model.grid.cellVolume/self.model.grid.ds[dim]
        rhs_const = rhs_const * self.model.viscosity * self.model.grid.cellVolume/self.model.grid.ds[dim]
        return lhs, rhs, rhs_const

    def implicit(self, dim, c):
        # simply returns the matrix in different order
        rhs, lhs, rhs_const = self.explicit(dim, c)
        return lhs, rhs, rhs_const

class gradientPressureSchemes(object):
    """
    Provides the discretization of the pressure gradient for the velocity correction

    :Returns: array-like of size self.model.grid.nr_of_cells: rhs contribution to the velocity correction
    """
    def __init__(self, model):
        self.model = model
        self.negative_coefficient_matrix = [-1/2, -1/2, 0]
        self.positive_coefficient_matrix = [0, 1/2, 1/2]

    def explicit(self, c, dim):
        rhs = np.zeros(self.model.grid.nr_of_cells)
        nsn, psn = c.neighbors[dim]
        
        negbound = False
        posbound = False

        nsnc = self.model.grid.locations.get(nsn)
        psnc = self.model.grid.locations.get(psn)

        selector = np.s_[:]
        p_vector = [nsnc, c.ID, psnc]
        normal_distance = self.model.grid.ds[dim]
        if nsnc == None:
            negbound = True
            selector = np.s_[1:]

        elif psnc == None:
            posbound = True
            selector = np.s_[:-1] 

        if not negbound:
            rhs[p_vector[selector]] += self.negative_coefficient_matrix[selector] / normal_distance
        if not posbound:
            rhs[p_vector[selector]] += self.positive_coefficient_matrix[selector] / normal_distance
        
        rhs = -1 * rhs * self.model.dt / self.model.density

        return rhs

class laplacianPressureSchemes(object):
    """
    Provide laplacian discetization in the given dimension for the given cell, i.e. rank in the matrix. 

    :Returns: array-like of size self.model.grid.nr_of_cells: lhs contribution to the equation for the given cell in the given dimension
    """
    def __init__(self, model):
        self.model = model
        self.negative_coefficient_matrix = [1, -1, 0]
        self.positive_coefficient_matrix = [0, -1, 1]

    def explicit(self, c, dim):
        lhs = np.zeros(self.model.grid.nr_of_cells)
        nsn, psn = c.neighbors[dim]
        
        negbound = False
        posbound = False

        nsnc = self.model.grid.locations.get(nsn)
        psnc = self.model.grid.locations.get(psn)

        selector = np.s_[:]
        p_vector = [nsnc, c.ID, psnc]
        if nsnc == None:
            negbound = True
            selector = np.s_[1:]
        elif psnc == None:
            posbound = True
            selector = np.s_[:-1] 
        normal_distance = self.model.grid.ds[dim]
        if not negbound:
            lhs[p_vector[selector]] += self.negative_coefficient_matrix[selector] / normal_distance
        if not posbound:
            lhs[p_vector[selector]] += self.positive_coefficient_matrix[selector] / normal_distance
        
        lhs = lhs * self.model.grid.cellVolume/self.model.grid.ds[dim]
        return lhs
        
class divMomentumSchemes(object):
    """
    Provide diffusion discretization functions based on the given dimension. Returns the terms for the local matrix based on temporal and spatial discretization scheme.

    :Returns: [lhs, rhs, rhs_const]
    """
    def __init__(self, model):
        self.model = model
        self.negative_coefficient_matrix = [-1/2, -1/2, 0]
        self.positive_coefficient_matrix = [0, 1/2, 1/2]

    def explicit(self, c, dim):
        rhs = np.zeros(self.model.grid.vector_size)
        nsn, psn = c.neighbors[dim]
        negbound = False
        posbound = False

        nsnc = self.model.grid.locations.get(nsn)
        psnc = self.model.grid.locations.get(psn)

        selector = np.s_[:]
        u_vector, _, _ = self.model.getLocalVector(dim, dim, c)
        if nsnc == None:
            negbound = True
            selector = np.s_[1:]
        elif psnc == None:
            posbound = True
            selector = np.s_[:-1] 

        normal_distance = self.model.grid.ds[dim]
        if not negbound:
            rhs[u_vector[selector]] += self.negative_coefficient_matrix[selector]
        if not posbound:
            rhs[u_vector[selector]] += self.positive_coefficient_matrix[selector]
        
        rhs = rhs * self.model.density/self.model.dt/self.model.grid.ds[dim]
        return rhs

class visualisationObj(object):
    def __init__(self, model):
        self.model = model

    def get_data(self,n=10):
        plot_index = list(set([int(i) for i in np.linspace(0, self.model.timesteps.shape[0]-1, n)]))
        plot_result = self.model.result[:, plot_index]
        self.df = pd.DataFrame(plot_result, columns=[t for t in self.model.timesteps[plot_index]])

    def plot1D(self):
        self.get_data(n=5)
        if self.model.grid.dim != 1:
            raise Exception(f"ERROR: Plotting 1D but model has {self.model.grid.dim} dimensions")
        
        sns.lineplot(self.df)

        plt.savefig(f"./results/1D-{self.model.name}.png")
        plt.cla()
        plt.close("all")

    def plot2D(self):
        self.get_data(n=40)
        if self.model.grid.dim != 2:
            raise Exception(f"ERROR: Plotting 2D but model has {self.model.grid.dim} dimensions")
            
        vmin = self.df[0].min()
        vmax = self.df[0].max()
        print(vmin, vmax)
        for t in self.df.columns:
            data = np.array(self.df[t].values)
            data = data.reshape((self.model.grid.res[1], self.model.grid.res[0]))
            plot_df = pd.DataFrame(data)
            g = sns.heatmap(plot_df, cmap='RdBu_r', vmin=vmin, vmax=vmax)
            g.invert_yaxis()
            plt.savefig(f"./results/2D-{self.model.name}-t{t:.3f}.png")
            plt.cla()
            plt.close("all")
    
    def plot2DVectorField(self, data, fname="2D-quiver"):
        #split data
        split_data = data.reshape(self.model.grid.nr_of_cells, 2)
        u_data = split_data[0:, 0]
        v_data = split_data[0:, 1]

        x,y = np.meshgrid(np.linspace(0,self.model.grid.domain[0],self.model.grid.res[0]),
                            np.linspace(0,self.model.grid.domain[1], self.model.grid.res[1]),
                            )
        
        u_data = u_data.reshape((self.model.grid.res[1], self.model.grid.res[0]))
        v_data = v_data.reshape((self.model.grid.res[1], self.model.grid.res[0]))

        fig = plt.figure()
        #plt.quiver(x,y,u_data,v_data, scale=1.5*10**5*self.model.viscosity*self.model.dt*self.model.grid.ds.mean())
        plt.quiver(x,y,u_data,v_data)
        plt.savefig(f"./results/{fname}.png")
        plt.cla()
        plt.close("all")
        
    def plot3DVectorField(self, data):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        x,y,z = np.meshgrid(np.linspace(0,self.model.grid.domain[0],self.model.grid.res[0]),
                            np.linspace(0,self.model.grid.domain[1], self.model.grid.res[1]),
                            np.linspace(0,self.model.grid.domain[2], self.model.grid.res[2]),
                            )

        split_data = data.reshape(self.model.grid.nr_of_cells, 3)
        u_data = split_data[0:, 0]
        v_data = split_data[0:, 1]
        w_data = split_data[0:, 2]

        u_data = u_data.reshape((self.model.grid.res[1], self.model.grid.res[0]))
        v_data = v_data.reshape((self.model.grid.res[1], self.model.grid.res[0]))
        w_data = w_data.reshape((self.model.grid.res[1], self.model.grid.res[0]))

        ax.quiver(x, y, z, u_data, v_data, w_data, scale=250*self.model.viscosity*self.model.dt*self.model.grid.ds.mean(), color = 'black')
        plt.savefig("./results/3D-quiver.png")
        plt.cla()
        plt.close("all")
        
    def plot2DContour(self, data, fname="2D-contour"):
        x,y = np.meshgrid(np.linspace(0,self.model.grid.domain[0],self.model.grid.res[0]),
                            np.linspace(0,self.model.grid.domain[1], self.model.grid.res[1]),
                            )
        
        p_data = data.reshape((self.model.grid.res[1], self.model.grid.res[0]))
        
        fig,ax = plt.subplots()
        cs = ax.contourf(x,y,p_data)
        cbar = fig.colorbar(cs)
        cbar.ax.set_ylabel('Pressure')
        plt.savefig(f"./results/{fname}.png")
        plt.cla()
        plt.close("all")
    
    def plotTimeStepIteration(self, t=0):
        self.plot2DVectorField(self.model.vfinal_result[0:, t], fname=f"vfinal-t{t}-vector")
        self.plot2DVectorField(self.model.vstar_result[0:, t], fname=f"vstar-t{t}-vector")
        self.plot2DContour(self.model.pressure_result[0:, t], fname=f"pressure-t{t}-contour")
        self.plot2DVectorField(self.model.vcorr_result[0:, t], fname=f"vcorr-t{t}-vector")
        self.plot2DVectorField(self.model.vfinal_result[0:, t+1], fname=f"vfinal-t{t+1}-vector")

class analyticalSolution(object):
    def __init__(self, model):
        self.model = model
    
    def analytical_p(self, coord):
        return np.prod([np.cos(k*x*np.pi*2/L) for k, x, L in zip(self.model.initial_wavenumbers, coord, self.model.grid.domain)])

    def test_equation(self,coord): 
        """
        Generally provides the value of the Q test equation for the given coordinates
        
        Test equation: Q = SUM[ c^2 ] * PROD [ cos(c*x_i) ]
        
        For 2D and a frequency fitted to the domain x:[0,L_x] and y:[0,L_y]: Q = ((k2pi/L_x)^2 + (k2pi/L_y)^2) * cos(k2pix/L_x) * cos(k2piy/L_y)

        :coord: array-like: coordinates of the point for which Q needs to be calculated
        :Returns: 
        """
        # Create cosus product of all dimensions
        cos_prod = self.analytical_p(coord)
        # Create terms in front of cosus 
        derivation_const = np.sum([-(k*np.pi*2/L)**2 for k, L in zip(self.model.initial_wavenumbers, self.model.grid.domain)])
        return derivation_const*cos_prod


