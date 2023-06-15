import numpy as np
import matplotlib.pyplot as plt
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
        # Initialize model
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
        self.test_poisson()

        # Set initial conditions
        self.timesteps = np.arange(0, self.t_end+self.dt, self.dt)
        self.current_timestep_index = 0
        self.result = np.zeros((self.grid.vector_size, self.timesteps.shape[0]))
        self.initialConditions = initialConditions(self).sinusoid(self.initial_amplitudes, self.initial_wavenumbers)

    def get_basic_settings(self, modelName):
        with open('working_models.json') as json_file:
            self.settings = json.load(json_file)[modelName]

        self.log = self.settings.get("log") == 1

        self.name = modelName
        self.initial_amplitudes = np.array(self.settings.get("initial_amplitudes"))
        self.initial_wavenumbers = np.array(self.settings.get("initial_wavenumbers"))
        self.fixedFaceValues = np.array(self.settings.get("fixedFaceValues"))
        
        self.t_end = self.settings.get("t_end")
        self.dt = self.settings.get("dt")
        self.u = np.array(self.settings.get("u"))
        self.kappa = np.array(self.settings.get("kappa"))
        self.viscosity = self.settings.get("viscosity")

    def check_initial_stability(self):
        # Stability checks
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
        # Initialize global matrices 
        self.POIS_LHS = np.zeros((self.grid.vector_size, self.grid.vector_size))
        self.POIS_RHS = np.zeros((self.grid.vector_size, self.grid.vector_size))
        self.POIS_RHS_const = np.zeros((self.grid.vector_size, self.grid.vector_size))
        
        for ID, c in self.grid.cells.items():
            lhs = np.zeros(self.grid.vector_size)
            rhs = np.zeros(self.grid.vector_size)
            rhs_const = np.zeros(self.grid.vector_size)

            # positive side cell and negative side cell neighbours in each dimension
            for d, (nsn, psn) in enumerate(c.neighbors):
                nsnc = self.grid.locations.get(nsn)
                psnc = self.grid.locations.get(psn)

                # Get discrete contributions of all terms in dimension d
                laplace_pres = self.laplacePres(d)
                div_mom  = self.divMomentum(d)

                # Add diagonal conv and diff terms
                lhs[ID] += div_mom[0][1] + laplace_pres[0][1]
                rhs[ID] += div_mom[1][1] + laplace_pres[1][1]

                # nsnc = negative side neighbouring cell 
                if not nsnc == None: # remove if boundary dict is available
                    lhs[nsnc] += div_mom[0][0] + laplace_pres[0][0]
                    rhs[nsnc] += div_mom[1][0] + laplace_pres[1][0]

                else:
                    pressure_bc = self.grid.cellFaces.get(nsn).get("pressure")
                    velocity_bc = self.grid.cellFaces.get(nsn).get("velocity")
                    pressure_bc_contribution = self.boundaryConditions.get_contribution(nsn, pressure_bc, "pressure", "poisson")
                    #velocity_bc_contribution = self.boundaryConditions.get_contribution(nsn, velocity_bc, "velocity", "poisson")
                    lhs += pressure_bc_contribution
                    #rhs += velocity_bc

                # psnc = positive side neighbouring cell 
                if not psnc == None: # remove if boundary dict is available
                    lhs[psnc] += div_mom[0][2] + laplace_pres[0][2]
                    rhs[psnc] += div_mom[1][2] + laplace_pres[1][2]

                else:
                    pressure_bc = self.grid.cellFaces.get(psn).get("pressure")
                    velocity_bc = self.grid.cellFaces.get(psn).get("velocity")
                    pressure_bc_contribution = self.boundaryConditions.get_contribution(psn, pressure_bc, "pressure", "poisson")
                    #velocity_bc_contribution = self.boundaryConditions.get_contribution(psn, velocity_bc, "velocity", "poisson")
                    lhs += pressure_bc_contribution
                    #rhs += velocity_bc
            
            self.POIS_LHS[ID] = lhs
            self.POIS_RHS[ID] = rhs
            self.POIS_RHS_const[ID] = rhs_const
    
    def create_pressure_correction_matrices(self):
        pass

    def test_poisson(self):
        # Set Q
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
        # Calculate 
        # create b with tn
        #RHS = (np.identity(self.grid.vector_size) + self.RHS)
        #b = np.dot(RHS, tn) + np.dot(self.RHS_const, np.ones(self.grid.vector_size))
        # create A
        #A = np.identity(self.grid.vector_size) - self.LHS

        # solve A, b to obtain tn+1
        #tn = np.linalg.solve(A, b)
        #self.result[0:, ti] = tn
        """
       
        # Update non linear advection terms in RHS_adv
        for ID,c in self.grid.cells.items():
            for d in range(self.grid.dim):
                rank = ID*self.grid.dim + d
                self.EOM_RHS_adv[rank] = self.advVel(d, c)

        pass

    def solve_pressure_poisson(self):
        pass

    def calc_velocity_correction(self):
        pass
    
    def correct_velocities(self):
        pass

    def solve(self):
        # Set t0
        tn = self.result[:,0]
        self.current_timestep_index = 0
        for ti in range(1, self.timesteps.shape[0]):
            # Solve EOM: vn+1 = EOM(u,v)
            # TODO:
            # - Implement boundary conditions
            self.solve_eom()

            # Calculate new pressures pn+1 = Poisson(vn+1)
            # TODO: 
            # - Implement RHS grad(vn+1) matrices
            # - Implement boundary conditions
            self.solve_pressure_poisson()

            # Calculate velocity correction: vc = - dt/rho * grad(pc)
            # TODO: 
            # - Implement matrices grad(pn+1)
            # - Implement boundary conditions
            self.calc_velocity_correction()

            # Correct velocities: vn+1 = vn+1 + vc
            # TODO:
            # - Implement velocity correction
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
            self.model.result[ID][0] = np.prod(sin_array)


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
            if not negbound:
                advectiveChange += np.sum(outer_prod * self.negative_coefficient_matrix) * self.model.grid.cellVolume/self.model.grid.ds[ddim]
            if not posbound:
                advectiveChange += np.sum(outer_prod * self.positive_coefficient_matrix) * self.model.grid.cellVolume/self.model.grid.ds[ddim]

        return advectiveChange 
        
    def getVectorValuesFromVector(self, vector, negbound, posbound):
        # TODO: possibly implement boundary conditions
        V_neg_rank, V_i_rank, V_pos_rank = vector
        if negbound: 
            V_vector = self.result[:, self.current_timestep_index][[V_i_rank, V_pos_rank]]
            V_vector[0] = 0
        elif posbound: 
            V_vector = self.result[:, self.current_timestep_index][[V_neg_rank, V_i_rank]]
            V_vector[2] = 0
        else:
            V_vector = self.result[:, self.current_timestep_index][[V_neg_rank, V_i_rank, V_pos_rank]]
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
        
        u_vector, negbound_normal, posbound_normal = self.model.getLocalVector(dim, dim, c)
        if negbound_normal:
            # implement BC du/dx i-1/2 - Always zero?
            selector = np.s_[1:]
            #u_bc_face = u_vector[0]
            #rhs_const += self.model.boundaryConditions.get_contribution(u_bc_face, "fixedWall", "velocity", "momentum"))
        elif posbound_normal:
            # implement BC du/dx i+1/2 - Always zero?
            selector = np.s_[:-1]
            #u_bc_face = u_vector[2]
            #print(u_bc_face, self.model.boundaryConditions.get_contribution(u_bc_face, "fixedWall", "velocity", "momentum"))
        
        u_vector = u_vector[selector]

        if not negbound_normal:
            rhs[u_vector] += self.normal_negative_coefficient_matrix[selector] * self.model.grid.cellVolume/self.model.grid.ds[dim]/self.model.grid.ds[dim] * self.model.viscosity

        if not posbound_normal:
            rhs[u_vector] += self.normal_positive_coefficient_matrix[selector] * self.model.grid.cellVolume/self.model.grid.ds[dim]/self.model.grid.ds[dim] * self.model.viscosity

        for d in range(self.model.grid.dim):
            negbound_shear = False
            posbound_shear = False
            if d == dim:
                continue
        
            v_vector_i = self.model.getLocalVector(d, dim, c)

            nsn, psn =  c.neighbors[d]
            nsn_id = self.model.grid.locations.get(nsn)
            if not nsn_id == None:
                nsnc = self.model.grid.cells.get(nsn_id)
                u_vector_n, negbound_temp, posbound_temp = self.model.getLocalVector(dim, dim, nsnc)
                if not negbound_temp == negbound_normal or not posbound_temp == posbound_normal:
                    raise Exception("Something is wrong 1")
                if negbound_normal:
                    # TODO: Apply BC in approximation of gradient 1.5 \Delta y
                    un_bc_face = u_vector_n[0]
                    #print(un_bc_face, self.model.boundaryConditions.get_contribution(un_bc_face, "fixedWall", "velocity", "momentum"))
                if posbound_normal:
                    # TODO: Apply BC in approximation of gradient 1.5 \Delta y
                    un_bc_face = u_vector_n[2]
                    #print(un_bc_face, self.model.boundaryConditions.get_contribution(un_bc_face, "fixedWall", "velocity", "momentum"))

                u_vector_n = u_vector_n[selector]

                rhs[u_vector_n] += self.shear_negative_coefficient_matrix[selector] * self.model.grid.cellVolume/self.model.grid.ds[dim]*self.model.viscosity * 1/2 * 1/(self.model.grid.ds[d]*2)
                rhs[u_vector] += self.shear_negative_coefficient_matrix[selector] * self.model.grid.cellVolume/self.model.grid.ds[dim]*self.model.viscosity * 1/2 * 1/(self.model.grid.ds[d]*2)

            else:
                # TODO: implement BC du/dy i-1/2
                negbound_shear = True

            psn_id = self.model.grid.locations.get(psn)
            if not psn_id == None:
                psnc = self.model.grid.cells.get(psn_id)
                u_vector_p, negbound_temp, posbound_temp = self.model.getLocalVector(dim, dim, psnc)
                if not negbound_temp == negbound_normal or not posbound_temp == posbound_normal:
                    raise Exception("Something is wrong 2")
                if negbound_normal:
                    # TODO: Apply BC in approximation of gradient 1.5 \Delta y
                    un_bc_face = u_vector_p[0]
                    #print(un_bc_face, self.model.boundaryConditions.get_contribution(un_bc_face, "fixedWall", "velocity", "momentum"))
                if posbound_normal:
                    # TODO: Apply BC in approximation of gradient 1.5 \Delta y
                    un_bc_face = u_vector_p[2]
                    #print(un_bc_face, self.model.boundaryConditions.get_contribution(un_bc_face, "fixedWall", "velocity", "momentum"))
                
                u_vector_p = u_vector_p[selector]
                rhs[u_vector_p] += self.shear_positive_coefficient_matrix[selector] * self.model.grid.cellVolume/self.model.grid.ds[dim]*self.model.viscosity * 1/2 * 1/(self.model.grid.ds[d]*2)
                rhs[u_vector] += self.shear_positive_coefficient_matrix[selector] * self.model.grid.cellVolume/self.model.grid.ds[dim]*self.model.viscosity * 1/2 * 1/(self.model.grid.ds[d]*2)

            else:
                # TODO: implement BC du/dy i+1/2
                posbound_shear = True

        return lhs, rhs, rhs_const

    def implicit(self, dim, c):
        # simply returns the matrix in different order
        rhs, lhs, rhs_const = self.explicit(dim, c)
        return lhs, rhs, rhs_const

class gradientPressureSchemes(object):
    """
    Provide diffusion discretization functions based on the given dimension. Returns the terms for the local matrix based on temporal and spatial discretization scheme.

    :Returns: [lhs, rhs, rhs_const]
    """
    def __init__(self, model):
        self.model = model

    def explicit(self, dim):
        return np.array([0, 0, 0]), np.array([1, -2, 1]) * self.model.dt/self.model.grid.cellVolume * self.model.kappa[dim]/self.model.grid.ds[dim]

    def implicit(self, dim):
        #return [1, -2, 1] * -1 * self.model.dt/self.model.grid.cellVolume * self.model.kappa/self.model.grid.ds[dim], [0, 0, 0]
        return np.array([1, -2, 1]) * self.model.dt/self.model.grid.cellVolume * self.model.kappa[dim]/self.model.grid.ds[dim], np.array([0, 0, 0])

class laplacianPressureSchemes(object):
    """
    Provide diffusion discretization functions based on the given dimension. Returns the terms for the local matrix based on temporal and spatial discretization scheme.

    :Returns: [lhs, rhs, rhs_const]
    """
    def __init__(self, model):
        self.model = model

    def explicit(self, dim):
        return np.array([1, -2, 1]) / self.model.grid.ds[dim]**2, np.array([0, 0, 0])

class divMomentumSchemes(object):
    """
    Provide diffusion discretization functions based on the given dimension. Returns the terms for the local matrix based on temporal and spatial discretization scheme.

    :Returns: [lhs, rhs, rhs_const]
    """
    def __init__(self, model):
        self.model = model

    def explicit(self, dim):
        return np.array([0, 0, 0]), np.array([0, 0, 0])


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


