import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
 
#TODO: 
# - Add fixed boundary condition
# - Add analytical solution
# CHECKS:
#- Test with analytical solution
#- Conservation ( total and at fluxes)
#- Consistency
#- Stability (CFL)

class convDiffModel():
    """
    Solver for the convection diffusion equations in multiple dimensions
        
    Generally solves; Dt = divConv + divDiff

    Periodic and constant boundary conditions available

    """
    def __init__(self, modelName):
        # Initialize model
        self.get_basic_settings(modelName)
        self.produce_error = False

        # Init grid
        self.grid = gridObj(
                self.settings.get("dimensions"),
                self.settings.get("resolution"),
                self.settings.get("domain"),
                self.settings.get("faces")
                )
        
        # Do initial checks
        self.check_initial_stability()

        # Set boundary conditions
        self.fixedFaceValues = self.settings.get("fixedFaceValues")
        self.boundaryConditions = boundaryConditions(self)
        
        # Set numerical schemes
        self.divConv = convectionSchemes(self).explicitCentral
        self.divDiff = diffusionSchemes(self).explicit

        # Setup matrix (BC are applied here)
        self.create_matrices()

        # Set initial conditions
        self.timesteps = np.arange(0, self.t_end+self.dt, self.dt)
        self.result = np.zeros((self.grid.vector_size, self.timesteps.shape[0]))
        self.initialConditions = initialConditions(self).sinusoid(self.initial_amplitudes, self.initial_wavenumbers)

        # Visualtisation
        self.analyticalSolution = analyticalSolution(self)
        self.plotter = visualisationObj(self)
    
    def get_basic_settings(self, modelName):
        with open('working_models.json') as json_file:
            self.settings = json.load(json_file)[modelName]

        self.name = modelName
        self.initial_amplitudes = np.array(self.settings.get("initial_amplitudes"))
        self.initial_wavenumbers = np.array(self.settings.get("initial_wavenumbers"))
        self.fixedFaceValues = np.array(self.settings.get("fixedFaceValues"))
        
        self.t_end = self.settings.get("t_end")
        self.dt = self.settings.get("dt")
        self.u = np.array(self.settings.get("u"))
        self.kappa = np.array(self.settings.get("kappa"))

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

    def create_matrices(self):
        # Initialize global matrices 
        self.LHS = np.zeros((self.grid.vector_size, self.grid.vector_size))
        self.RHS = np.zeros((self.grid.vector_size, self.grid.vector_size))
        self.RHS_const = np.zeros((self.grid.vector_size, self.grid.vector_size))
        
        for ID, c in self.grid.cells.items():
            lhs = np.zeros(self.grid.vector_size)
            rhs = np.zeros(self.grid.vector_size)
            rhs_const = np.zeros(self.grid.vector_size)

            # positive side cell and negative side cell neighbours in each dimension
            for d, (nsn, psn) in enumerate(c.neighbors):
                nsnc = self.grid.locations.get(nsn)
                psnc = self.grid.locations.get(psn)

                # Get discrete contributions of all terms in dimension d
                conv = self.divConv(d)
                diff = self.divDiff(d)

                # Add diagonal conv and diff terms
                lhs[ID] += conv[0][1]
                rhs[ID] += conv[1][1]

                lhs[ID] += diff[0][1]
                rhs[ID] += diff[1][1]

                # nsnc = negative side neighbouring cell 
                if not nsnc == None: # remove if boundary dict is available
                    lhs[nsnc] += conv[0][0]
                    rhs[nsnc] += conv[1][0]

                    #print(f"nsnc", d, nsnc, ID, conv[1][0])
                    lhs[nsnc] += diff[0][0]
                    rhs[nsnc] += diff[1][0]

                else:
                    # nsn = cellFace
                    cell_face_bc = self.grid.cellFaces.get(nsn)
                    bc_contribution = self.boundaryConditions.get_contribution(nsn, cell_face_bc)
                    lhs += bc_contribution[0]
                    rhs += bc_contribution[1]
                    rhs_const += bc_contribution[2]
                
                # psnc = positive side neighbouring cell 
                if not psnc == None: # remove if boundary dict is available
                    lhs[psnc] += conv[0][2]
                    rhs[psnc] += conv[1][2]

                    #print("psnc", d, psnc, ID, conv[1][2])
                    lhs[psnc] += diff[0][2]
                    rhs[psnc] += diff[1][2]

                else:
                    cell_face_bc = self.grid.cellFaces.get(psn)
                    bc_contribution = self.boundaryConditions.get_contribution(psn, cell_face_bc)
                    lhs += bc_contribution[0]
                    rhs += bc_contribution[1]
                    rhs_const += bc_contribution[2]
            
            self.LHS[ID] = lhs
            self.RHS[ID] = rhs
            self.RHS_const[ID] = rhs_const
        
    def solve(self):
        # Set t0
        if self.produce_error:
            self.error = np.zeros((self.grid.vector_size, self.timesteps.shape[0]))

        tn = self.result[:,0]
        for ti in range(1, self.timesteps.shape[0]):
            # create b with tn
            RHS = (np.identity(self.grid.vector_size) + self.RHS)
            b = np.dot(RHS, tn) + np.dot(self.RHS_const, np.ones(self.grid.vector_size))

            # create A
            A = np.identity(self.grid.vector_size) - self.LHS

            # solve A, b to obtain tn+1
            tn = np.linalg.solve(A, b)
            self.result[0:, ti] = tn



class gridObj(object):
    """Grid definition of a CFD program"""

    def __init__(self, dim, res, domain, faces):
        """
        :dim: int
            number of dimensions 
        :res: array_like
            number of cells in each direction, should be of size 'dim'
        :domain: array_like
            size of the domain in each direction, should be of size 'dim'
        :faces: array of size [dim, 2]
            boundary condition types for each face of the domain, for both the positive- and negative normal direction.
        """
        self.dim = dim
        self.res = np.array(res)
        self.domain = np.array(domain)
        self.ds = self.domain/self.res
        self.cellVolume = np.prod(self.ds)
        self.vector_size = np.prod(self.res)
        self.span_size = np.zeros(self.dim)
        self._setup_span_size()
        self.cells = dict() 
        self.locations = dict()
        self.faces = faces
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
        for i in range(self.vector_size):
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
                    self.cellFaces[nsn_str] = self.faces[n][0]

                posside_neighbor = [int(t) for t in loc]
                if l + 1 < self.res[n]:
                    posside_neighbor[n] = int(l + 1)
                    psn_str = ".".join([str(k) for k in posside_neighbor])
                else:
                    posside_neighbor[n] = "+"
                    psn_str = ".".join([str(k) for k in posside_neighbor])
                    self.cellFaces[psn_str] = self.faces[n][1]

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
                "periodic": self.periodicBC,
                "fixedValue": self.fixedValueBC,
                }
        self.model = model

    def get_contribution(self, cell_face, bc_type, **kwargs):
        return self.available_bc.get(bc_type)(cell_face, **kwargs)

    def periodicBC(self, cell_face, **kwargs): 
        """
        Periodic BC so cell at the other end of this dimension contributes to the current cell; 

        The other end of this dimension can be found by res[d]*span[d] for all d 
        
        :Returns: [lhs, rhs, rhs_const] which are all arrays of size model.vector_size
        """
        lhs = np.zeros(self.model.grid.vector_size)
        rhs = np.zeros(self.model.grid.vector_size)
        rhs_const = np.zeros(self.model.grid.vector_size)
        loc = cell_face.split(".")
        for d,l in enumerate(loc):
            conv_contribution = self.model.divConv(d)
            diff_contribution = self.model.divDiff(d)
            #print(conv_contribution[1], diff_contribution[1])
            if l == "-":
                cell_loc = cell_face.replace("-", "0")
                opposite = int(self.model.grid.locations[cell_loc] + self.model.grid.span_size[d] - self.model.grid.span_size[d]/self.model.grid.res[d])
                #print("-", cell_loc, d, opposite)
                lhs[opposite] = conv_contribution[0][0] + diff_contribution[0][0]
                rhs[opposite] = conv_contribution[1][0] + diff_contribution[1][0]
            elif l == "+":
                cell_loc = cell_face.replace("+", str(self.model.grid.res[d]-1))
                opposite = int(self.model.grid.locations[cell_loc] - self.model.grid.span_size[d] + self.model.grid.span_size[d]/self.model.grid.res[d])
                #print("+", cell_loc, d, opposite)
                lhs[opposite] = conv_contribution[0][2] + diff_contribution[0][2]
                rhs[opposite] = conv_contribution[1][2] + diff_contribution[1][2]
        return lhs, rhs, rhs_const

    def fixedValueBC(self, cell_face, **kwargs):
        """
        Fixed BC, so the boundary cell face will have a fixed value, as stored in the model self.model.fixedFaceValues

        :Returns: [lhs, rhs, rhs_const] which are all arrays of size model.vector_size
        """
        lhs = np.zeros(self.model.grid.vector_size)
        rhs = np.zeros(self.model.grid.vector_size)
        rhs_const = np.zeros(self.model.grid.vector_size)
        loc = cell_face.split(".")
        for d,l in enumerate(loc):
            conv_contribution = self.model.divConv(d)
            diff_contribution = self.model.divDiff(d)
            if l == "-":
                cell_loc = cell_face.replace("-", "0")
                cell_id = self.model.grid.locations[cell_loc]
                lhs[cell_id] = conv_contribution[0][0] + diff_contribution[0][0]
                rhs[cell_id] = conv_contribution[1][0] + diff_contribution[1][0]
                rhs_const[cell_id] = (conv_contribution[1][0] + diff_contribution[1][0])*2*self.model.fixedFaceValues[d][0]
            elif l == "+":
                cell_loc = cell_face.replace("+", str(self.model.grid.res[d]-1))
                cell_id = self.model.grid.locations[cell_loc]
                lhs[cell_id] = conv_contribution[0][2] + diff_contribution[0][2]
                rhs[cell_id] = conv_contribution[1][2] + diff_contribution[1][2]
                rhs_const[cell_id] = (conv_contribution[1][2] + diff_contribution[1][2])*2*self.model.fixedFaceValues[d][1]
        
        return lhs, rhs, rhs_const

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

class convectionSchemes(object):
    """
    Provide convection discretization functions based on the given dimension. Returns the terms for the local matrix based on temporal and spatial discretization scheme.

    :Returns: [lhs, rhs, rhs_const]
    """
    def __init__(self, model):
        self.model = model

    def explicitCentral(self, dim):
        return np.array([0, 0, 0]), np.array([1/2, 0, -1/2]) * self.model.dt/self.model.grid.cellVolume * self.model.u[dim]

    def implicitCentral(self, dim):
        return np.array([1/2, 0, -1/2]) * self.model.dt/self.model.grid.cellVolume * self.model.u[dim], np.array([0, 0, 0])

    def explicitUpwindPos(self, dim):
        return np.array([0, 0, 0]), np.array([1/2, -1/2, 0]) * self.model.dt/self.model.grid.cellVolume * self.model.u[dim]

    def implicitUpwindPos(self, dim):
        return np.array([1/2, -1/2, 0]) * self.model.dt/self.model.grid.cellVolume * self.model.u[dim], np.array([0, 0, 0])

    def explicitUpwindNeg(self, dim):
        return np.array([0, 0, 0]), np.array([0, -1/2, 1/2]) * self.model.dt/self.model.grid.cellVolume * self.model.u[dim]

    def implicitUpwindNeg(self, dim):
        return np.array([0, -1/2, 1/2]) * self.model.dt/self.model.grid.cellVolume * self.model.u[dim], np.array([0, 0, 0])
 
class diffusionSchemes(object):
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
        self.x = np.linspace(0, self.model.grid.domain[0], 100)

    def convDiff1D_periodic(self,t):
        return np.sin(np.pi*2/self.model.grid.domain[0]*self.model.initial_wavenumbers[0]*(self.x - self.model.u[0]*t))*np.e**(-1*self.model.kappa*(np.pi*2/self.model.grid.domain[0])**2*self.model.initial_wavenumbers[0]**2*t)

