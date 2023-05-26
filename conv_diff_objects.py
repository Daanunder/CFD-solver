import numpy as np

class convDiffModel():
    def __init__(self, dimensions, resolution, domain, faces):
        # Initialize model
        self.grid = gridObj(
                dimensions, 
                resolution,
                domain,
                faces
                )

        self.boundaryConditions = boundaryConditions(self)

        self.divConv = convectionSchemes(self).explicitCentral
        self.divDiff = diffusionSchemes(self).explicit
        self.dt = 1
        self.u = [1]
        self.kappa = 1

        # Setup matrix
        self.create_matrices()

        # solve
        self.t_end = 10
        self.result = np.zeros((int(self.t_end/self.dt), self.grid.vector_size))
        self.setup_initial_conditions()
    
    def create_matrices(self):
        # Initialize global matrices 
        self.LHS = np.zeros((self.grid.vector_size, self.grid.vector_size))
        self.RHS = np.zeros((self.grid.vector_size, self.grid.vector_size))
        
        for ID, c in self.grid.cells.items()    :
            lhs = np.zeros(self.grid.vector_size)
            rhs = np.zeros(self.grid.vector_size)

            # positive side cell and negative side cell neighbours in each dimension
            for d, (nsn, psn) in enumerate(c.neighbors):
                nsnc = self.grid.locations.get(nsn)
                psnc = self.grid.locations.get(psn)

                # Get discrete contributions of all terms in dimension d
                conv = self.divConv(d)
                diff = self.divDiff(d)
                lhs[ID] += conv[0][1]
                rhs[ID] += diff[1][1]

                # TODO: Add boundary.get()
                
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
            
            self.RHS[ID] = rhs
            self.LHS[ID] = lhs
        
    def setup_initial_conditions(self):
        pass
        
    def solve_matrix(self):
        pass

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
            loc = np.zeros(self.dim)
            loc[0] = i % self.res[0]
            loc[1:] = [int((i / s) % r) for s,r in zip(self.span_size[:-1], self.res[1:])]
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

class boundaryConditions(object):
    def __init__(self, model):
        self.available_bc = {
                "periodic": self.periodicBC
                }
        self.model = model

    def get_contribution(self, cell_face, bc_type):
        return self.available_bc.get(bc_type)(cell_face)

    def periodicBC(self, cell_face): 
        """
        Should return [lhs, rhs] of size model.vector_size

        Depends on divConv, divDiff, divDt

        Periodic BC so cell at the other end of this dimension contributes to the current cell; 

        The other end of this dimension can be found by res[d]*span[d] for all d 

        """
        lhs = np.zeros(self.model.grid.vector_size)
        rhs = np.zeros(self.model.grid.vector_size)
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
        return lhs, rhs

class convectionSchemes(object):
    """
    Provide discretization functions that return [LHS, RHS] for the local matrix
    """
    def __init__(self, model):
        self.model = model

    def explicitCentral(self, dim):
        #return np.array([0, 0, 0]), np.array([1/2, 0, -1/2]) * self.model.dt/self.model.grid.cellVolume * self.model.u[dim]
        return np.array([0, 0, 0]), np.array([-1/2, 0, 1/2]) 

    def implicitCentral(self, dim):
        #return np.array([1/2, 0, -1/2]) * -1 * self.model.dt/self.model.grid.cellVolume * self.model.u[dim], np.array([0, 0, 0]) 
        return np.array([-1/2, 0, 1/2]) * -1 , np.array([0, 0, 0]) 

    def explicitUpwindPos(self, dim):
        #return np.array([0, 0, 0]), np.array([-1/2, 1/2, 0]) * self.model.dt/self.model.grid.cellVolume * self.model.u[dim]
        return np.array([0, 0, 0]), np.array([-1/2, 1/2, 0]) 

    def implicitUpwindPos(self, dim):
        #return np.array([-1/2, 1/2, 0]) * -1 * self.model.dt/self.model.grid.cellVolume * self.model.u[dim], np.array([0, 0, 0]) 
        return np.array([-1/2, 1/2, 0]) * -1 , np.array([0, 0, 0]) 

    def explicitUpwindNeg(self, dim):
        #return np.array([0, 0, 0]), np.array([0, -1/2, 1/2]) * self.model.dt/self.model.grid.cellVolume * self.model.u[dim]
        return np.array([0, 0, 0]), np.array([0, -1/2, 1/2]) 

    def implicitUpwindNeg(self, dim):
        #return np.array([0, -1/2, 1/2]) * -1 * self.model.dt/self.model.grid.cellVolume * self.model.u[dim], np.array([0, 0, 0]) 
        return np.array([0, -1/2, 1/2]) * -1 , np.array([0, 0, 0]) 
 
class diffusionSchemes(object):
    """
    Provide discretization functions that return [LHS, RHS] for the local matrix
    """
    def __init__(self, model):
        self.model = model

    def explicit(self, dim):
        #return [0, 0, 0], [1, -2, 1] * self.model.dt/self.model.grid.cellVolume * self.model.kappa/self.model.grid.ds[dim]
        return [0, 0, 0], [1, -2, 1]

    def implicit(self, dim):
        #return [1, -2, 1] * -1 * self.model.dt/self.model.grid.cellVolume * self.model.kappa/self.model.grid.ds[dim], [0, 0, 0]
        return [1, -2, 1] * -1, [0, 0, 0]
