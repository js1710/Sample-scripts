cimport cython
cimport numpy as np
import numpy as np
from libcpp.vector cimport vector

FTYPE = np.float32
ctypedef np.float32_t FTYPE_t

def get_neighbours_pyx(tuple coords, int nx, int ny, np.ndarray[long, ndim=2] cells):
    '''Cython wrapper for C++-like implementation of neighbour algorithm 'get_neighbours_c' '''
    return get_neighbours_c(coords, nx, ny, cells)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef get_neighbours_c(tuple coords, int nx, int ny, np.ndarray[long, ndim=2] cells):
    '''
    returns the 3x3 grid of cells around the cell that contains the 2D point 'coords' that are occupied. Note 
    that one cell will only ever contain one point.
    :param coords: 2D indices of cell in which point residues
    :param nx: number of cells in x dimension
    :param ny: number of cells in y dimension
    :param cells: array of cells which contain the coordinates of points that have already been assigned
    :return: list of points near to 'coords'
    '''
    cdef int i, j, neighbour_cell, x, y
    cdef vector[int] neighbours
    for i in range(-2, 3):
        x = coords[0] + i
        x -= int(x / nx) * nx
        for j in range(-2, 3):
            if abs(i) == 2 and abs(j) == 2:
                continue

            y = coords[1] + j
            y -= int(y/ny) * ny
            neighbour_cell = cells[x, y]
            if neighbour_cell > -1:
                neighbours.push_back(neighbour_cell)
    return neighbours


class PoissonDisc():
    def __init__(self, size, r, k=30):
        '''
        Class for the Poisson Disk algorithm in a 2D periodic box
        :param size: dimensions of 2D box
        :param r: minimum radius between points
        :param k: maximum number of trial points per generated point
        '''
        self.size = np.array(size, dtype=float)
        self.r = r
        self.k = k
        #cell side length
        self.l = r / np.sqrt(2.)
        self.nx, self.ny = int(size[0] / self.l) + 1, int(size[1] / self.l) + 1
        self.pt_base = np.zeros(2)
        self._rsq = r**2


    def clear(self):
        '''
        Clears any previous generated points
        :return:
        '''
        self.cells = np.zeros((self.nx, self.ny), dtype=int)
        self.cells.fill(-1)

    def get_cell_inds(self, pt):
        '''
        Returns the indices of the cell that contains point 'pt'
        :param pt:
        :return:
        '''
        return  int(pt[0] / self.l), int(pt[1] / self.l)

    def pbc(self, pt):
        '''
        enforces cuboid periodic boundary conditions
        :param pt: 2D point
        :return: 2D point within box 'self.size'
        '''
        pt -= np.floor(pt / self.size) * self.size
        return pt

    def generate_point(self, refpt):
        '''
        Generates trial points until a valid point that is not within radius 'self.r' of any other point is
        found or trials exceeds 'k'.
        :param refpt: reference point
        :return:
        '''
        i = 0
        pt = np.copy(self.pt_base) #minimize the number of times array is initialised
        while i < self.k:
            rho = np.random.uniform(self.r, 2*self.r )
            theta = np.random.uniform(0, 2*np.pi)
            pt[0] = refpt[0] + rho*np.cos(theta)
            pt[1] = refpt[1] + rho*np.sin(theta)
            pt = self.pbc(pt)
            if self.point_valid(pt):
                return pt
            i +=1
        return False

    def point_valid(self, pt):
        '''
        Determines if a trial point 'pt' is not too close to any previously generated point
        :param pt: trial point
        :return: True if point is a valid candidate
        '''
        cell_coords = self.get_cell_inds(pt)
        for idx in self.get_neighbours(cell_coords):
            nearby_pt = self.samples[idx]
            # Squared distance between candidate point, pt, and this nearby_pt.
            dr = nearby_pt - pt

            dr -= np.rint(dr/self.size) * self.size
            distance2 = np.sum(np.square(dr))

            if distance2 < self._rsq:
                # The points are too close, so pt is not a candidate.

                return False
        return True


    def get_neighbours(self, coords):
        '''
        Top level wrapper for cython implementation 'get_neighbours_pyx' of neighbour algorithm
        :param coords: tuple of the cell indices that contain a trial point
        :return:
        '''
        return get_neighbours_pyx(coords, self.nx, self.ny, self.cells)



    def sample(self):
        '''
        Generates points in a 2D peridoic box that are not within 'self.r' of each other. Results are returned to
        'self.samples'
        :return:
        '''
        self.clear()
        #pick ransom point to start with
        pt = np.array([np.random.uniform(0, self.size[0]), np.random.uniform(0, self.size[0])])
        self.samples = [pt]
        self.cells[self.get_cell_inds(pt)] = 0
        active = [0]
        nsamples = 1
        while active:
            #choose random reference point from the active list
            idx = np.random.choice(active)
            refpt = self.samples[idx]
            #pick a new point relative to reference point
            pt = self.generate_point( refpt)
            if pt is not False:
                # point pt is valid: add it to samples list and mark as active
                self.samples.append(pt)
                nsamples = len(self.samples) - 1
                active.append(nsamples)
                self.cells[self.get_cell_inds(pt)] = nsamples
            else:
                # Cannot find a valid point near refpt and so we remove
                # refpt from the list of "active" points
                active.remove(idx)

        self.samples = np.array(self.samples)