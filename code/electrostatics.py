import numpy as np
from iodata import load_one, dump_one

#https://github.com/molmod/molmod
from molmod.units import angstrom, debye


'''
Generation of a meshgrid, the boundaries are chosen such that
the molecule is padded on each side with a distance r_max.
'''
def generate_grid(R, r_max=7., min_points=1000.):
    # 1D array from -r_max to r_max to generate mesh.
    # The amount of steps is chosen such that the mesh
    # consists of at least min_points points.
    max_dist = np.max(np.abs(R)) + r_max
    grid_length = int(np.rint(np.power(min_points, 1/3.) + .5))
    grid_axis = np.linspace(-max_dist, max_dist, grid_length)
    
    grid = np.array(np.meshgrid(grid_axis, grid_axis, grid_axis, indexing='ij'))
    return grid

'''The efficient numpy way of computing distances'''
def distance_grid(grid, offset):
    return np.sqrt((grid[0] - offset[0])**2 +
                   (grid[1] - offset[1])**2 +
                   (grid[2] - offset[2])**2)
        
'''
Generation of the on-off mask to generate intermediate distance
gridpoints.
'''
def generate_mask(grid, R, Z, r_max=7.):
    # Deletion of zero-entries
    Z = np.array(Z[np.where(Z != 0)], dtype=int)
    R = np.array(R)[:len(Z)]
    
    mask = np.ones(np.shape(grid[0])).astype('int')
    mask_max = np.zeros(np.shape(grid[0])).astype('int')
    # List of VdW radii in element order.
    vdw_radii = np.array([1.2, 1.4, 0., 0., 0., 1.7, 1.55, 1.52, 1.47, 1.54,
                          0., 0., 0.,  2.1, 1.8, 1.8, 1.75, 1.88]) + 0.2
    for pos, z in zip(R, Z):
        dist_grid = distance_grid(grid, pos)
        
        # if gridpoint lays within the VdW radius, it
        # gets turned off.
        grid_min = np.where(dist_grid < vdw_radii[int(z)-1])
        mask[grid_min] = 0
        
        # Check if gridpoint lays outside r_max
        grid_max = np.where(dist_grid > r_max)
        mask_max[grid_max] += 1
    
    # If gridpoint lays outside r_max for every
    # atom, it gets turned off.
    max_idx = np.where(mask_max == len(Z))
    mask[max_idx] = 0
    return mask

'''
Function for generating gridpoints that lay in the
intermediate range of 2-7 angstrom.
'''
def generate_coords(R, Z, r_max, min_points):
    grid = generate_grid(R, r_max, min_points)
    mask = generate_mask(grid, R, Z, r_max)
    
    # Put grid in a format so that psi4 can read it in.
    grid = np.moveaxis(grid,0,-1)
    
    # Psi4 will look for a grid.dat file when calculating
    # the ESP using the density.
    np.savetxt('grid.dat', grid[np.where(mask == 1)])
    return np.moveaxis(grid[np.where(mask == 1)], 0, 1)

'''Computation of the classical electrostatic pot.'''
def calculate_ESP(grid, R, Z):
    dist_grid = np.zeros((len(R),np.shape(grid)[1]))
    for i,r in enumerate(R):
        dist_grid[i] = distance_grid(grid,r)
    inv_dist_grid = np.power(dist_grid, -1)
    grid_esp = np.einsum('ij,i->j', inv_dist_grid, Z)
    return grid_esp
        
        