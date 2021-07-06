import numpy as np
from scipy.io import loadmat
from localizer_hpc import Localizer

#https://github.com/molmod/molmod
from molmod.units import angstrom, debye

import matplotlib.pyplot as plt

#importing the centered QM7 database
Z = np.load('../data/molecule_geometries/benchmarking_dataset_Z.npy')
R = np.load('../data/molecule_geometries/benchmarking_dataset_R.npy')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run optimization for 1 system')
    parser.add_argument('-index', default=0, type=int)
    parser.add_argument('-scheme', default='FB', type=str)
    parser.add_argument('-penalty', default=0., type=float)
    args = parser.parse_args()
    
    foldername = '../data/xyz_files/test_runs/'
    
    loc = Localizer(Z[args.index],R[args.index],args.index)
    loc.set_scheme(name=args.scheme, p=args.penalty)
    conv = loc.optimize_line_search(nsteps=1000, psi4_guess=False)
    
    quad_compared = loc.compare_quadrupole()
    quad_loc_qm7 = loc.total_loc_quadrupole.flatten()
    quad_qm7 = loc.total_quadrupole.flatten()
    
    loc.write_centers(folder=foldername)

