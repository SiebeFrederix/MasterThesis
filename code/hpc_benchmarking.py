import numpy as np
import scipy.linalg
from scipy.stats import ortho_group
from scipy.io import loadmat
from localizer import Localizer
import psi4
from iodata import load_one, dump_one

#https://github.com/molmod/molmod
from molmod.units import angstrom, debye

import matplotlib.pyplot as plt

#importing the QM7 database
data = dict(loadmat('qm7.mat'))
Z = data['Z'] # numbers
R = data['R']/angstrom # positions in angstrom

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run optimization for 1 system')
    parser.add_argument('-index', default=0, type=int)
    parser.add_argument('-scheme', default='FB', type=str)
    parser.add_argument('-penalty', default=0., type=str)
    args = parser.parse_args()
    
    foldername = 'xyz_files/hpc_run1/' + args.scheme + '/'
    
    loc = Localizer(Z[args.index],R[args.index],args.index)
    loc.set_scheme(name=args.scheme, p=args.penalty)
    conv = loc.optimize_line_search(nsteps=1000, psi4_guess=False)
    
    quad_compared = loc.compare_quadrupole()
    quad_loc_qm7 = loc.total_loc_quadrupole.flatten()
    quad_qm7 = loc.total_quadrupole.flatten()
    
    loc.write_centers(folder=foldername)

