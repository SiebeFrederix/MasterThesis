import numpy as np
from scipy.io import loadmat
from localizer_hpc import Localizer

#https://github.com/molmod/molmod
from molmod.units import angstrom, debye

import matplotlib.pyplot as plt
import json

#importing the QM7 database
data = dict(loadmat('../molecule_geometries/qm7.mat'))
Z = data['Z'] # numbers
R = data['R']/angstrom # positions in angstrom

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run localization for qm7 database')
    parser.add_argument('-start', default=0, type=int)
    parser.add_argument('-stop', default=0, type=int)
    parser.add_argument('-cores', default=1, type=int)
    # memory will have to be in Gb
    parser.add_argument('-memory', default=2., type=float)
    parser.add_argument('-scheme', default='FB', type=str)
    parser.add_argument('-penalty', default=0., type=float)
    args = parser.parse_args()
    
    foldername = 'hpc_run1/'
    basisset = 'aug-cc-pvtz'
    lot = 'scf'
    
    output_list = [{'BasisSet' : basisset,
                    'LevelOfTheory' : lot,
                    'Scheme' : args.scheme,
                    'Penalty' : args.penalty,
                    'StartIndex' : args.start,
                    'StopIndex' : args.stop,
                    'Cores' : args.cores,
                    'Memory' : args.memory}]
    
    for i in range(args.start,args.stop):
        loc = Localizer(Z[i], R[i], i, basisset = basisset, lot = lot,
                       cores = args.cores, memory = args.memory*1e+09)
        loc.set_scheme(name=args.scheme, p=args.penalty)
        conv = loc.optimize_line_search(nsteps=1000, psi4_guess=False)
        
        output_dict = {'Convergence' : conv,
                       'QuadCompare' : loc.compare_quadrupole().tolist(),
                       'QuadTotal' : loc.total_quadrupole.tolist(),
                       'QuadLocal' : loc.total_loc_quadrupole.tolist(),
                       'ValueCostV4' : float(loc.V4_cost(loc.W)),
                       'nSteps' : int(np.where(loc.conv_hist != 0.)[0][-1])}
    
        loc.write_centers(folder= '../xyz_files/' + foldername)
        
        output_list.append(output_dict)
    
    json_filename = '../benchmarking/' + foldername + args.scheme + '/'
    json_filename += args.scheme + '_' + str(args.start) + '_' + str(args.stop) + '.json'
    with open(json_filename, 'w') as fout:
        json.dump(output_list, fout)
    fout.close()
        
        

