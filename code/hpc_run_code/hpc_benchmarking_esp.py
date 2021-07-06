import numpy as np
from scipy.io import loadmat
from localizer_hpc import Localizer

#https://github.com/molmod/molmod
from molmod.units import angstrom, debye

import matplotlib.pyplot as plt
import json
import gc

#importing the centered QM7 database
Z = np.load('../data/molecule_geometries/benchmarking_dataset_Z.npy')
R = np.load('../data/molecule_geometries/benchmarking_dataset_R.npy')
qm7_idxs = np.load('../data/molecule_geometries/benchmark_idxs.npy')
opt_weights = np.load('../data/benchmarking/hpc_run3/V5/opt_weights.npy')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run localization for qm7 database')
    parser.add_argument('-start', default=0, type=int)
    parser.add_argument('-stop', default=0, type=int)
    parser.add_argument('-cores', default=1, type=int)
    # memory will have to be in Gb
    parser.add_argument('-memory', default=2., type=float)
    parser.add_argument('-mingrid', default=100000, type=int)
    parser.add_argument('-lot', default='scf', type=str)
    args = parser.parse_args()
    
#     foldername = 'hpc_run3/'
    foldername = 'test_runs/'
#     basisset = 'aug-cc-pvtz'
    basisset = '6-311ppg_d_p_'
#     lot = 'scf'
#     lot = 'b3lyp'

    scheme_list = ['PM', 'FB', 'V5']
    
    output_list = [{'BasisSet' : basisset,
                    'LevelOfTheory' : args.lot,
                    'Scheme' : 'PM, FB, V5',
                    'StartIndex' : args.start,
                    'StopIndex' : args.stop,
                    'Cores' : args.cores,
                    'Memory' : args.memory}]
    
    for i in range(args.start,args.stop):
        loc = Localizer(Z[i], R[i], qm7_idxs[i], basisset = basisset, lot = args.lot,
                       cores = args.cores, memory = args.memory*1e+09, check_import=False)
        
        output_dict = []
        for scheme in scheme_list:
            loc.set_scheme(name=scheme, p=opt_weights[i])
            conv = loc.optimize_line_search(nsteps=1000, psi4_guess=False)

            if conv:
                esp_rmsd = loc.compute_esp_rmsd(r_max=10.0, min_points=args.mingrid)
                output_dict.append({'Scheme' : scheme,
                                    'Convergence' : conv,
                                    'ElectrostaticRMSD' : esp_rmsd,
                                    'ValueCostV4' : float(loc.V4_cost(loc.W))})
            else:
                output_dict.append({'Convergence' : conv})
        
        output_list.append(output_dict)
        
        if not loc.is_imported:
            loc.model.clean()
            del loc
            gc.collect()
        
    json_filename = '../data/benchmarking/' + foldername + 'esp/'
    json_filename += 'esp' + '_' + str(args.start) + '_' + str(i) + '.json'
    with open(json_filename, 'w') as fout:
        json.dump(output_list, fout)
    fout.close()
    
        
        

