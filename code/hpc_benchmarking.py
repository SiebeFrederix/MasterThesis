import numpy as np
from scipy.io import loadmat
from localizer_hpc import Localizer

#https://github.com/molmod/molmod
from molmod.units import angstrom, debye

import matplotlib.pyplot as plt
import json

#importing the centered QM7 database
Z = np.load('../data/molecule_geometries/benchmarking_dataset_Z.npy')
R = np.load('../data/molecule_geometries/benchmarking_dataset_R.npy')
qm7_idxs = np.load('../data/molecule_geometries/benchmark_idxs.npy')

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
    parser.add_argument('-lot', default='scf', type=str)
    args = parser.parse_args()
    
    foldername = 'hpc_run2/'
#     foldername = 'test_runs/'
    basisset = 'aug-cc-pvtz'
#     basisset = '6-311ppg_d_p_'
#     lot = 'scf'
#     lot = 'b3lyp'
    import_data = True
    
    output_list = [{'BasisSet' : basisset,
                    'LevelOfTheory' : args.lot,
                    'Scheme' : args.scheme,
                    'Penalty' : args.penalty,
                    'StartIndex' : args.start,
                    'StopIndex' : args.stop,
                    'Cores' : args.cores,
                    'Memory' : args.memory}]
    
    if args.scheme == 'PM':
        import_data = False
    elif args.scheme == 'ER':
        import_data = False
    
    for i in range(args.start,args.stop):
        loc = Localizer(Z[i], R[i], qm7_idxs[i], basisset = basisset, lot = args.lot,
                       cores = args.cores, memory = args.memory*1e+09, check_import=import_data)
        loc.set_scheme(name=args.scheme, p=args.penalty)
        conv = loc.optimize_line_search(nsteps=1000, psi4_guess=False)
        
        if conv:
            output_dict = {'Convergence' : conv,
                           'QuadCompare' : loc.compare_quadrupole().tolist(),
                           'QuadTotal' : loc.total_quadrupole.tolist(),
                           'QuadLocal' : loc.total_loc_quadrupole.tolist(),
                           'ValueCostV4' : float(loc.V4_cost(loc.W))}
            loc.write_centers(folder= '../data/xyz_files/' + foldername)
            if args.scheme != 'PM':
                output_dict['nSteps'] = int(np.where(loc.conv_hist != 0.)[0][-1])
        else:
            output_dict = {'Convergence' : conv}
        
        output_list.append(output_dict)
        
        if not loc.is_imported:
            loc.model.clean()
    
    json_filename = '../data/benchmarking/' + foldername + args.scheme + '/'
    json_filename += args.scheme + '_' + str(args.start) + '_' + str(args.stop) + '.json'
    with open(json_filename, 'w') as fout:
        json.dump(output_list, fout)
    fout.close()
        
        

