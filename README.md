# MasterThesis
This is the code base that I will be using/have used throughout my master thesis.
You will need to have Numpy, SciPy, Psi4 and IOData installed. MatplotLib can be usefull,
but it is not necessary to run the core code.

## Build of the directories
We have two main folders:
Data: Contains all the data needed for calculation, folders for storing output data and some additional folders for storing figures.
Code: Contains the main code base for localization calculations. There are some additional Python Notebooks which where used for testing the code and doing some preliminary calculations.

## The data folder

### ab_initio_data:
json files containing the Alpha Coefficients, amount of orbitals, dipole matrix elements of the orbitals, quadrupole tensor matrix elements of the orbitals, quadrupole tensors of the orbitals and the total energy.__
Naming:__
foldername= basisset__
filename= (index in QM7)+(molecular formula)+(basisset)+(level of theory).json

### benchmarking:
Json files containing calculation parameters, quadrupole tensors calculated with the AEP and the MOs, the final value of the cost function and the amount of steps for convergence.
Naming:
foldername=(runname)/(cost function name)
filename=(cost function name)+(start index)+(stop index).json

### figures:
Contains plots en visualisations of the molecules

### molden-fchk_files:
Contains fchk-files which were altered in order to visualize the localized MOs.
Naming:
foldername=(cost function name)
filename=(index in QM7)+(molecular formula)+(cost function name).fchk

### molecule_geometries:
Contains npy/mat files with molecule geometries and nuclear charges
benchmarking_.npy: 500 randomly selected molecules of the QM7 dataset.
qm7.mat: QM7 dataset as downloaded from http://quantum-machine.org/datasets/.
qm7_R_centered.npy: QM7 geometries centered at the origin.
optimized_.npy: Select number of molecule from QM7, optimized using HF.
positions.npy+nuclei_numbers.npy: Set of molecules with "weird" AEPs using FB.

### Potential_data:
Contains the value of the ESP calculated at all satifactory points. When the grid is generated, we flatten it so we get a array of x-,y-,z-coordinates. This grid is fed into psi4 and it calculates the ESP at these points so we get an array of ESP values. These arrays are contained in these files.
Naming:
foldername=(basisset)
filename=(index in QM7)+(molecular formula)+(basisset)+(level of theory)+(minimal amount of points)+(r_max).dat

### psi4_output:
Folder where psi4 dumps its output.

### xyz_files:
XYZ-files containing the nuclear geometry and the average electron positions. The AEPs are given the element Es. When a parameter sweep is performed, every intermediary average electron configuration is also included in the XYZ-file. These files are included in the 'sweep' folder.
Naming:
foldername=(runname)/(cost function name)
filename=(index in QM7)+(molecular formula)+(cost function name)+(basisset)+(level of theory).xyz
If the file contians a sweep:
foldername=(runname)/(cost function name)
filename=(index in QM7)+(molecular formula)+(cost function name)+(min value of param)+(max value of param)+(amount of steps).xyz

## Code folder:
core.py: Main interface to the Psi4 program. Performs SCF calculations, calculations of the integrals...
localizer.py: Imports core.py. Contains the methods for calculating the LMOs, AEPs, reproduction of the traceless quadrupole...
electrostatics.py (PROTOTYPE): Main code base to look at the reproduction of the electrostatic potential.

### hpc_run_code:
Examples of python code which were used to benchmark the rQM7 dataset of the hpc.

### post_processing:
Code that was used to debug localizer.py, plot results of the hpc runs, do local runs...


