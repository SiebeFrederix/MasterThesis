import numpy as np

import psi4
#https://github.com/molmod/molmod
from molmod.periodic import periodic
from molmod.units import *
import shutil
import os

def make_comment(energy, rvecs, field_type, field_info, nuclear_repulsion = None, nai = None, kin = None, forces = None, charge = 0):
    comment = ''
    if not rvecs is None:
        comment += 'Lattice="'
        comment += '%f %f %f %f %f %f %f %f %f"' % tuple(rvecs.flatten())
    if not forces is None:
        comment += 'Properties=species:S:1:pos:R:3:Z:I:1:force:R:3 energy='
    else:
        comment += 'Properties=species:S:1:pos:R:3:Z:I:1 energy='
    comment += str(energy)
    
    if field_type == 'dipole':
        comment += ' field_type=dipole'
        comment += ' dipole="%f %f %f"' % tuple(field_info)
    
    if not nuclear_repulsion is None:
        comment += ' nuclear_repulsion=%f' % nuclear_repulsion
    comment += ' charge=%d' % charge
    
    if not nai is None:
        comment += ' nai="'
        for i in range(len(nai)):
            comment += '%f ' % nai[i]
        comment = comment[:-1]
        comment += '"'
    
    if not kin is None:   
        comment += ' kin="'
        for i in range(len(kin)):
            comment += '%f ' % kin[i]   
        comment = comment[:-1]
        comment += '"'
        
    return comment

class XYZLogger(object):
    def __init__(self, filename, start = 0, step = 1, append = False):
        self.filename = filename
        if not append:
            file = open(self.filename, 'w')
            file.close()

    def write(self, numbers, positions, centers, energy, field_type='dipole', field_info=[0, 0, 0], nuclear_repulsion=None, nai=None, kin=None, forces = None, charge = 0, rvec=None):
        all_numbers = np.concatenate((numbers, 99 * np.ones(centers.shape[0], dtype=np.int)), axis=0)
        all_positions = np.concatenate((positions, centers), axis=0)
    
        self.file = open(self.filename, 'a+')
        
        N = np.shape(all_positions)[0]

        self.file.write('%d\n' % N)
        self.file.write(make_comment(energy, rvec, field_type, field_info, nuclear_repulsion, nai, kin, forces = forces, charge = charge) + '\n')
        
        for atom in range(N):
            symbol = periodic[all_numbers[atom]].symbol
            if forces is None:
                newline = '%s\t%f\t%f\t%f\t%d' % (symbol, all_positions[atom, 0], all_positions[atom, 1], all_positions[atom, 2], all_numbers[atom])
            else:
                if all_numbers[atom] == 99: # The electroncenters do not have forces acting on them
                    fx, fy, fz = 0, 0, 0
                else:
                    fx, fy, fz = tuple(forces[atom])
                newline = '%s\t%f\t%f\t%f\t%d\t%f\t%f\t%f' % (symbol, all_positions[atom, 0], all_positions[atom, 1], all_positions[atom, 2], all_numbers[atom], fx, fy, fz)

            self.file.write(newline + '\n')

        self.file.close()
    
class MoleculeString(object):
    def __init__(self, elements, positions, charge = 0, multiplicity = 1, reorient = True, no_symmetry = False, no_com = False, verbose = True):
        self.geometry = "\n%d %d\n" % (charge, multiplicity)
        for index, atom in enumerate(elements):
            self.geometry += "%s %f %f %f\n" % (periodic[atom].symbol, positions[index, 0], positions[index, 1], positions[index, 2])
        
        if not reorient: # The molecule wont be rotated
            self.geometry += 'no_reorient\n'
        
        if no_symmetry:
            self.geometry += 'symmetry c1\n'
            
        if no_com:
            self.geometry += 'no_com\n'
            
        self.geometry = self.geometry[:-1]
        
        if verbose:
            print('Molecule string generated:')
            print(self.geometry)
            print('')
        
class Psi4Model(object):
    def __init__(self, method = 'pbe', basis = 'aug-cc-pvtz', psi4_output = 'output.dat', memory = 20e+09, cores = 12, scratch_dir = 'scratch'):
        self.basis = basis
        self.method = method
        
        if not os.path.exists(scratch_dir):
            os.mkdir(scratch_dir)
            
        else:
            shutil.rmtree(scratch_dir)
            os.mkdir(scratch_dir)
            
        self.scratch_dir = scratch_dir
        
        # optional
        psi4.core.IOManager.shared_object().set_default_path(scratch_dir)
        psi4.set_memory(int(memory))
        psi4.set_num_threads(cores)
        
        # Overwrites the output.dat
        psi4.set_output_file(psi4_output, False)
        
        print('Psi4 Model initiated with method %s and basis set %s' % (self.method, self.basis))
        
    def clean(self):
        psi4.core.clean()
        print('Psi4 core cleaned')
        
        shutil.rmtree(self.scratch_dir)
        
    def compute(self, elements, positions, rvec = None, charge = 0, multiplicity = 1, field_type = 'dipole', dipole_strength = [0, 0, 0], forces = False, optimize = False,
                opt_cartesian = False, uhf = False, max_iters = None):
        if not rvec is None:
            print('Warning: psi4 does not work with periodic boundary conditions. Ignoring them.')
        
        self.molecule_string = MoleculeString(elements, positions, charge = charge, multiplicity = multiplicity, reorient = False, no_symmetry = True, no_com = True, verbose = False)
        self.mol = psi4.geometry(self.molecule_string.geometry) # De input moet in angstrom staan
        
        if field_type == 'dipole':
            #print('Perturbing with a constant electric field')  
            
            options = {'basis': self.basis,
                       'PERTURB_H': True,
                       'PERTURB_WITH' : 'DIPOLE',
                       'PERTURB_DIPOLE' : list(dipole_strength)}
            
        if opt_cartesian:
            options['OPT_COORDINATES'] = 'cartesian'
            
        if uhf:
            options['reference'] = 'uhf'
        
        if not max_iters is None:
            options['GEOM_MAXITER'] = max_iters
            
        psi4.set_options(options)
            
        if optimize:
            self.energy, self.wavefunction = psi4.optimize(self.method, return_wfn = True)
            if forces:
                self.gradient = self.wavefunction.gradient()
                return self.energy, mol.geometry().np / angstrom, self.gradient.np
            else:
                return self.energy, self.mol.geometry().np / angstrom

        if forces:
            self.gradient, self.wavefunction = psi4.gradient(self.method, return_wfn = True)
            self.energy = self.wavefunction.energy()
            return self.energy, self.gradient.np
        else:
            self.energy, self.wavefunction = psi4.energy(self.method, return_wfn = True)
            return self.energy
        
    def psi4_localize(self, scheme='PIPEK_MEZEY'):
        # Returns the used basisset
        basis = self.wavefunction.basisset()

        # Returns the requested alpha orbital subset
        # with symmetry AO and only the OCCupied orbitals
        C_occ = self.wavefunction.Ca_subset("AO", "OCC") # canonical C_occ coefficients
        
        self.C_occ_orb = C_occ

        # Builds the localizer. Args: type, basis, orbitals
        Local = psi4.core.Localizer.build(scheme , basis, C_occ) # Pipek-Mezey Localization
        #Local = psi4.core.Localizer.build("BOYS", basis, C_occ) # Boys Localization

        # Performs the localization procedure
        Local.localize()
        
        return np.array(Local.U)
        
    def get_quadrupole(self):
        # Het moleculair quadropule moment, wordt ook weggeschreven in de .dat file
        oe = psi4.core.OEProp(self.wavefunction)
        oe.add('QUADRUPOLE')
        oe.compute()
        
        xx = self.wavefunction.variable(" QUADRUPOLE XX")
        xy = self.wavefunction.variable(" QUADRUPOLE XY")
        xz = self.wavefunction.variable(" QUADRUPOLE XZ")
        yy = self.wavefunction.variable(" QUADRUPOLE YY")
        yz = self.wavefunction.variable(" QUADRUPOLE YZ")
        zz = self.wavefunction.variable(" QUADRUPOLE ZZ")
        
        quadrupole_matrix = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
        traceless = quadrupole_matrix - np.trace(quadrupole_matrix) * np.eye(3) / 3.
        
        # Om het quadropool moment per gelokaliseerd orbitaal te berekenen (op dezelfde manier als de gewone dipool)
        quadrupole = self.mints.ao_quadrupole()
    
        multipole = np.zeros([self.C_occ_orb.shape[1], 3, 3])
        for i in range(6): # xx, xy, xz, yy, yz, zz
            quadrupole_component = - np.array(quadrupole[i]) #/ angstrom**2
            comp = np.einsum('ji,jk,ki->i', self.C_occ_orb, quadrupole_component, self.C_occ_orb, optimize = True)
            if i == 0: # Ik construeer hier terug de 3 bij 3 tensor
                multipole[:, 0, 0] = comp
            elif i == 1:
                multipole[:, 0, 1] = comp
                multipole[:, 1, 0] = comp
            elif i == 2:
                multipole[:, 0, 2] = comp
                multipole[:, 2, 0] = comp    
            elif i == 3:
                multipole[:, 1, 1] = comp
            elif i == 4:
                multipole[:, 1, 2] = comp
                multipole[:, 2, 1] = comp
            elif i == 5:
                multipole[:, 2, 2] = comp
            
        return 3 * traceless * debye * angstrom, multipole
