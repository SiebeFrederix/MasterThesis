import jax.numpy as jnp
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
# jax.config.update('jax_log_compiles',True)
import scipy.linalg
from scipy.stats import ortho_group
from core import Psi4Model, XYZLogger, MoleculeString #, load_xyz
# from iodata import load_one, dump_one
import electrostatics as esp
import psi4
import json

#https://github.com/molmod/molmod
from molmod.units import angstrom, debye
from molmod.io.xyz import XYZReader

import os, sys

'''Localizer class'''

class Localizer:
    
    '''Initializer'''
    def __init__(self, numbers, positions, index=-1, basisset='6-311ppg_d_p_',
                 lot='scf', cores=2, memory=3e+09, check_import=False):
        # index of the molecule in the qm7 data set needed
        # filenames of the xyz files.
        self.index = index
        
        # FB-cost will always be the standard cost.
        self.scheme = ['FB', 1.0]
        self.set_scheme('FB')
        
        # initialization of the nulcei charges and positions
        self.Z = np.array(numbers[np.where(numbers != 0)], dtype=int)
        self.R = np.array(positions)[:len(self.Z)]
        
        # Choosing the basisset, for testing use a smaller set. The ERI's
        # scale very badly as a function of the basisset size.
        self.basisset = basisset
        self.lot = lot
#         basisset = 'aug-cc-pvtz'
#         basisset = '6-311ppg_d_p_'
        
        # Check if we want to import the data, when using the ER
        # scheme we will have to do the calculation. This is because
        # saving the ERI's uses way to much memory.
        if check_import:
            self.import_psi4_data(cores, memory)
        else:
            self.generate_psi4_data(cores, memory)
    
    '''
    Generating all the nessecarry data needed for all localization calculations.
    '''
    def generate_psi4_data(self, cores=2, memory=3e+09):
        print('generating data for: ' + self.generate_molname(scheme_name=False))
        
        # calculating energies and initialising the psi4 integral calculator
        self.model = Psi4Model(method = self.lot, basis = self.basisset,
                  psi4_output = '../data/psi4_output/psi4_output.dat', cores=cores, memory=memory)
        self.energy = self.model.compute(self.Z, self.R, field_type = 'dipole',
                                         dipole_strength = [0, 0, 0], charge = 0)
        self.mints = psi4.core.MintsHelper(self.model.wavefunction)
        
        # Retrieve occupied canonical coefficients
        self.C_occ = np.array(self.model.wavefunction.Ca_subset('AO', 'OCC'))
        
        # Calculate the amount of occupied orbitals
        self.N_occ = len(self.C_occ[0])
        
        # calculation of the center_matrix from the dipole integrals,
        # i.e. bra(psi_i)*vec(r)*ket(psi_j)
        dimc = len(self.C_occ)
        self.dipole_int = np.zeros((3,dimc,dimc))
        self.quadrupole_int = np.zeros((6,dimc,dimc))
        for i in range(3):
            self.dipole_int[i] = np.array(self.mints.ao_dipole()[i])
        self.center_matrix = np.einsum('ji,ajk,kl->ila', self.C_occ, -self.dipole_int, 
                                       self.C_occ, optimize=True)
        
        # calculating the components of the quadrupole matrix 
        # (defined in the same way as the dipole matrix). 
        # The indices are: xx,xy,xz,yy,yz,zz
        for i in range(6):
            self.quadrupole_int[i] = np.array(self.mints.ao_quadrupole()[i])
        quadrupole_components = np.einsum('ji,ajk,kl->ail', self.C_occ, self.quadrupole_int, 
                                       self.C_occ, optimize=True)
        
        # Setting up the quadrupole matrix, we agian have to do this in a 
        # cumbersome way because psi4 outputs only the components of the quadrupole
        # matrix and nog the matrix itself. Also, when writing out the localized
        # orbitals, the calcuation has to be performed to get the fchk file.
        self.quadrupole_matrix_elem = np.zeros((3,3,self.N_occ,self.N_occ))
        self.quadrupole_matrix_elem[0,0] = quadrupole_components[0]
        self.quadrupole_matrix_elem[0,1] = quadrupole_components[1]
        self.quadrupole_matrix_elem[1,0] = quadrupole_components[1]
        self.quadrupole_matrix_elem[0,2] = quadrupole_components[2]
        self.quadrupole_matrix_elem[2,0] = quadrupole_components[2]
        self.quadrupole_matrix_elem[1,1] = quadrupole_components[3]
        self.quadrupole_matrix_elem[1,2] = quadrupole_components[4]
        self.quadrupole_matrix_elem[2,1] = quadrupole_components[4]
        self.quadrupole_matrix_elem[2,2] = quadrupole_components[5]
        
        # Calculating the quadrupole matrix for each orbital.
        self.quadrupole_matrix = np.einsum('abii->iab', self.quadrupole_matrix_elem, 
                                           optimize = True)
        
        # No harm in writing out the data
        self.write_psi4_data()
        
        # Can be used to check if the model has to be cleaned afterwards
        self.is_imported = False
    
    '''
    If the psi4 data already exist, it gets read in with this function
    in order to do no unnessecarry calculations. If it doesn't exist,
    the data will be generated using another function.
    '''
    def import_psi4_data(self, cores=2, memory=3e+09):
        # Set path to read data
        foldername = '../data/ab_initio_data/' + self.basisset + '/'
        filename = self.generate_molname(scheme_name=False) + '_' + self.basisset + '_' + self.lot + '.json'
        
        # Check if the json file already exists, in that case import it.
        try:
            with open(foldername + filename, 'r') as read_file:
                input_dict = json.load(read_file)
            print('Imported data for: ' + self.generate_molname(scheme_name=False))

            self.C_occ = np.array(input_dict['AlphaCoeffs'])
            self.N_occ = input_dict['NOrbitals']
            self.center_matrix = np.array(input_dict['CenterMatrix'])
            self.quadrupole_matrix_elem = np.array(input_dict['QuadrupoleMatrixElems'])
            self.quadrupole_matrix = np.array(input_dict['QuadrupoleMatrix'])
            self.energy = input_dict['TotalEnergy']
            self.is_imported = True
        # if the json file does not exist, we perform the calculation and store
        # its output in a new file.
        except IOError:
            self.generate_psi4_data(cores, memory)
#             self.model.clean()
    
    '''
    Function which writes out the psi4 data such that this doesn't
    have to be calculated twice
    '''
    def write_psi4_data(self):
        print('writing out psi4 data for molecule: ' + self.generate_molname(scheme_name=False))
        
        # Put all relavent data in a dictionary (converting np.array's
        # to lists). We only save the necessary variables, these json
        # can get big.
        output_dict = {'AlphaCoeffs' : self.C_occ.tolist(),
                       'NOrbitals' : int(self.N_occ),
#                        'DipoleIntegrals' : self.dipole_int.tolist(),
#                        'QuadrupoleIntegrals' : self.quadrupole_int.tolist(),
                       'CenterMatrix' : self.center_matrix.tolist(),
                       'QuadrupoleMatrixElems' : self.quadrupole_matrix_elem.tolist(),
                       'QuadrupoleMatrix' : self.quadrupole_matrix.tolist(),
                       'TotalEnergy' : float(self.energy)}
        
        # Set path to save the data
        foldername = '../data/ab_initio_data/' + self.basisset + '/'
        filename = self.generate_molname(scheme_name=False) + '_' + self.basisset + '_' + self.lot + '.json'
        
        # Write out data to a json-file
        with open(foldername + filename, 'w') as fout:
            json.dump(output_dict, fout)
        fout.close()
    
    
    '''Generating molecule name from the components of the molecule'''
    def generate_molname(self, scheme_name=True):
        # stupid way of creating molecule names, but it works
        # First write index of array, then molecular contents
        # also use '0003' for 3 element in array
        filename = ''
        if self.index != -1:
            # adding zero's in front if index is given
            len_idx = len(str(self.index))
            filename = (4-len_idx)*'0' + str(self.index) + '_'
        else:
            filename = ''
        elem_numbers = np.array([16,8,7,6,5,1])
        elem_names = ['S','O','N','C','B','H']
        for i,el in enumerate(elem_numbers):
            n_el = len(np.where(self.Z == el)[0])
            if n_el == 1:
                filename += elem_names[i]
            elif n_el != 0:
                filename += elem_names[i] + str(n_el)
        if scheme_name:
            filename += '_' + self.scheme[0]
            if self.scheme[0] in ['FB_p', 'V2', 'V5']:
                filename += '_p_' + str(self.scheme[1])[:5]
        return filename
        
    
    '''
    Writer:
    Initializes the writer, is not in __init__ because it is not always needed.
    Includes a stupid way of generating molecule names
    '''
    def write_centers(self, append=False, folder='../data/xyz_files/local_runs/', filename='None'):
        # Assign filename depending if it is given as an input
        # or if it has to be generated from the elements.
        if filename == 'None':
            self.filename = (folder + self.scheme[0] + '/' + self.generate_molname() + 
                             '_' + self.basisset + '_' + self.lot + '.xyz')
        else:
            self.filename = folder + filename
        
        # Wegschrijven van de data, kan ook via een .npz file zoals in Toon zijn script
        # Meer informatie wegschrijven is ook mogelijk:
        # writer.write(numbers, positions, centers, energy, 'dipole', [0, 0, 0], 
        # nuclear_repulsion, nai, kin, forces = -gpos, charge = 0)
        self.writer = XYZLogger(self.filename, append = append)

        self.L_centers = np.einsum('ji,jka,ki->ia', self.W,
                                   self.center_matrix, self.W, optimize=True)/angstrom
        self.writer.write(self.Z, self.R, self.L_centers, self.energy)
        
        print('Geometry data written to: ' + self.filename)
    
#     '''Method for writing out the localized orbital info to a fchk file'''
#     def write_orbitals(self, filename='None'):
#         # Assign filename depending if it is given as an input
#         # or if it has to be generated from the elements.
#         if filename == 'None':
#             filename = '../data/molden-fchk_files/' + self.scheme[0] + '/' + self.generate_molname()
#         else:
#             filename = '../data/molden-fchk_files/' + filename
        
#         # Write out the molden file using psi4
#         psi4.molden(self.model.wavefunction, filename + '.molden')
        
#         # Using IOData, we load in the molden file. We alter the 
#         # values of the coefficients using the localized coefficients.
#         # Finally we write them out to a FCHK file and remove the molden file
#         molecule = load_one(filename + '.molden')
#         molecule_mo = molecule.mo
#         mo_coeffs = molecule_mo.coeffs
#         # replace the occupied orbital coefficients with the localized ones.
#         self.L_occ = np.dot(self.C_occ,self.W)
#         mo_coeffs[:,:self.N_occ] = self.L_occ
#         molecule_mo.coeffs = mo_coeffs
#         dump_one(molecule, filename + '.fchk')
#         os.remove(filename + '.molden')
    
    '''Simple setter to choose cost function'''
    def set_scheme(self, name, p=1.0):
        if name == 'FB':
            self.cost_function = jax.jit(self.FB_cost)
            self.cost_grad = jax.jit(jax.grad(self.cost_function))
            self.q = 4
        elif name == 'FB_p':
            self.cost_function = jax.jit(self.FB_p_cost)
            self.cost_grad = jax.jit(jax.grad(self.cost_function))
            self.q = p*4
            
        # Calculating the eri are expensive, so we only calculate them
        # if they are needed for the ER scheme.
        elif name == 'ER':
            # Retrieving the Electronic Repulsion Integrals (ERI).
            self.eri_int = np.array(self.mints.ao_eri())

            # Calculating the repulsion integrals where the indices are 
            # orbitals.
            self.eri = np.einsum('ia,jb,ijkl,kc,ld->abcd', self.C_occ, self.C_occ,
                                 self.eri_int, self.C_occ, self.C_occ, optimize=True)
            
            self.cost_function = jax.jit(self.ER_cost)
            self.cost_grad = jax.jit(jax.grad(self.cost_function))
            self.q = 4
        elif name == 'V1':
            self.cost_function = jax.jit(self.V1_cost)
            self.cost_grad = jax.jit(jax.grad(self.cost_function))
            self.q = 4
        elif name == 'V2':
            self.cost_function = jax.jit(self.V2_cost)
            self.cost_grad = jax.jit(jax.grad(self.cost_function))
            self.q = 4
        elif name == 'V3':
            self.cost_function = jax.jit(self.V3_cost)
            self.cost_grad = jax.jit(jax.grad(self.cost_function))
            self.q = 8
        elif name == 'V4':
            self.cost_function = jax.jit(self.V4_cost)
            self.cost_grad = jax.jit(jax.grad(self.cost_function))
            self.q = 8
        elif name == 'V5':
            self.cost_function = jax.jit(self.V5_cost)
            self.cost_grad = jax.jit(jax.grad(self.cost_function))
            self.q = 8
        elif name != 'PM':
            print('Error: chosen cost function is not yet implemented.')
            return
        
        # [0] is the costfunction name, [1] is a parameter
        # that it can use
        self.scheme[0] = name
        self.scheme[1] = p
    
    ''' 
    Foster Boys cost function:
    Takes a unitary matrix as input and outputs the value
    of the FB cost function.
    '''
    def FB_cost(self, W):
        # calculating the dipole moment using the localized orbitals
        # index i is the occupied orbital index, index a is for 
        # the spatial dimension.
        p_loc = jnp.einsum('ji,jka,ki->ia', W, self.center_matrix, 
                          W , optimize = True)
        return jnp.einsum('ia,ia', p_loc, p_loc, optimize=True)
    
    ''' 
    Foster Boys cost function with additional penalty:
    Takes a unitary matrix as input and outputs the value
    of the FB cost function. p is a penalty that restricts
    the size of the largest orbital.
    '''
    def FB_p_cost(self, W):
        # The p-value can be set in the set_scheme function.
        # Note that this value has to be an INTEGER. A new
        # initialization for W is needed for large values of p.
        p = self.scheme[1]
        
        # calculating the dipole moment using the localized orbitals
        # index i is the occupied orbital index, index a is for 
        # the spatial dimension.
        p_loc = jnp.einsum('ji,jka,ki->ia', W, self.center_matrix, 
                          W , optimize = True)
        
        # calculating the diagonal components of the quadrupole moment 
        # using the localized orbitals
        q_loc = jnp.einsum('ji,aajk,ki->ia', W, -self.quadrupole_matrix_elem,
                         W, optimize = True)
        
        # intermediate result to enable the implementation of the penalty p.
        intermediate = jnp.power((jnp.einsum('ia->i', q_loc, optimize=True) -
                                jnp.einsum('ia,ia->i', p_loc, p_loc, optimize=True)),p)

        # When p=1, this is the normal FB cost function and the quadrupole has no
        # effect on the calculation.
        return -jnp.sum(intermediate)
    
    '''Electronic Repulsion Costfunction'''
    def ER_cost(self, W):
        # Calculation of the orbital self repulsion
        return jnp.einsum('ji,ki,jklm,li,mi->', W, W, self.eri, W, W, optimize=True)
    
    '''
    My own implementation of the cost function with non-spherical penalty.
    The quadrupole is not shifted to the origin, giving rise to some wonky
    results.
    Version 1.3
    '''
    def V1_cost(self, W):
        # Weight factor, determining the weigth of the penalty
        weight = self.scheme[1]
        
        # Calculating the quadrupole matrix using the occupied orbitals.
        q_loc = jnp.einsum('ji,abjk,ki->iab', W, self.quadrupole_matrix_elem,
                                W, optimize=True)
        
        # Calculating the trace of the quadrupole matrices.
        q_trace = jnp.einsum('iaa->i', q_loc, optimize=True)/3.
        
        # Making the quadrupole traceless, we do this by constructing an
        # array of identity matrices and filling them with their corresponding
        # traces.
        q_loc_traceless = q_loc - jnp.kron(q_trace,jnp.eye((3))).transpose().reshape(self.N_occ,3,3)  
        
        # Calculating the matrix norm of the quadrupole matrix of 
        # every occupied orbital.
        penalty = jnp.sum(q_loc_traceless**2)
        
        return self.FB_cost(W) - weight*penalty
    
    '''
    My own implementation of the cost function with non-spherical penalty.
    The quadrupole is translated to the origin, this should (hopefully) fix
    the problems we had with Version 1.3.
    Version 2.0
    '''
    def V2_cost(self, W):
        # Weight factor, determining the weigth of the penalty
        weight = self.scheme[1]
        
        # calculating the dipole moment using the localized orbitals
        # index i is the occupied orbital index, index a is for 
        # the spatial dimension.
        p_loc = jnp.einsum('ji,jka,ki->ia', W, self.center_matrix,
                           W, optimize = True)
        
        # Calculating the quadrupole matrix using the occupied orbitals.
        q_loc = jnp.einsum('ji,abjk,ki->iab', W, self.quadrupole_matrix_elem,
                           W, optimize = True)
        
        # Constructing the matrix which shifts the quadrupole tensor to the origin.
        p_matrix = jnp.einsum('ia,ib->iab', p_loc, p_loc, optimize=True)
        
        # Shifting the quadrupole tensor to the origin.
        q_origin = q_loc + p_matrix
#         self.q_origin = q_origin
        
        # Calculating the trace of the quadrupole matrices.
        q_trace = (jnp.einsum('iaa->i', q_origin, optimize=True)/3.).reshape(self.N_occ,1,1)
        
        # Making the quadrupole traceless, we do this by constructing an
        # array of identity matrices and filling them with their corresponding
        # traces.
        q_traceless = q_origin - q_trace*jnp.eye((3)).reshape(1,3,3)
        
        # Constructing the penalty term.
        penalty = jnp.sum(q_traceless**2)
        
        return self.FB_cost(W) - weight*penalty
#         return -weight*penalty
    
    '''
    My own implementation of the cost function. This has high resemblence
    with the Foster-Boys cost function. The main goal is that the quadrupole
    of the localized centers is as close as posible to the total QM quadrupole.    
    Version 3.0
    '''
    def V3_cost(self, W):
        # calculating the dipole moment using the localized orbitals
        # index i is the occupied orbital index, index a is for 
        # the spatial dimension.
        p_loc = jnp.einsum('ji,jka,ki->ia', W, self.center_matrix,
                           W, optimize = True)
        
        # The quadrupole tensor calculated classically using the center
        # postitions of the localized orbitals
        q_center = jnp.einsum('ia,ib->iab', p_loc, p_loc, optimize = True)
        
        
        q_diff = (jnp.einsum('iab->ab', -self.quadrupole_matrix, optimize = True) -
                  jnp.einsum('iab->ab', q_center, optimize = True))
        
        return -jnp.sum(q_diff**2)
    
    '''
    My own implementation of the cost function. This has high resemblence
    with the Foster-Boys cost function. The main goal is that the traceless quadrupole
    of the localized centers is as close as posible to the total QM traceless quadrupole.    
    Version 4.0
    '''
    def V4_cost(self, W):
        # calculating the dipole moment using the localized orbitals
        # index i is the occupied orbital index, index a is for 
        # the spatial dimension.
        p_loc = jnp.einsum('ji,jka,ki->ia', W, self.center_matrix,
                           W, optimize = True)
        
        # The quadrupole tensor calculated classically using the center
        # postitions of the localized orbitals
        q_center = jnp.einsum('ia,ib->iab', p_loc, p_loc, optimize = True)
        
        # calculating the difference between the two quadrupole tensors.
        q_diff = (jnp.einsum('iab->ab', -self.quadrupole_matrix, optimize = True) -
                  jnp.einsum('iab->ab', q_center, optimize = True))
        
        # Making the quadrupole difference traceless.
        q_diff_traceless = q_diff - jnp.trace(q_diff)*jnp.eye((3))/3.
        
        return -jnp.sum(q_diff_traceless**2)
    
    '''
    My own implementation of the cost function. The main goal is that the 
    traceless quadrupole of the localized centers is as close as possible 
    to the total QM traceless quadrupole. This difference is used as a 
    penalty to the FB cost function.
    Version 4.0
    '''
    def V5_cost(self,W):
        return self.FB_cost(W) + self.scheme[1]*self.V4_cost(W)
    
    ''' 
    Function that calculates the conjugate gradient parameter of
    Polak-Ribiere-Polyak. H_k = G_k + par*H_(k-1)
    '''
    def CGPR(self, r_grad_old, r_grad_new):
        gamma = np.einsum('ij,ij',r_grad_new,
                          (r_grad_new-r_grad_old),optimize=True)
        return gamma/np.einsum('ij,ij',r_grad_old,r_grad_old,optimize=True)
    
    '''
    This function finds the smallest positive root of a "deg" degree
    polynomial fitted to the data x and y.
    '''
    def find_smallest_root(self, x, y, deg, save_hist=False):
        # calculates the polynomial coefficients, highest order first
        coeffs = np.polyfit(x, y, deg)
        
        # Save the polynomial expansion coefficients
        if save_hist:
            self.iteration_hist['PolyCoeffs'] = coeffs

        # Calculate the roots of the polynomial
        roots = np.roots(coeffs)

        #finding the smallest positive root
        sp_root = 10
        for root in roots:
            if np.isreal(root):
                if (root > 0) and (root<sp_root):
                    sp_root = root

        # If no valid root is found, the stepsize will be zero
        if sp_root == 10:
            print('No root found')
            sp_root = 0.

        if sp_root > x[-1]:
            print('root out of range')
            # we take the x for which y is smallest in this case
            sp_root = x[np.argmin(np.abs(y))]

        return sp_root
    
    '''
    Determining the stepsize:
    Here we approximate the geodisec with tangent vector H using
    a line search algorithm. The geodesic is parameterized in mu,
    i.e. the stepsize. We approximate its derivative with respect
    to mu in a region (0;T) using a polynomial. The first positive 
    zero crossing of the polynomial will approximate the position
    mu where the cost function has its first maximum allong the 
    geodesic.
    '''
    def det_step_size(self, W, H, save_hist = False):
        # when looking at the expression of the FB cost function
        # we can see that q=4, since J_FB ~ W^4
        # We multiply by i to make H hermitian.
        eigval = scipy.linalg.eigvalsh(1j*H)
        omega = np.max(np.abs(eigval))
        T = (2*np.pi)/(self.q*omega)
        
        # We will approximate the geodesic with a polynomial of 
        # power 4.
        P = 4
        
        # initializing the exponentials of H in such a way that only
        # one matrix exponential has to be evaluated.
        R = np.zeros((P+1,len(W),len(W)))
        R[0] = np.eye(len(W))
        R[1] = scipy.linalg.expm((T/P)*H)
        R[2] = np.dot(R[1],R[1])
        R[3] = np.dot(R[2],R[1])
        R[4] = np.dot(R[3],R[1])
        
        # we now evaluate the first order derivate of the cost function
        # with respect to the stepsize in every point mu_i. 
        # An analytic expression is found in the paper about optimization
        # with unitary constraints.
        mu = np.linspace(0,T,P+1)
        J_mu = np.zeros((P+1))
        for i in range(P+1):
            inter_grad = np.array(self.cost_grad(np.dot(R[i],W)))
            J_mu[i] = -2*np.einsum('ij,kj,lk,il',inter_grad, W, R[i],H)
        
        # determine smallest positive zerocrossing of the first derivative
        # of the cost function wrt the stepsize.
        step = self.find_smallest_root(mu, J_mu, P, save_hist=save_hist)
        
        # Save the interval size of the line search and the obtained MIN.
        if save_hist:
            self.iteration_hist['Root'] = step
            self.iteration_hist['MuMax'] = T
        
        return np.real(step)

    '''
    Gradient ascent of the cost function, using line search algorithm:
    Implemantation of gradient ascent with unitary constraints. The implementation
    makes use of a momentum-like term (CGPR parameter). Implementation makes use of
    JAX's automatic differentiation to calculate the gradient of the cost function.
    A line search algorithm along the geodesic is used to find the optimal stepsize.
    '''
    def optimize_line_search(self, nsteps=600, psi4_guess=False, save_iteration=-1,
                             n_restarts=0):
        # The alg can get stuck in a reset loop, if this happens too many
        # times we will abort the calculation. This is probably due to 
        # the cost function being badly conditioned.
        if n_restarts == 15:
            print('Restarted 15 times, algorithm will not converge: ABORT')
            return False
        
        print('Optimizing cost function: ' + self.scheme[0])
        # If the Pipek-Mezey cost function has to be optimized,
        # we run it under the hood via Psi4.
        if self.scheme[0] == 'PM':
            self.Localize_pipek_mezey()
            # returns True to confirm convergence
            return True
        
        # initialization of the history arrays
        conv = np.zeros((nsteps))
        cost = np.zeros((nsteps))
        step_hist = np.zeros((nsteps))
        
        # initialize number of consecutive resets
        consec_resets = 0
        
        # This allows us to check how what the optimizer was doing
        # when estimating the stepsize.
        if save_iteration != -1:
            self.iteration_hist = {'Iteration' : save_iteration}
        
        # The psi4 localizer is very fast (it is written in C++),
        # so initializing our first guess with the outcome of this routine
        # may speed up our localization drastically.
        if psi4_guess:
            W = self.model.psi4_localize(scheme="BOYS")
        else:
            W = ortho_group.rvs(self.N_occ)
            # make sure we initialize a orthogonal matris with det=1
            while(scipy.linalg.det(W) < 0):
                W = ortho_group.rvs(self.N_occ)

        # initiate optimization loop
        for i in range(nsteps):
            # Calculating the euclidian derivative
#             grad = np.array(jax.jit(jax.grad(cost_function, argnums=0))(W))
            grad = self.cost_grad(W)
            # grad = calc_fosterboys_grad(C_occ,W)

            # The conjugate gradient of PR needs the k-1'th gradient
            if i!=0:
                riemann_grad_old = riemann_grad.copy()
            
            # Calculating the Riemann derivative
            riemann_grad = (np.einsum('ij,kj->ik',grad,W,optimize=True) - 
                            np.einsum('ij,kj->ik',W,grad,optimize=True))
            if i!=0:
                H_old = H.copy()
                gamma = self.CGPR(riemann_grad_old,riemann_grad)
                
                H = riemann_grad + gamma*H_old
                
                # We look at the angle between the gradient and H, if
                # this angle is close to 90 degrees we will reset H. 
                # This can sometimes induce some problems when the cost
                # function isn't well conditioned, leading to H being reset
                # every single loop with the optimization not converging.
                # If H is reset 5 consecutive times, we will restart the
                # full optimization.
                numerator = np.einsum('ij,ij',H,riemann_grad,optimize=True)
                denominator = np.sqrt(np.einsum('ij,ij',H,H,optimize=True)*
                                      np.einsum('ij,ij',riemann_grad,riemann_grad,optimize=True))
                if consec_resets == 5:
                    print('Conjugate gradient resetted 5 consecutive times: REINITIALIZING')
                    return self.optimize_line_search(nsteps, save_iteration=save_iteration,
                                                     n_restarts= n_restarts + 1)
                elif(numerator < denominator*1e-3):
                    print('reset at iteration:' + str(i))
                    H = riemann_grad
                    consec_resets += 1
                elif consec_resets > 0:
                    consec_resets = 0
                    
                    
#                 if(i%((self.N_occ*(self.N_occ-1))//2) == 0):
#                     print('reset')
#                     H = riemann_grad
                    
            else:
                gamma = 0
                H = riemann_grad
            
            # We divide by self.N_occ**2 so that the convergence criterion scales.
            conv[i] = np.einsum('ij,ij',riemann_grad,riemann_grad,optimize=True)/(2*self.N_occ**2)
            cost[i] = float(self.cost_function(W))
            
            # check convergence criterion
            if(conv[i] < 10**(-5)):
                print('Convergence: norm of riemann gradient below 10^-5')
                self.conv_hist = conv
                self.cost_hist = cost
                self.step_hist = step_hist
                self.W = W
                self.L_occ = np.dot(self.C_occ, W)
                # returns True to confirm convergence
                return True
            
            if i == save_iteration:
                self.iteration_hist['H'] = H
                self.iteration_hist['W'] = W
                mu = self.det_step_size(W, H, save_hist=True)
            else: 
                mu = self.det_step_size(W, H, save_hist=False)
            step_hist[i] = mu
            
            # This criterion is checked so that the loop can be
            # exited early if the stepsize becomes too small. 
            # If this happens in the first iteration, we start a new
            # run. This means that the first guess was bad.
            if((mu < 1e-10) and (i == 0)):
                print('Stepsize < 10^-10 in first iteration: REINITIALIZING')
                return self.optimize_line_search(nsteps, save_iteration=save_iteration,
                                                 n_restarts=n_restarts)
            elif((mu < 1e-10)):
                print('No convergence: stepsize < 10^-10')
                self.conv_hist = conv
                self.cost_hist = cost
                self.step_hist = step_hist
                # We do not save matrices that don't contain
                # interesting info.
#                 self.W = W
#                 self.L_occ = np.dot(self.C_occ, W)
                return False
            
            # update unitary matrix
            P = scipy.linalg.expm(mu*H)
            W = np.dot(P,W)
        
        self.conv_hist = conv
        self.cost_hist = cost
        self.step_hist = step_hist
        # We do not save matrices that don't contain
        # interesting info.
#         self.W = W
#         self.L_occ = np.dot(self.C_occ, W)
        print('No convergence')
        return False
    
    '''Using the Pipek-Mezey optimizer implemented in psi4'''
    def Localize_pipek_mezey(self):
        self.scheme[0] = 'PM'
        self.W = self.model.psi4_localize()
    
    '''
    Calculating the classical quadrupole components using the
    positions and charges.
    '''
    def calc_classic_quadrupole(self, positions, charges, traceless=False):
        # The classical way of calculating the quadrupole moment.
        quadrupole = np.einsum('i,ia,ib->ab', charges, positions,
                               positions, optimize = True)
        
        # Making the quadrupoles traceless
        if traceless:
            quadrupole -= np.trace(quadrupole)*np.eye((3))/3.
        return quadrupole
    
    '''
    A function that calculates the total quadrupole moment of the molecule
    in two ways. The first uses the quadrupole integrals supplied by psi4.
    The second method calculates them classically from the localized centers.
    '''
    def compare_quadrupole(self, traceless=True):
        # Calculating the classical traceless quadrupole components due to
        # the nuclei
        self.total_nuclei_quadrupole = self.calc_classic_quadrupole(self.R, self.Z, traceless=traceless)
        
        # calculating the components of the total quadrupole using the 
        # localized orbitals. The indices are: xx,xy,xz,yy,yz,zz
        # We divide by angstrom**2 because the quadrupoles are in 
        # atomic units.
        self.total_quadrupole = 2*np.einsum('ji,abjk,ki->ab', self.W, self.quadrupole_matrix_elem, 
                                       self.W, optimize=True)/(angstrom**2)
        
        # Making the quadrupoles traceless
        if traceless:
            self.total_quadrupole -= np.trace(self.total_quadrupole)*np.eye((3))/3.
            
        # Adding the quadrupole of the nuclei
        self.total_quadrupole += self.total_nuclei_quadrupole
        
        # now we calculate the quadrupole components classically using the
        # center positions of the localized orbitals. We divide by angstrom
        # because the quadrupoles are in atomic units.
        self.L_centers = np.einsum('ji,jka,ki->ia', self.W,
                                   self.center_matrix, self.W, optimize=True)/angstrom
        
        # Calculating the classical traceless quadrupole components due to 
        # the localized centers
        self.total_loc_quadrupole = self.calc_classic_quadrupole(self.L_centers,
                                        charges=np.full((self.N_occ),-2.), traceless=traceless)
            
        # Adding the quadrupole of the nuclei
        self.total_loc_quadrupole += self.total_nuclei_quadrupole
        
        # Return the difference in quadrupole moment in atomic units.
        return (self.total_quadrupole - self.total_loc_quadrupole)
    
    
    '''
    A method of calculating the relative-root-mean-square-deviation between the
    electrostatic potential computed quantum mechanically and classically using
    the localized centers. Here, the electrostatic.py file is used to generate
    the grid and calculate the classical ESPs.
    '''
    def compute_esp_rmsd(self, r_max=7., min_points=50000):
        print('generating grid')
        grid = esp.generate_coords(self.R, self.Z, r_max=r_max, min_points=min_points)
        
        folder = '../data/potential_data/' + self.basisset + '/'
        filename = (self.generate_molname(scheme_name=False) + '_' + self.basisset + 
                    '_' + self.lot + '_mp_' + str(min_points) + '_rm_' + 
                    str(r_max)[:5].replace('.','-') + '.dat')
        
        try:
            qm_esp = np.loadtxt(folder+filename)
            print('Potential data loaded')
        except OSError:
            print('Calculating QM ESP at the gridpoints')
            psi4.oeprop(self.model.wavefunction, 'GRID_ESP')
            print('Saving QM ESP')
            qm_esp = np.loadtxt('grid_esp.dat')
            np.savetxt(folder+filename, qm_esp)
        
        print('Computing classical ESP')
        self.L_centers = np.einsum('ji,jka,ki->ia', self.W,
                                   self.center_matrix, self.W, optimize=True)/angstrom
        
        nuc_esp = esp.calculate_ESP(grid, self.R, self.Z)
        elec_esp = esp.calculate_ESP(grid, self.L_centers, np.full((self.N_occ),-2.))
        clas_esp = (nuc_esp + elec_esp)/angstrom
#         print(np.min(np.abs(clas_esp)))
        
        print('Valid points: ' + str(len(clas_esp)))
#         self.Errmsd = np.sqrt(np.sum(((qm_esp - clas_esp)/clas_esp)**2)/len(clas_esp))
        self.Ersd = np.sqrt(np.sum((qm_esp - clas_esp)**2))
        self.esp_points = len(clas_esp)
        return self.Ersd
    
    '''
    Root mean distance metric, this is for measuring how much the center
    positions deviate from the FB center positions.
    '''
    def rmd_centers(self, ref_r, prev_r, next_r):
        re_idx = np.zeros(self.N_occ).astype(int)
        
        # Loop over all centers of the new iteration
        for i, c in enumerate(next_r):
            # Calculate all the distances wrt to all centers from
            # the previous iteration. This to sort them in the 
            # order of ref_r.
            distances = esp.distance_grid(prev_r.transpose(), c)

            # We sort the centers according to the original FB_centers
            re_idx[np.argmin(distances)] = i

        next_r = next_r[re_idx]
        ref_dist = esp.distance_grid((next_r-ref_r).transpose())
        
        return next_r, np.sqrt(np.sum(ref_dist**2)/self.N_occ)
        
    '''
    Internal method of Psi4 for calculating the quadrupole moments.
    This is to check wether the ones calculated above are correct
    '''
    def get_psi4_quadrupole(self):
        # OEProp is a method for evaluating one electron
        # properties
        oe = psi4.core.OEProp(self.model.wavefunction)
        oe.add('QUADRUPOLE')
        oe.compute()
        
        xx = self.model.wavefunction.variable(' QUADRUPOLE XX')
        xy = self.model.wavefunction.variable(' QUADRUPOLE XY')
        xz = self.model.wavefunction.variable(' QUADRUPOLE XZ')
        yy = self.model.wavefunction.variable(' QUADRUPOLE YY')
        yz = self.model.wavefunction.variable(' QUADRUPOLE YZ')
        zz = self.model.wavefunction.variable(' QUADRUPOLE ZZ')
        
        quadrupole_matrix = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
        traceless = quadrupole_matrix - np.trace(quadrupole_matrix) * np.eye(3) / 3.
        
        # Print out the matrix in angstrom**2 instead of debye*angstrom
        return traceless*debye/angstrom
    
    '''
    Function that perform a parameter sweep of the free variable in
    the cost function (if it has one). It writes out the important 
    variables to a data file.
    '''
    def perform_sweep(self, p_min=0., p_max=1., steps=10,
                      scheme='V5', inc_pot=False, folder='none'):
        # Fixing the filename to include sweep parameters
        self.set_scheme(scheme)
        # Making sure that we can write to the right file if we
        # perform sweeps on the hpc
        if folder == 'none':
            folder = '../data/xyz_files/local_runs/' + scheme + '/'
        filename = (self.generate_molname() + '_MIN_' + (str(p_min)[:5]).replace('-','n') 
                    + '_MAX_' + str(p_max)[:5] + '_S_' + str(steps) + '.xyz')
        
        # Making sure the file is empty when writing to it
        open(folder + filename,'w').close()
        
        # Generate the FB reference centers for the rmd metric
        self.set_scheme('FB')
        self.optimize_line_search()
        self.L_centers = np.einsum('ji,jka,ki->ia', self.W,
                                   self.center_matrix, self.W, optimize=True)/angstrom
        FB_ref_centers = self.L_centers
        prev_centers = self.L_centers
        
        # Initializing loop for sweep, every points parameters are
        # writen to the terminal. The quadrupole difference and weigths
        # are also saved and returned.
        weight = np.linspace(p_min,p_max,steps)
        rmd_FB = np.zeros((steps))
        if inc_pot:
            esp_rmsd = np.zeros((steps))
        quad_diff = np.zeros((steps,3,3))
        quad_total = np.zeros((steps,3,3))
        quad_local = np.zeros((steps,3,3))
        V4_cost_val = np.zeros((steps))
        optimizer_convergence = np.zeros((steps))
        n_iterations = 0
        
        for i,w in enumerate(weight):
            self.set_scheme(scheme, w)
            print(self.scheme)
            conv = self.optimize_line_search()
            if not conv:
                print('parameter too high, cost badly conditioned: NO CONVERGENCE/ABORTED')
                n_iterations = i
                break
            self.write_centers(append=True, folder=folder, filename=filename)
            
            # Calculate the rmd wrt to the FB centers.
            next_centers = self.L_centers
            prev_centers, rmd_FB[i] = self.rmd_centers(FB_ref_centers, prev_centers,
                                                       next_centers)
            
            # If requested, we calculate the rmsd of the
            # electrostatic potential. (NOTE: this is expensive)
            if inc_pot:
                esp_rmsd[i] = self.compute_esp_rmsd()
            
            quad_diff[i] = self.compare_quadrupole()
            quad_total[i] = self.total_loc_quadrupole
            quad_local[i] = self.total_quadrupole
            V4_cost_val[i] = self.V4_cost(self.W)
            # pick the last non-zero entry
            optimizer_convergence[i] = self.conv_hist[np.where(self.conv_hist != 0.)][-1]
        
        # We output the relevant data using a dictionary such
        # that it can be easily written to a json file.
        output_dict = {'PenaltyVals' : weight.tolist(),
                       'RootMeanDisplacement' : rmd_FB.tolist(),
                       'QuadDifference' : quad_diff.tolist(),
                       'QuadTotal' : quad_total.tolist(),
                       'QuadLocal' : quad_local.tolist(),
                       'V4CostVals' : V4_cost_val.tolist(),
                       'OptimizerConv' : optimizer_convergence.tolist(),
                       'nIterations' : n_iterations}
        if inc_pot:
            output_dict['ESPrmsd'] = esp_rmsd.tolist()
        
        return output_dict
        
    def calc_shifted_quadrupole(self):
        # calculating the dipole moment using the localized orbitals
        # index i is the occupied orbital index, index a is for 
        # the spatial dimension.
        p_loc = jnp.einsum('ji,jka,ki->ia', self.W, self.center_matrix,
                           self.W, optimize = True)
        
        # Calculating the quadrupole matrix using the occupied orbitals.
        q_loc = jnp.einsum('ji,abjk,ki->iab', self.W, self.quadrupole_matrix_elem,
                           self.W, optimize = True)
        
        # Constructing the matrix which shifts the quadrupole tensor to the origin.
        p_matrix = jnp.einsum('ia,ib->iab', p_loc, p_loc, optimize=True)
        
        # Shifting the quadrupole tensor to the origin.
        self.q_origin = q_loc + p_matrix