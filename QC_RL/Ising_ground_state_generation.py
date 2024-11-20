# Linear Algebra
import numpy as np
from scipy.linalg import svd
from numpy.linalg import qr
import scipy.sparse.linalg.eigen.arpack as arp
import scipy.sparse as sparse
import warnings
import scipy.integrate
import tensornetwork as tn
from tqdm import tqdm


# Own modules
from DMRG import * # here is the main part of the DMRG code

# we'll need them a lot
sxx = np.array([[0., 1.], [1., 0.]])
syy = np.array([[0., -1j], [1j, 0.]])
szz = np.array([[1., 0.], [0., -1.]])

"""Provides exact ground state energies for the transverse field ising model for comparison.

The Hamiltonian reads
.. math ::
    H = - J \\sum_{i} \\sigma^x_i \\sigma^x_{i+1} - g \\sum_{i} \\sigma^z_i
"""
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg.eigen.arpack as arp
import warnings
import scipy.integrate


def exact_E(L, J, h):
    """For comparison: obtain ground state energy from exact diagonalization.
    Exponentially expensive in L, only works for small enough `L` <~ 20.
    """
    if L >= 20:
        warnings.warn("Large L: Exact diagonalization might take a long time!")
    # get single site operaors
    sx = sparse.csr_matrix(np.array([[0., 1.], [1., 0.]]))
    sz = sparse.csr_matrix(np.array([[1., 0.], [0., -1.]]))
    id = sparse.csr_matrix(np.eye(2))
    sx_list = []  # sx_list[i] = kron([id, id, ..., id, sx, id, .... id])
    sz_list = []
    for i_site in range(L):
        x_ops = [id] * L
        z_ops = [id] * L
        x_ops[i_site] = sx
        z_ops[i_site] = sz
        X = x_ops[0]
        Z = z_ops[0]
        for j in range(1, L):
            X = sparse.kron(X, x_ops[j], 'csr')
            Z = sparse.kron(Z, z_ops[j], 'csr')
        sx_list.append(X)
        sz_list.append(Z)
    H_x = sparse.csr_matrix((2**L, 2**L))
    H_zz = sparse.csr_matrix((2**L, 2**L))
    for i in range(L - 1):
        H_zz = H_zz + sz_list[i] * sz_list[(i + 1) % L]
    for i in range(L):
        H_x = H_x + sx_list[i]
    H = -J * H_zz - h * H_x
    E, V = arp.eigsh(H, k=2, which='SA', return_eigenvectors=True, ncv=20)
    return V[:,0], E

def calculate_optimal_MPS_DMRG(J,h,L,d=2,iterations = 5):
    mpo = Ising_MPO(L=L, d=d, h=h, J=J)
    init_mps = random_MPS(L=L, d=d, chi_max=20)
    dmrg = DMRG(init_mps, mpo, chi_max=20, test=False)
    for i in range(0, iterations):
        dmrg.left_to_right()
        dmrg.right_to_left()
    mps = dmrg.MPS
    a = tn.Node(mps.Ms[0])
    for i in range(1, L):
        tmp = tn.Node(mps.Ms[i])
        edge = a[-1] ^ tmp[0]
        a = tn.contract(edge)
    a = np.squeeze(a.tensor)
    a = a.reshape(2 ** L)
    return a

state_num = int(input("number of states:"))
L = 6
h = 0.5
ratio_space = np.linspace(0.5,2,state_num)
for i in tqdm(range(0,state_num)):
    J = ratio_space[i]*h
    state = exact_E(L,J,h)
    # state2 = calculate_optimal_MPS_DMRG(J,h,L)
    np.save('Ising/Ising_ground_state_6qubit_exact' + str(i), state)
    # np.save('Ising/Ising_ground_state_10qubit_dmrg' + str(i), state2)

