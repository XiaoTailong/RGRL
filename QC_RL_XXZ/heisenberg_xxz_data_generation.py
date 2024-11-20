import numpy as np
from numpy import linalg as LA
from ncon import ncon
import pickle
from tqdm import tqdm

def Hamiltonian_to_MPO(B, C, D):
    n = len(D)
    assert(len(B) == n)
    assert(len(C) == n)
    chid = B[0].shape[1]
    for i in range(n):
        assert(B[i].shape[1] == chid)
        assert(B[i].shape[2] == chid)
        assert(C[i].shape[2] == chid)
        assert(C[i].shape[1] == chid)
    maxk = 1
    for i in range(n):
        k = B[i].shape[0]
        assert(C[i].shape[0] == k)
        maxk = max(maxk, k)
    M = [0] * n
    chim = maxk + 2
    for i in range(n):
        M[i] = np.zeros([chim, chim, chid, chid], dtype=complex) #left, right, out, in
        M[i][-2, -2, :, :] = np.eye(chid)
        M[i][-1, -1, :, :] = np.eye(chid)
        for k in range(maxk):
            M[i][-2, k, :, :] = B[i][k, :, :] # changed from B[i]
            M[i][k, -1, :, :] = C[i][k, :, :]
        M[i][-2, -1, :, :] = D[i]
    ML = np.array([0] * maxk + [1,0], dtype=complex).reshape(chim,1,1) #left MPO boundary
    MR = np.array([0] * maxk + [0,1], dtype=complex).reshape(chim,1,1) #right MPO boundary
    return ML, M, MR

def doDMRG_MPO_site_dependent(A, ML, M, MR, chi, numsweeps=10, dispon=2, updateon=True, maxit=2, krydim=4):
    """
------------------------
by Glen Evenbly (c) for www.tensors.net, (v1.1) - last modified 19/1/2019
------------------------
Implementation of DMRG for a 1D chain with open boundaries, using the \
two-site update strategy. Each update is accomplished using a custom \
implementation of the Lanczos iteration to find (an approximation to) the \
ground state of the superblock Hamiltonian. Input 'A' is containing the MPS \
tensors whose length is equal to that of the 1D lattice. The Hamiltonian is \
specified by an MPO with 'ML' and 'MR' the tensors at the left and right \
boundaries, and 'M' the bulk MPO tensor. Automatically grow the MPS bond \
dimension to maximum dimension 'chi'. Outputs 'A' and 'B' are arrays of the \
MPS tensors in left and right orthogonal form respectively, while 'sWeight' \
is an array of the Schmidt coefficients across different lattice positions. \
'Ekeep' is a vector describing the energy at each update step.

Optional arguments:
`numsweeps::Integer=10`: number of DMRG sweeps
`dispon::Integer=2`: print data never [0], after each sweep [1], each step [2]
`updateon::Bool=true`: enable or disable tensor updates
`maxit::Integer=2`: number of iterations of Lanczos method for each diagonalization
`krydim::Integer=4`: maximum dimension of Krylov space in superblock diagonalization
"""

## update by Pt-Cr
## site-dependent: M is a list, M[i] is the objective MPO on i-th site
##    2|
## 0--M[i]--1
##    3|

##          |1
## left 0--A[i]--2 right

    ##### left-to-right 'warmup', put MPS in right orthogonal form
    chid = M[0].shape[2]  # local dimension
    Nsites = len(A)
    L = [0 for x in range(Nsites)]
    L[0] = ML
    R = [0 for x in range(Nsites)]
    R[Nsites - 1] = MR
    for p in range(Nsites - 1):
        chil = A[p].shape[0]
        chir = A[p].shape[2]
        utemp, stemp, vhtemp = LA.svd(A[p].reshape(chil * chid, chir), full_matrices=False)
        A[p] = utemp.reshape(chil, chid, chir)
        A[p + 1] = ncon([np.diag(stemp) @ vhtemp, A[p + 1]], [[-1, 1], [1, -2, -3]]) / LA.norm(stemp)
        L[p + 1] = ncon([L[p], M[p], A[p], np.conj(A[p])], [[2, 1, 4], [2, -1, 3, 5], [4, 5, -3], [1, 3, -2]])

    ### A has been normalized
    #      ┌───       ┌───A'[p]──1
    #      │1         │     │
    #    L[p+1]──0 = L[p]──M[p]──0
    #      │2         │     │
    #      └───       └────A[p]──2

    chil = A[Nsites - 1].shape[0]
    chir = A[Nsites - 1].shape[2]
    utemp, stemp, vhtemp = LA.svd(A[Nsites - 1].reshape(chil * chid, chir), full_matrices=False)
    A[Nsites - 1] = utemp.reshape(chil, chid, chir)
    sWeight = [0 for x in range(Nsites + 1)]
    sWeight[Nsites] = (np.diag(stemp) @ vhtemp) / LA.norm(stemp)

    Ekeep = np.array([])
    B = [0 for x in range(Nsites)]
    for k in range(1, numsweeps + 2):

        ##### final sweep is only for orthogonalization (disable updates)
        if k == numsweeps + 1:
            updateon = False
            dispon = 0

        ###### Optimization sweep: right-to-left
        for p in range(Nsites - 2, -1, -1):

            ##### two-site update
            chil = A[p].shape[0]
            chir = A[p + 1].shape[2]
            psiGround = ncon([A[p], A[p + 1], sWeight[p + 2]], [[-1, -2, 1], [1, -3, 2], [2, -4]]).reshape(
                chil * chid * chid * chir)
            if updateon:
                psiGround, Entemp = eigLanczos(psiGround, doApplyMPO, (L[p], M[p], M[p + 1], R[p + 1]), maxit=maxit,
                                               krydim=krydim)
                Ekeep = np.append(Ekeep, Entemp)

            utemp, stemp, vhtemp = LA.svd(psiGround.reshape(chil * chid, chid * chir), full_matrices=False)
            chitemp = min(len(stemp), chi)
            A[p] = utemp[:, range(chitemp)].reshape(chil, chid, chitemp)
            sWeight[p + 1] = np.diag(stemp[range(chitemp)] / LA.norm(stemp[range(chitemp)]))
            B[p + 1] = vhtemp[range(chitemp), :].reshape(chitemp, chid, chir)

            ##### new block Hamiltonian
            R[p] = ncon([M[p + 1], R[p + 1], B[p + 1], np.conj(B[p + 1])], [[-1, 2, 3, 5], [2, 1, 4], [-3, 5, 4], [-2, 3, 1]])
            #   ───┐       1──B'[p+1]──┐
            #      │1          │       │
            #  0──R[p] =   0──M[?]──R[p+1]
            #      │2          │       │
            #   ───┘       2──A[p+1]───┘

            if dispon == 2:
                print('Sweep: %d of %d, Loc: %d,Energy: %f' % (k, numsweeps, p, Ekeep[-1]))

        ###### left boundary tensor
        chil = A[0].shape[0]
        chir = A[0].shape[2]
        Atemp = ncon([A[0], sWeight[1]], [[-1, -2, 1], [1, -3]]).reshape(chil, chid * chir)
        utemp, stemp, vhtemp = LA.svd(Atemp, full_matrices=False)
        B[0] = vhtemp.reshape(chil, chid, chir)
        sWeight[0] = utemp @ (np.diag(stemp) / LA.norm(stemp))

        ###### Optimization sweep: left-to-right
        for p in range(Nsites - 1):

            ##### two-site update
            chil = B[p].shape[0]
            chir = B[p + 1].shape[2]
            psiGround = ncon([sWeight[p], B[p], B[p + 1]], [[-1, 1], [1, -2, 2], [2, -3, -4]]).reshape(
                chil * chid * chid * chir)
            if updateon:
                psiGround, Entemp = eigLanczos(psiGround, doApplyMPO, (L[p], M[p], M[p + 1], R[p + 1]), maxit=maxit,
                                               krydim=krydim)
                Ekeep = np.append(Ekeep, Entemp)

            utemp, stemp, vhtemp = LA.svd(psiGround.reshape(chil * chid, chid * chir), full_matrices=False)
            chitemp = min(len(stemp), chi)
            A[p] = utemp[:, range(chitemp)].reshape(chil, chid, chitemp)
            sWeight[p + 1] = np.diag(stemp[range(chitemp)] / LA.norm(stemp[range(chitemp)]))
            B[p + 1] = vhtemp[range(chitemp), :].reshape(chitemp, chid, chir)

            ##### new block Hamiltonian
            L[p + 1] = ncon([L[p], M[p], A[p], np.conj(A[p])], [[2, 1, 4], [2, -1, 3, 5], [4, 5, -3], [1, 3, -2]])

            ##### display energy
            if dispon == 2:
                print('Sweep: %d of %d, Loc: %d,Energy: %f' % (k, numsweeps, p, Ekeep[-1]))

        ###### right boundary tensor
        chil = B[Nsites - 1].shape[0]
        chir = B[Nsites - 1].shape[2]
        Atemp = ncon([B[Nsites - 1], sWeight[Nsites - 1]], [[1, -2, -3], [-1, 1]]).reshape(chil * chid, chir)
        utemp, stemp, vhtemp = LA.svd(Atemp, full_matrices=False)
        A[Nsites - 1] = utemp.reshape(chil, chid, chir)
        sWeight[Nsites] = (stemp / LA.norm(stemp)) * vhtemp

        if dispon == 1:
            print('Sweep: %d of %d, Energy: %12.12d, Bond dim: %d' % (k, numsweeps, Ekeep[-1], chi))

    return Ekeep, A, sWeight, B

# -------------------------------------------------------------------------
def doApplyMPO(psi, L, M1, M2, R):
    """ function for applying MPO to state """

    return ncon([psi.reshape(L.shape[2], M1.shape[3], M2.shape[3], R.shape[2]), L, M1, M2, R],
                [[1, 3, 5, 7], [2, -1, 1], [2, 4, -2, 3], [4, 6, -3, 5], [6, -4, 7]]).reshape(
        L.shape[2] * M1.shape[3] * M2.shape[3] * R.shape[2])

# -------------------------------------------------------------------------
def eigLanczos(psivec, linFunct, functArgs, maxit=2, krydim=4):
    """ Lanczos method for finding smallest algebraic eigenvector of linear \
    operator defined as a function"""

    if LA.norm(psivec) == 0:
        psivec = np.random.rand(len(psivec))

    psi = np.zeros([len(psivec), krydim + 1], dtype=complex) #Pt-Cr: added complex type 2022.4.25
    A = np.zeros([krydim, krydim], dtype=complex) #Pt-Cr: added complex type 2022.4.25
    dval = 0

    for ik in range(maxit):

        psi[:, 0] = psivec / max(LA.norm(psivec), 1e-16)
        for ip in range(1, krydim + 1):

            psi[:, ip] = linFunct(psi[:, ip - 1], *functArgs)

            for ig in range(ip):
                A[ip - 1, ig] = np.dot(psi[:, ip], psi[:, ig])
                A[ig, ip - 1] = np.conj(A[ip - 1, ig])

            for ig in range(ip):
                psi[:, ip] = psi[:, ip] - np.dot(psi[:, ig], psi[:, ip]) * psi[:, ig]
                psi[:, ip] = psi[:, ip] / max(LA.norm(psi[:, ip]), 1e-16)

        [dtemp, utemp] = LA.eigh(A)
        psivec = psi[:, range(0, krydim)] @ utemp[:, 0]

    psivec = psivec / LA.norm(psivec)
    dval = dtemp[0]

    return psivec, dval

def generate_Heisenberg_state(Nsites, chi=25, J=1, J_p=1, delta=1, OPTS_numsweeps=5, OPTS_dispon=0, OPTS_updateon=True, OPTS_maxit=2, OPTS_krydim=4):
    B = []
    XYZ_h = np.zeros(Nsites)
    XYZ_h[0] = 0.05
    for iN in range(0, Nsites):
        if iN % 2 == 0:
            B.append(np.stack((PauliX * J / 2, PauliY * J / 2, PauliZ * J / 2 * delta)))
            # B.append(np.stack((PauliX * J_p / 2, PauliY * J_p / 2, PauliZ * J_p / 2 * delta)))
        else:
            B.append(np.stack((PauliX * J_p / 2, PauliY * J_p / 2, PauliZ * J_p / 2 * delta)))
            # B.append(np.stack((PauliX * J / 2, PauliY * J / 2, PauliZ * J / 2 * delta)))
    C = [np.stack((PauliX, PauliY, PauliZ)) for i in range(Nsites)]
    D = [-XYZ_h[i] * PauliZ for i in range(Nsites)]
    ML, M, MR = Hamiltonian_to_MPO(B, C, D)
    #### Initialize MPS tensors
    A = [0 for x in range(Nsites)]
    A[0] = np.random.rand(1, chid, min(chi, chid))
    for k in range(1, Nsites):
        A[k] = np.random.rand(A[k - 1].shape[2], chid,
                              min(min(chi, A[k - 1].shape[2] * chid), chid ** (Nsites - k - 1)))
    A1 = A.copy()

    A2 = []
    for i in range(Nsites - 1, -1, -1):
        A2.append(np.transpose(A[i], [2, 1, 0]))  # exchange left and right indicies

    M_sites = []
    for i in range(Nsites):
        M_sites.append(M[i])

    M_sites_rev = []
    for i in range(Nsites - 1, -1, -1):
        M_sites_rev.append(np.transpose(M_sites[i], [1, 0, 2, 3]))  # exchange left and right indicies

    En1, A, sWeight, B = doDMRG_MPO_site_dependent(A1, ML, M_sites, MR, chi, numsweeps=OPTS_numsweeps,
                                                   dispon=OPTS_dispon,
                                                   updateon=OPTS_updateon, maxit=OPTS_maxit, krydim=OPTS_krydim)

    return A

def calculate_fidelity(A,B):
    assert (len(A) == len(B))
    Al_temp = ncon((A[0], B[0]), ((-1, 1, -3), (-2, 1, -4)))
    for i in range(1, len(A)):
        Al_temp = ncon((Al_temp, A[i], B[i]), ((-1, -2, 1, 2), (1, 3, -3), (2, 3, -4)))
    return np.abs(np.squeeze(Al_temp).real)

def calculate_probability_distribution3(A, ms_mats1, ms_mats2, ms_mats3, ms_index):
    if ms_index > 0:
        Al_temp = ncon((A[0], A[0]), ((-1, 1, -3), (-2, 1, -4)))
        for i in range(1, ms_index):
            Al_temp = ncon((Al_temp, A[i], A[i]), ((-1, -2, 1, 2), (1, 3, -3), (2, 3, -4)))
    else:
        Al_temp = np.ones((1, 1, 1, 1))
    if ms_index < len(A) - 3:
        Ar_temp = ncon((A[-1], A[-1]), ((-1, 1, -3), (-2, 1, -4)))
        for j in range(len(A) - 2, ms_index + 2, -1):
            Ar_temp = ncon((Ar_temp, A[j], A[j]), ((1, 2, -3, -4), (-1, 3, 1), (-2, 3, 2)))
    else:
        Ar_temp = np.ones((1, 1, 1, 1))
    probs = []

    for i1 in range(0, 2):
        for i2 in range(0, 2):
            for i3 in range(0, 2):
                A_temp = ncon((Al_temp, A[ms_index], A[ms_index], A[ms_index + 1], A[ms_index + 1],
                               A[ms_index+2], A[ms_index+2], Ar_temp, ms_mats1[i1], ms_mats2[i2], ms_mats3[i3]), (
                              (-1, -2, 1, 2), (1, 3, 5), (2, 4, 6), (5, 7, 9), (6, 8, 10),
                              (9, 11, 13), (10, 12, 14), (13, 14, -3, -4), (3, 4), (7, 8),(11, 12)))
                probs.append(np.squeeze(A_temp))
    return np.array(probs).real


##### Set bond dimensions and simulation options
OPTS_numsweeps = 5 # number of DMRG sweeps
OPTS_dispon = 0 # level of output display
OPTS_updateon = True # level of output display
OPTS_maxit = 2 # iterations of Lanczos method
OPTS_krydim = 4 # dimension of Krylov subspace

PauliZ = np.array([[1,0],[0,-1]], dtype=complex)
PauliX = np.array([[0,1],[1,0]], dtype=complex)
PauliY = np.array([[0,-1j],[1j,0]], dtype=complex)

E, Vx = np.linalg.eig(PauliX)
E, Vy = np.linalg.eig(PauliY)
E, Vz = np.linalg.eig(PauliZ)
V = [Vx, Vy, Vz]
mats = []
for i in range(0,3):
    mat = []
    for j in range(0,2):
        v = V[i][:,j]
        tmp = np.matmul(v.reshape(2,1),v.reshape(1,2).conj())
        mat.append(tmp)
    mats.append(mat)
mats = np.array(mats)


Nsites = 50
chi = 25
chid = 2
J = 1

# J_p = 1
# delta = 1
# A = generate_Heisenberg_state(Nsites=Nsites, J_p=J_p, delta=delta)

J_ps = np.linspace(0,3,64)
deltas = np.linspace(0,4,64)
J_p = J_ps[1]
delta = deltas[1]
file_name = 'Heisenberg/rand' + str(Nsites) + 'qubits_Jp' + str(J_p) + '_delta' + str(delta)
with open(file_name, "rb") as fp:
    A = pickle.load(fp)
state_probs = []
for ms_index in range(0, Nsites - 2):
    for k1 in range(0, 3):
        for k2 in range(0, 3):
            for k3 in range(0, 3):
                prob = calculate_probability_distribution3(A, mats[k1], mats[k2], mats[k3], ms_index)
                state_probs.append(prob)
state_probs = np.array(state_probs)
