import numpy as np
import itertools
from tqdm import tqdm

Pauli_I = np.array([[1,0],[0,1]],dtype=np.complex64)
Pauli_x = np.array([[0,1],[1,0]],dtype=np.complex64)
Pauli_y = np.array([[0,-1j],[1j,0]],dtype=np.complex64)
Pauli_z = np.array([[1,0],[0,-1]],dtype=np.complex64)

num_bit = int(input("number of bits:"))


def generate_basis_observables(num_bit):
    permutation_list = list(itertools.product(list(range(1,4)),repeat=num_bit))
    # float_observables = []
    # observables = []
    print(len(permutation_list))
    for k in tqdm(range(0,len(permutation_list))):
        index_list = []
        if permutation_list[k][0] == 0:
            tmp = Pauli_I
        elif permutation_list[k][0] == 1:
            tmp = Pauli_x
            index_list += [1, 0, 0]
        elif permutation_list[k][0] == 2:
            tmp = Pauli_y
            index_list += [0, 1, 0]
        elif permutation_list[k][0] == 3:
            tmp = Pauli_z
            index_list += [0, 0, 1]
        for i in range(1,num_bit):
            if permutation_list[k][i] == 0:
                tmp = np.kron(tmp,Pauli_I)
            elif permutation_list[k][i] == 1:
                tmp = np.kron(tmp,Pauli_x)
                index_list += [1, 0, 0]
            elif permutation_list[k][i] == 2:
                tmp = np.kron(tmp,Pauli_y)
                index_list += [0, 1, 0]
            elif permutation_list[k][i] == 3:
                tmp = np.kron(tmp,Pauli_z)
                index_list += [0, 0, 1]
        observable = tmp
        np.save(str(num_bit) + 'qubit/observable' + str(num_bit) + str(k), observable)
        # observables.append(observable)

        observable = observable.reshape(observable.shape[0]*observable.shape[1])
        observable_real = observable.real
        observable_imag = observable.imag
        observable = np.concatenate((observable_real,observable_imag))
        np.save(str(num_bit) + 'qubit/float_observable' + str(num_bit) + str(k), observable)
        # float_observables.append(observable)
        # np.save(str(num_bit) + 'qubit/observable_index'+ str(num_bit) + str(k), np.array(index_list))

    return 0

_ = generate_basis_observables(num_bit)
# for i in range(0,len(basis_observables)):
#     np.save(str(num_bit)+'qubit/observable'+str(num_bit)+str(i),basis_observables[i])
#     np.save(str(num_bit)+'qubit/float_observable'+str(num_bit)+ str(i), float_basic_observables[i])


