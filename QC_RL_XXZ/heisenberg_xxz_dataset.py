import torch
from torch.utils.data import Dataset
import numpy as np

class HeisenbergXXZMeasurementResultData(Dataset):
    def __init__(self,num_qubits=50, num_measure_qubits=3):
        observables = []
        for j in range(0, 3 ** num_measure_qubits):
            observable = np.load(
                'Heisenberg/float_observable' + str(num_measure_qubits) + str(j) + '.npy')
            observables.append(observable)

        index_observables = []
        combination_list = np.load(
            'Heisenberg/' + str(num_qubits) + 'qubit_' + str(num_measure_qubits) + 'combination_list.npy')
        for j in range(0, len(combination_list)):
            for i in range(0, 3 ** num_measure_qubits):
                tmp = np.concatenate((observables[i], np.array([combination_list[j][0]])))
                index_observables.append(tmp)
        self.observables = np.array(index_observables)

        values = []
        J_ps = np.linspace(0, 3, 64)
        deltas = np.linspace(0, 4, 64)
        for id in range(1, len(deltas), 3):
            for iJ in range(1, len(J_ps), 3):
                J_p = J_ps[iJ]
                delta = deltas[id]
                value = np.load(
                    'Heisenberg/prob3_' + str(num_qubits) + 'qubits_Jp' + str(J_p) + '_delta' + str(delta) + '.npy')
                value = value.reshape(-1, 2 ** num_measure_qubits)
                values.append(value)
        self.expectation_values = np.array(values)

    def __getitem__(self, idx):
        assert idx < len(self.expectation_values)
        return self.observables, self.expectation_values[idx]

    def __len__(self):
        return len(self.expectation_values)