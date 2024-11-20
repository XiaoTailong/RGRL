import torch
from torch.utils.data import Dataset
import numpy as np

class StateMeasurementResultData(Dataset):
    def __init__(self,num_observables=27,num_states=2000):
        observables = []
        for i in range(0,num_observables):
            tmp = np.load('6qubit/float_observable6'+str(i)+'.npy')
            observables.append(np.array(tmp))
        self.observables = np.array(observables)

        values = []
        for i in range(0, num_states):
            tmp = np.load('6qubit/Ising_ground_state_6qubit_probs_random_'+ str(i)+'.npy')
            tmp = tmp.reshape(-1, 2**6)
            values.append(np.array(tmp, dtype=np.float32))
        self.expectation_values = np.array(values)

    def __getitem__(self, idx):
        assert idx < len(self.expectation_values)
        return self.observables, self.expectation_values[idx]

    def __len__(self):
        return len(self.expectation_values)



