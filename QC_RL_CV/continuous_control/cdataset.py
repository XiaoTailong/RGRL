from torch.utils.data import Dataset
import numpy as np
import random


class CatStateData(Dataset):
    def __init__(self,a,p,num_states=2500):
        self.observables = np.load('data/phis_p9.npy')
        self.values = np.load('data/cat_probs_'+'a'+str(a)+'_p'+str(p)+'_'+str(num_states)+'states.npy')
    def __getitem__(self, idx):
        assert idx < len(self.values)
        return self.observables, self.values[idx]
    def __len__(self):
        return len(self.values)
