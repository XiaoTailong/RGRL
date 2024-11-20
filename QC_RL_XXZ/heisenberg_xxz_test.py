import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.distributions import Normal, kl_divergence

import numpy as np
from tqdm import tqdm
import random
import itertools

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


from gqn import GenerativeQueryNetwork
from heisenberg_xxz_dataset import HeisenbergXXZMeasurementResultData
# Data

test_flag = 0
Nsites = num_qubits = 50
num_measure_qubits = num_bits = 3
split_ratio = 0.9
num_observables = 3**num_bits*(Nsites-2)

ds = HeisenbergXXZMeasurementResultData(num_qubits=Nsites,num_measure_qubits=num_bits)
train_size = 40 #int(0.2 * len(ds))
test_size = len(ds) - train_size
torch.manual_seed(10)
# generator1 = torch.Generator().manual_seed(10)
train_ds, test_ds = random_split(ds, [train_size, test_size])
# torch.manual_seed(42)
# train_ds, test_ds = random_split(ds, [int(split_ratio*len(ds)),len(ds)-int(split_ratio*len(ds))])
# test_indices = test_ds.indices
# np.save("10qubit_ground_test_indices_partial2",test_indices)
train_loader = DataLoader(train_ds, batch_size=20)
test_loader = DataLoader(test_ds)



# Model
device_ids=range(torch.cuda.device_count())
r_dim = 16
h_dim = 48
z_dim = 16
model = GenerativeQueryNetwork(x_dim=2**num_bits, v_dim=4**num_bits*2+1,r_dim=r_dim, h_dim=h_dim, z_dim=z_dim, L=2)
model = torch.nn.DataParallel(model, device_ids=device_ids)
model = model.cuda(device=device_ids[0])
try:
    model.load_state_dict(torch.load('models/XXZ_' + str(Nsites) + 'qubit_partial3_'+str(r_dim)+'_'+str(h_dim)+'_'+str(z_dim)+'_softmax'))
    print("Total number of param in Model is ", sum(x.numel() for x in model.parameters()))
except:
    print("NO load")
# torch.save(model.state_dict(), 'GHZ_state_6qubit_9_0.1pi_32_32_16_2_softmax_good')

sigma = 0.1

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
observables = np.array(index_observables)
observables = observables.reshape((1, num_observables,-1))
v = torch.Tensor(observables)

J_ps = np.linspace(0, 3, 64)
deltas = np.linspace(0, 4, 64)
J_p = J_ps[4]
delta = deltas[4]
value = np.load(
    'Heisenberg/prob3_' + str(num_qubits) + 'qubits_Jp' + str(J_p) + '_delta' + str(delta) + '.npy')
probs = value.reshape((1,-1, 2**num_bits))
x = torch.Tensor(probs)


v = v.cuda(device=device_ids[0])
x = x.cuda(device=device_ids[0])

batch_size, m, *_ = v.size()
# print(m)
# n_views = int((num_observables-1) * random.random())+1
n_views = 50
# print(n_views)
# n_views = 50
# indices = torch.arange(0,m,dtype=torch.long)
# indices = torch.randperm(m)
indices = list(range(0, m))
random.shuffle(indices)
representation_idx, query_idx = indices[:n_views], indices[n_views:]
# representation_idx, query_idx = [0]

# representation_idx, query_idx = indices[:n_views], indices[n_views:]
context_x, context_v = x[:, representation_idx], v[:, representation_idx]
query_x, query_v = x[:, query_idx], v[:, query_idx]
context_x = context_x.float()
context_v = context_v.float()
query_x = query_x.float()
query_v = query_v.float()

test_x = torch.tensor(np.ones(query_x.shape)/(2**num_bits)).cuda(device=device_ids[0])

x_mu, r, phi = model.module.sample(context_x, context_v, query_v)
x_mu = torch.relu(x_mu)
# np.save("test",x_mu.detach().cpu().float())

tmp = (torch.sum(torch.mul(torch.sqrt(x_mu), torch.sqrt(query_x)), dim=[2]))
sorted, indices = torch.sort(tmp)
# print(tmp.shape)

# test_loss += torch.mean((torch.sum(torch.mul(torch.sqrt(x_mu), torch.sqrt(query_x)), dim=[2])) ** 2).item()
test_loss = torch.mean(sorted).item()
test_loss2 = torch.mean(torch.abs(x_mu-query_x))
refer_loss = torch.mean((torch.sum(torch.mul(torch.sqrt(test_x), torch.sqrt(query_x)), dim=[2])) ).item()

print(test_loss)
print(test_loss2)
print(refer_loss)

