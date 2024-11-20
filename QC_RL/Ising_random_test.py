import torch
torch.cuda.empty_cache()

import numpy as np
import random

from gqn2 import GenerativeQueryNetwork

# Data
num_bits = 6
num_observables = 729

# Model
device_ids=range(torch.cuda.device_count())
r_dim = 32
h_dim = 96
z_dim = 32
sigma = 0.1
model = GenerativeQueryNetwork(x_dim=2**num_bits, v_dim=4**num_bits*2,r_dim=r_dim, h_dim=h_dim, z_dim=z_dim, L=2)
model = torch.nn.DataParallel(model, device_ids=device_ids)
model = model.cuda(device=device_ids[0])
try:
    model.load_state_dict(torch.load('models/Ising_random_' + str(num_bits) + 'qubit_' + str(r_dim) + '_' + str(h_dim) + '_' + str(z_dim) + '_softmax'))
    print("Total number of param in Model is ", sum(x.numel() for x in model.parameters()))
except:
    print("NO Load")


observables = []
for i in range(0,num_observables):
    tmp = np.load(str(num_bits)+'qubit/float_observable'+str(num_bits)+str(i)+'.npy')
    observables.append(np.array(tmp))
observables = np.array(observables)
observables = observables.reshape((1, num_observables,-1))
v = torch.Tensor(observables)

probs = np.load(str(num_bits)+'qubit/Ising_ground_state_'+str(num_bits)+'qubit_probs_random_1999.npy')
probs = probs.reshape((1,-1, 2**num_bits))
x = torch.Tensor(probs)

# probs2 = np.load(str(num_bits)+'qubit/Ising_ground_state_'+str(num_bits)+'qubit_probs_random_5.npy')
# probs2 = probs2.reshape((1,-1, 2**num_bits))
# x2 = torch.Tensor(probs2)


v = v.cuda(device=device_ids[0])
x = x.cuda(device=device_ids[0])

# x2 = x2.cuda(device=device_ids[0])

batch_size, m, *_ = v.size()
n_views = 30 # select n_views measurement basis randomly as the inputs of the representation network
indices = list(range(0, m))
random.shuffle(indices)
representation_idx, query_idx = indices[:n_views], indices[n_views:]

context_x, context_v = x[:, representation_idx], v[:, representation_idx]
query_x, query_v = x[:, query_idx], v[:, query_idx]
context_x = context_x.float()
context_v = context_v.float()
query_x = query_x.float()
query_v = query_v.float()

# query_x2 = x2[:, query_idx]
# query_x2 = query_x2.float()

test_x = torch.tensor(np.ones(query_x.shape)/(2**num_bits)).cuda(device=device_ids[0])

x_mu , r, phi = model.module.sample(context_x, context_v, query_v)
x_mu = torch.relu(x_mu)

test_loss = torch.mean((torch.sum(torch.mul(torch.sqrt(x_mu), torch.sqrt(query_x)), dim=[2])) ).item()
refer_loss = torch.mean((torch.sum(torch.mul(torch.sqrt(test_x), torch.sqrt(query_x)), dim=[2])) ).item()
worst_fidelity = torch.min((torch.sum(torch.mul(torch.sqrt(x_mu), torch.sqrt(query_x)), dim=[2])) ).item()

# test_loss2 = torch.mean((torch.sum(torch.mul(torch.sqrt(x_mu), torch.sqrt(query_x2)), dim=[2])) ).item()

print(test_loss)
print(refer_loss)
print(worst_fidelity)
# print(test_loss2)









