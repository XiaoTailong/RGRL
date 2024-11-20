import torch
torch.cuda.empty_cache()

import numpy as np
from torch.utils.data import DataLoader
from cgqn import GenerativeQueryNetwork
from cdataset import KerrData_output, TestKerrData2

# Data
num_phi = 100
num_train_states = 50
kappa = 1
var = 1

# Model
r_dim = 32
h_dim = 96
z_dim = 32
device_ids=range(torch.cuda.device_count())
model = GenerativeQueryNetwork(x_dim=100, v_dim=1,r_dim=r_dim, h_dim=h_dim, z_dim=z_dim, L=2)
model = torch.nn.DataParallel(model, device_ids=device_ids)
model = model.cuda(device=device_ids[0])
try:
    model.load_state_dict(torch.load(
        "models/" + str(r_dim) + '_' + str(h_dim) + '_' + str(
                   z_dim) + "_tanh_kerr_thermal_range_0_3_num_phi" + str(num_phi) + "_kappa" + str(kappa) + "_var" + str(var) + "_" + str(
                   num_train_states) + "trainstates_output"))

except:
    print("NO LOAD!")


sigma = 0.1


observables = np.linspace(0,1,num_phi+1)[0:num_phi]*np.pi
observables = np.array(observables).reshape((1, num_phi))
v = torch.Tensor(observables)

probs = np.load('test_output_thermal_data_para_range_0_3_kappa'+str(kappa)+'_phi' + str(num_phi) +'_var'+str(var)+'.npy')[0]
probs = probs.reshape((1,100, 100))
x = torch.Tensor(probs)


v = v.cuda(device=device_ids[0])
x = x.cuda(device=device_ids[0])
v = v.unsqueeze(dim=2)

batch_size, m, *_ = v.size()
n_views = 10

indices = torch.randperm(m)
representation_idx, query_idx = indices[:n_views], indices[n_views:]


context_x, context_v = x[:, representation_idx], v[:, representation_idx]
query_x, query_v = x[:, query_idx], v[:, query_idx]

context_x = context_x.float()
context_v = context_v.float()
query_x = query_x.float()
query_v = query_v.float()

x_mu,r,phi = model.module.sample(context_x, context_v, query_v)
x_mu = torch.relu(x_mu)
query_x = torch.relu(query_x)

x_mu_sum = torch.sum(x_mu,dim=[2])
x_mu_sum = x_mu_sum.unsqueeze(dim=2)
x_mu_sum = x_mu_sum.repeat((1,1,100))
x_mu = torch.div(x_mu,x_mu_sum)

query_x_sum = torch.sum(query_x, dim=[2])
query_x_sum = query_x_sum.unsqueeze(dim=2)
query_x_sum = query_x_sum.repeat((1, 1, 100))
query_x = torch.div(query_x, query_x_sum)

cfidelity = torch.mean((torch.sum(torch.mul(torch.sqrt(query_x), torch.sqrt(x_mu)), dim=[2]))).item()
test_loss = torch.abs((x_mu - query_x)).mean()


#     print('------------')
print(test_loss)
print(cfidelity)














