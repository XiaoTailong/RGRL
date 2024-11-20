import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

import numpy as np
from tqdm import tqdm
import random
import itertools


from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split


from cgqn import GenerativeQueryNetwork
from cdataset import CatStateData


num_states = 1000
a = 1+1j
p = 0

ds = CatStateData(a=a,p=p,num_states=num_states)

torch.manual_seed(42)
# torch.manual_seed(2)
train_ds, test_ds = random_split(ds, [int(0.9*len(ds)),len(ds) - int(0.9*len(ds))])
train_loader = DataLoader(train_ds,batch_size = 20)
test_loader = DataLoader(test_ds)

# Model
r_dim = 16
z_dim = 32
h_dim = 32
device_ids=range(torch.cuda.device_count())
model = GenerativeQueryNetwork(x_dim=100, v_dim=1,r_dim=r_dim, h_dim=h_dim, z_dim=z_dim, L=2)
model = torch.nn.DataParallel(model, device_ids=device_ids)
model = model.cuda(device=device_ids[0])

try:
    model.load_state_dict(torch.load("models/" + str(r_dim)+"_"+str(h_dim)+"_"+str(z_dim)+'_cat_a'+str(a)+'_p'+str(p)))
    print("Total number of param in Model is ", sum(x.numel() for x in model.parameters()))
except:
    print("No Load!")

test_flag = 0
sigma = 0.1
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epochs = 100
max = 1
test_losses = []

for i in tqdm(range(0, epochs)):
    print(i)
    test_loss = 0
    train_loss = 0
    count1 = 0
    count2 = 0
    tmp = 0
    train_fidelity = 0
    cfidelity = 0
    worst_cfidelity = 0
    if test_flag == 0:
        for v, x in train_loader:
            v = v.cuda(device=device_ids[0])
            x = x.cuda(device=device_ids[0])

            v = v.unsqueeze(dim=2)

            # Sample random number of views for a scene
            batch_size, m, *_ = v.size()
            n_views = 50 + 250*int(random.random())
            # n_views = 50
            #         indices = torch.arange(0,m,dtype=torch.long)
            #         print(indices)

            indices = torch.randperm(m)
            representation_idx, query_idx = indices[:n_views], indices[n_views:]

    #         indices = list(range(1, m))
    #         random.shuffle(indices)
    #         representation_idx, query_idx = [0]+indices[:n_views-1], indices[n_views-1:]

            context_x, context_v = x[:, representation_idx], v[:, representation_idx]
            query_x, query_v = x[:, query_idx], v[:, query_idx]
            context_x = context_x.float()
            context_v = context_v.float()
            query_x = query_x.float()
            query_v = query_v.float()

            (x_mu, r, kl) = model(context_x, context_v, query_x, query_v)
            nll = -Normal(x_mu, sigma).log_prob(query_x)
            reconstruction = torch.mean(nll.view(batch_size, -1), dim=0).mean()
            #         reconstruction = torch.abs((x_mu - query_x).mean())
            kld = torch.mean(kl.view(batch_size, -1), dim=0).mean()

            x_mu = torch.relu(x_mu)
            # x_mu = x_mu * 12 / 99
            # query_x = query_x * 12 / 99

            # print(torch.abs((x_mu - query_x)))
            # print("******")
    #         train_loss += torch.mean((torch.sum(torch.mul(x_mu, query_x), dim=[1, 2]) + torch.sum(torch.mul(context_x, context_x), dim=[1, 2])) / 2**num_bits).item()
            train_loss += torch.abs((x_mu - query_x)).mean()
    #         train_loss += torch.mean((torch.sum(torch.mul(torch.sqrt(x_mu), torch.sqrt(query_x)), dim=[2])) ** 2).item()
            count1 += 1
            #         print(r)
            # print(x_mu)
            #         print(query_x)
            #         print(torch.abs((x_mu - query_x)).mean())
            #         print(kld)
            elbo = reconstruction + kld
            #         print(elbo)
            #         print('---------------')
    #         print(kld)
            elbo.backward()

            optimizer.step()
            optimizer.zero_grad()

            x_mu_sum = torch.sum(x_mu, dim=[2])
            x_mu_sum = x_mu_sum.unsqueeze(dim=2)
            x_mu_sum = x_mu_sum.repeat((1, 1, 100))
            x_mu = torch.div(x_mu, x_mu_sum)

            query_x_sum = torch.sum(query_x, dim=[2])
            query_x_sum = query_x_sum.unsqueeze(dim=2)
            query_x_sum = query_x_sum.repeat((1, 1, 100))
            query_x = torch.div(query_x, query_x_sum)

            train_fidelity+=torch.mean((torch.sum(torch.mul(torch.sqrt(query_x), torch.sqrt(x_mu)), dim=[2]))).item()

            # if tmp%20 == 0:
            #     print(x_mu)
            #     print(torch.abs((x_mu - query_x)).mean())
            #     print(np.count_nonzero(x_mu.detach().numpy() > 0.03) / batch_size/250)
        print(train_loss / count1)
        print(train_fidelity/count1)


    for v, x in test_loader:
        v = v.cuda(device=device_ids[0])
        x = x.cuda(device=device_ids[0])

        v = v.unsqueeze(dim=2)

        batch_size, m, *_ = v.size()
        n_views = 10

        indices = torch.randperm(m)
        representation_idx, query_idx = indices[:n_views], indices[n_views:]


        context_x, context_v = x[:, representation_idx], v[:, representation_idx]

        query_x, query_v = x[:, query_idx], v[:, query_idx]
        # np.save("query_x", query_x)

        context_x = context_x.float()
        context_v = context_v.float()
        query_x = query_x.float()
        query_v = query_v.float()

        x_mu,r,phi = model.module.sample(context_x, context_v, query_v)
        x_mu = torch.relu(x_mu)
        query_x = torch.relu(query_x)
        # print(x_mu.shape)

        x_mu_sum = torch.sum(x_mu,dim=[2])
        x_mu_sum = x_mu_sum.unsqueeze(dim=2)
        x_mu_sum = x_mu_sum.repeat((1,1,100))
        x_mu = torch.div(x_mu,x_mu_sum)


        query_x_sum = torch.sum(query_x, dim=[2])
        query_x_sum = query_x_sum.unsqueeze(dim=2)
        query_x_sum = query_x_sum.repeat((1, 1, 100))
        query_x = torch.div(query_x, query_x_sum)

        cfidelity += torch.mean((torch.sum(torch.mul(torch.sqrt(query_x), torch.sqrt(x_mu)), dim=[2]))).item()
        worst_cfidelity += torch.min((torch.sum(torch.mul(torch.sqrt(query_x), torch.sqrt(x_mu)), dim=[2]))).item()
        test_loss += torch.abs((x_mu - query_x)).mean()

        count2 += 1
        tmp += 1

    #     print('------------')
    print(test_loss / count2)
    print(cfidelity/count2)
    print(worst_cfidelity/count2)

    test_losses.append(cfidelity/count2)
    if test_flag == 0:
        torch.save(model.state_dict(), "models/"+str(r_dim)+"_"+str(h_dim)+"_"+str(z_dim)+'_cat_a'+str(a)+'_p'+str(p))
        torch.save(model.module.state_dict(),
                   "models/" + str(r_dim) + "_" + str(h_dim) + "_" + str(z_dim) + '_cat_a' + str(a) + '_p' + str(p)+'_cpu')









