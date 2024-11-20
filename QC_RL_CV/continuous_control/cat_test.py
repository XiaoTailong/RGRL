import torch
torch.cuda.empty_cache()

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
# train_loader = DataLoader(train_ds,batch_size = 20)
test_loader = DataLoader(test_ds)

# Model
r_dim = 16
z_dim = 32
h_dim = 32
device_ids=range(torch.cuda.device_count())
model = GenerativeQueryNetwork(x_dim=100, v_dim=1, r_dim=r_dim, h_dim=h_dim, z_dim=z_dim, L=2)
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

for v, x in test_loader:
    v = v.cuda(device=device_ids[0])
    x = x.cuda(device=device_ids[0])
    v = v.unsqueeze(dim=2)

    # print(x.shape)
    # print(v.shape)

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

    x_mu, r, phi = model.module.sample(context_x, context_v, query_v)




