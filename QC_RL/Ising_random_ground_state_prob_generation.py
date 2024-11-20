import numpy as np
from tqdm import tqdm

vs = []
for j in range(0,729):
    observable = np.load('6qubit/observable6' + str(j) + '.npy')
    vs.append(np.linalg.eig(observable)[1])

h = 1
num_states = 2000
states = np.load('6qubit/Ising_ground_state_6qubit_random_'+str(num_states)+'.npy')
for i in tqdm(range(0,num_states)):
    values = []
    state = states[i]
    for j in range(0,729):
        tmp = []
        for k in range(0,2**6):
            tmp.append(np.abs(np.inner(state.conj().T,vs[j][:,k]))**2)
        # print(tmp)
        values.append(tmp)
    np.save('6qubit/Ising_ground_state_6qubit_probs_random_'+ str(i),np.array(values))
