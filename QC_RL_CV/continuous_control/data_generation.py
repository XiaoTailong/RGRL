import strawberryfields as sf
import numpy as np
from tqdm import tqdm
from strawberryfields.ops import *

num_phi = 300
phis = np.linspace(0, np.pi, num_phi)
scale = np.sqrt(sf.hbar)
quad_axis= np.linspace(-6, 6, 100)*scale

D_range = [0,1]
S_range = [0,1]
R_range = [0,np.pi]

num_states = 1000
a = 1+1j
p = 0

def generate_cat_state(a,p,d,s,r):
    prog_cat = sf.Program(1)
    with prog_cat.context as q:
        sf.ops.Catstate(a=a, p=p) | q
        Dgate(d) | q
        Rgate(r) | q
        Sgate(s) | q

    eng = sf.Engine("bosonic")
    cat = eng.run(prog_cat).state
    return cat

def generate_cat_homodyne_prob(state, phi, quad_axis):
    return state.marginal(0, quad_axis, phi=phi)

def generate_continuous_data(a, p, num_states, quad_axis, num_phi):
    phis = np.linspace(0, np.pi, num_phi)
    cat_probs = []
    random_drs = []

    for i in tqdm(range(num_states)):
        cat_prob = []
        d = np.random.random()
        r = np.random.random() * np.pi
        s = np.random.random()
        cat = generate_cat_state(a,p,d,s,r)
        for j in range(0, len(phis)):
            cat_prob.append(generate_cat_homodyne_prob(cat, phis[j], quad_axis))
        cat_probs.append(cat_prob)
        random_drs.append([d,r,s])
    return np.array(cat_probs), np.array(random_drs)

def calculate_fidelity(state1, state2):
    xvec = np.linspace(-15, 15, 401)
    W1 = state1.wigner(mode=0, xvec=xvec, pvec=xvec)
    W2 = state2.wigner(mode=0, xvec=xvec, pvec=xvec)
    return np.sum(W1*W2*30/400*30/400)*4*np.pi

cat_probs, random_drs = generate_continuous_data(a, p, num_states, quad_axis, num_phi)
np.save('data/cat_probs_'+'a'+str(a)+'_p'+str(p)+'_'+str(num_states)+'states',cat_probs)
np.save('data/cat_random_drs_'+'a'+str(a)+'_p'+str(p)+'_'+str(num_states)+'states',random_drs)


