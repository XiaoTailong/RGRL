import strawberryfields as sf
import numpy as np
from tqdm import tqdm
from strawberryfields.ops import *

num_phi = 300
phis = np.linspace(0, np.pi, num_phi)
scale = np.sqrt(sf.hbar)
quad_axis= np.linspace(-6, 6, 100)*scale

# D_range = [-1,1]
# S_range = [0,1]
# R_range = [0,np.pi]

# num_states = 2000
# a = 1+1j
# p = 0

# def generate_cat_state(a,p,d,s,r1,r2):
#     prog_cat = sf.Program(1)
#     with prog_cat.context as q:
#         sf.ops.Catstate(a=a, p=p) | q
#         Rgate(r1) | q
#         Sgate(s) | q
#         Rgate(r2) | q
#         Dgate(d) | q
#     eng = sf.Engine("bosonic")
#     cat = eng.run(prog_cat).state
#     return cat
#
# def generate_new_cat_state(org_state,d,r,s):
#     rs, Vs, ws = org_state.data
#     prog_cat = sf.Program(1)
#     with prog_cat.context as q:
#         sf.ops.Bosonic(weights=ws, covs=Vs, means=rs) | q
#         Dgate(d) | q
#         Rgate(r) | q
#         Sgate(s) | q
#     eng = sf.Engine("bosonic")
#     cat = eng.run(prog_cat).state
#     return cat

def generate_cat_state(a,p=0):
    prog_cat = sf.Program(1)
    with prog_cat.context as q:
        sf.ops.Catstate(a=a,p=p) | q
    eng = sf.Engine("bosonic")
    cat = eng.run(prog_cat).state
    return cat

def generate_cat_homodyne_prob(state,phi, quad_axis):
    return state.marginal(0, quad_axis, phi=phi)

# def generate_continuous_data(a, p, num_states, quad_axis, num_phi):
#     phis = np.linspace(0, np.pi, num_phi)
#     cat_probs = []
#     random_drs = []
#
#     for i in tqdm(range(num_states)):
#         cat_prob = []
#         d = np.random.random()*2-1
#         r1 = np.random.random() * np.pi
#         r2 = np.random.random() * np.pi
#         s = np.random.random()*0.1
#         cat = generate_cat_state(a,p,d,s,r1,r2)
#         for j in range(0, len(phis)):
#             cat_prob.append(generate_cat_homodyne_prob(cat, phis[j], quad_axis))
#         cat_probs.append(cat_prob)
#         random_drs.append([d,r1,r2,s])
#     return np.array(cat_probs), np.array(random_drs)

def calculate_fidelity(state1, state2):
    xvec = np.linspace(-15, 15, 401)
    W1 = state1.wigner(mode=0, xvec=xvec, pvec=xvec)
    W2 = state2.wigner(mode=0, xvec=xvec, pvec=xvec)
    return np.sum(W1*W2*30/400*30/400)*4*np.pi

def generate_continuous_data(num_alpha0, alpha0_range, quad_axis, num_phi):
    alpha0s = np.linspace(alpha0_range[0], alpha0_range[1], num_alpha0)
    alpha0_list = []
    phis = np.linspace(0, np.pi, num_phi)
    for i in range(0, num_alpha0):
        for j in range(0, num_alpha0):
            alpha0_list.append(alpha0s[i] + alpha0s[j] * 1j)
    # Create cat state
    cat_probs = []
    for i in tqdm(range(0, len(alpha0_list))):
        cat_prob = []
        prog_cat = sf.Program(1)
        with prog_cat.context as q:
            sf.ops.Catstate(a=alpha0_list[i]) | q

        eng = sf.Engine("bosonic")
        cat = eng.run(prog_cat).state
        for j in range(0, len(phis)):
            cat_prob.append(cat.marginal(0, quad_axis, phi=phis[j]))
        cat_probs.append(cat_prob)
    return np.array(cat_probs)

alpha0_range = [-2,2]
num_alpha0 = 81
num_states = 6561
cat_probs = generate_continuous_data(num_alpha0, alpha0_range, quad_axis, num_phi)
np.save('data/cat_probs_'+str(num_states)+'states_alpha_minus2_plus2',cat_probs)
# np.save('data/cat_random_drs_'+'a'+str(a)+'_p'+str(p)+'_'+str(num_states)+'states_s0.1',random_drs)


