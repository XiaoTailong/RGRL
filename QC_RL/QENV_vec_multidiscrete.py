from abc import ABC


import numpy as np

import torch
import gym
from gym import spaces
import pygame
from gqn2 import GenerativeQueryNetwork

import scipy.sparse as sparse
import scipy.sparse.linalg as arp
import warnings


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class QIsingEnv(gym.Env):

    def __init__(self, num_qubits=6, target_param=None, maximum_counter=50,
                 r_dim=32,
                 h_dim=96, z_dim=32, L=2,
                 n_views=30, n_views_target=729, action_step=0.01):
        self.num_qubits = num_qubits
        # self.h_dim = h_dim
        self.action_step = action_step

        if target_param is None:
            # randomly generating the target parameters
            self.target_param = self.np_random.random(self.num_qubits) * 2 - 1
        else:
            self.target_param = target_param

        # if agent_param is None:
        #     self.agent_param = self.np_random.random(self.num_qubits, dtype=np.float32) * 2 - 1
        #     param_distance = np.linalg.norm(self.target_param - self.agent_param, ord=1)
        #     # We will sample the target's location randomly until it does not coincide with the agent's location
        #     while param_distance < 6.0:
        #         self.agent_param = self.np_random.random(self.num_qubits) * 2 - 1
        #         param_distance = np.linalg.norm(self.target_param - self.agent_param, ord=1)
        # else:
        #     self.agent_param = agent_param

        # self.r_dim = r_dim
        # self.z_dim = z_dim
        self.n_views = n_views
        self.n_views_target = n_views_target

        self.maximum_counter = maximum_counter
        self.counter = 1

        # Observations are dictionaries with the agent's and the target's neural representation.
        # self.observation_space = spaces.Dict(
        #     {
        #         "agent": spaces.Box(low=-np.inf, high=np.inf, shape=(z_dim,), dtype=np.float32),
        #         "target": spaces.Box(low=-np.inf, high=np.inf, shape=(z_dim,), dtype=np.float32),
        #     }
        # )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(z_dim,), dtype=np.float32)

        # We have 18 actions, each arm corresponding to 3 actions. left, right and do not move.
        self.action_space = spaces.MultiDiscrete(np.array([2, 2, 2, 2, 2, 2]))
        # action space, for random Ising, 18 actions

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """

        # It can be initialized as ``MultiDiscrete([ 5, 2, 2 ])`` such that a sample might be
        # # ``array([2, 2, 1, 0, 1, 0])``.
        self._action_to_direction = {
            0: 1,
            1: -1,
        }

        self.model = GenerativeQueryNetwork(x_dim=2 ** num_qubits,
                                            v_dim=4 ** num_qubits * 2,
                                            r_dim=r_dim,
                                            h_dim=h_dim,
                                            z_dim=z_dim, L=L)

        # model = torch.nn.DataParallel(model, device_ids=device_ids)
        # model = model.cuda(device=device_ids[0])

        try:
            import os
            current_directory = os.path.dirname(os.path.abspath(__file__))
            self.model.load_state_dict(torch.load(
                current_directory+'/models/new_model/Ising_random_' + str(num_qubits) + 'qubit_' + str(r_dim)
                + '_' + str(h_dim) + '_' + str(z_dim) + '_softmax_cpu', map_location='cpu'))
            self.model.to(device)
            self.model.eval()
            print("Total number of param in Model is ", sum(x.numel() for x in self.model.parameters()))
        except:
            print("NO Load")

        # load observables
        self.observable = []  # float
        # for target state, we use all the observables, i.e. 729
        for i in range(0, 3 ** self.num_qubits):
            tmp = np.load(str(self.num_qubits) + 'qubit/float_observable' + str(self.num_qubits) + str(i) + '.npy')
            self.observable.append(np.array(tmp))

        self.vs = []  # complex
        for j in range(0, 3 ** self.num_qubits):
            observable = np.load(str(self.num_qubits) + 'qubit/observable' +
                                 str(self.num_qubits) + str(j) + '.npy')
            self.vs.append(np.linalg.eig(observable)[1])

        self.target_observation, self.target_state = self._get_target_obs()

    def _get_target_obs(self):
        """
        generate the neural representation of target state
        :return: the neural representation of target state
        """

        observables = np.array(self.observable)  # 729, 64, 64
        observables = observables.reshape((1, 3 ** self.num_qubits, -1))
        v = torch.FloatTensor(observables).to(device)  # 1, 729, 4096

        target_state = exact_E_rand_Js(self.num_qubits, self.target_param, h=1)
        probs = exact_E_rand_Js_probs(target_state, self.num_qubits, self.vs)
        probs = np.array(probs)  # 729, 64
        probs = probs.reshape((1, -1, 2 ** self.num_qubits))
        x = torch.FloatTensor(probs).to(device)

        batch_size, m, *_ = v.size()
        indices = list(range(0, m))
        np.random.shuffle(indices)
        representation_idx, query_idx = indices[:self.n_views_target], indices[self.n_views_target:]
        context_x, context_v = x[:, representation_idx], v[:, representation_idx]

        b, m, *x_dims = context_x.shape
        _, _, *v_dims = context_v.shape
        context_x = context_x.view((-1, *x_dims))
        context_v = context_v.view((-1, *v_dims))

        with torch.no_grad():
            phi = self.model.representation(context_x, context_v)
            _, *phi_dims = phi.shape
            phi = phi.view((b, m, *phi_dims))
            # sum over n_views to obtain representations
            neural_rep_target = torch.mean(phi, dim=1)

        return neural_rep_target.squeeze().cpu().numpy(), target_state

    def _get_agent_obs(self):
        """
        both the target and agent observation should output
        :return: returns the observation of the quantum ground state of the Hamiltonian
        """
        observables = np.array(self.observable)
        observables = observables.reshape((1, 3 ** self.num_qubits, -1))
        v = torch.FloatTensor(observables).to(device)

        agent_state = exact_E_rand_Js(self.num_qubits, self.agent_param, h=1)
        probs = exact_E_rand_Js_probs(agent_state, self.num_qubits, self.vs)
        probs = np.array(probs)
        probs = probs.reshape((1, -1, 2 ** self.num_qubits))
        x = torch.FloatTensor(probs).to(device)
        # each time to randomly choose the views to enhance the robustness
        batch_size, m, *_ = v.size()
        indices = list(range(0, m))
        np.random.shuffle(indices)
        representation_idx, query_idx = indices[:self.n_views], indices[self.n_views:]
        context_x, context_v = x[:, representation_idx], v[:, representation_idx]

        b, m, *x_dims = context_x.shape
        _, _, *v_dims = context_v.shape
        context_x = context_x.view((-1, *x_dims))
        context_v = context_v.view((-1, *v_dims))

        with torch.no_grad():
            phi = self.model.representation(context_x, context_v)
            _, *phi_dims = phi.shape
            phi = phi.view((b, m, *phi_dims))
            # sum over n_views to obtain representations
            neural_rep_agent = torch.mean(phi, dim=1)

        return neural_rep_agent.squeeze().cpu().numpy(), agent_state

    def _get_info(self, agent_state, target_state):
        """
        :return: returns the distance of the target and agent hamiltonian cofficients
        """
        quantum_fidelity = np.abs(np.inner(agent_state.conj().T, target_state)) ** 2

        return {"param_distance": np.linalg.norm(self.agent_param - self.target_param, ord=1),
                "quantum_fidelity": quantum_fidelity}

    def reset(self, seed=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        # if seed is None:
        # super().reset()
        np.random.seed(seed)
        # Choose the agent's location uniformly at random
        self.agent_param = self.np_random.random(self.num_qubits, dtype=np.float32) * 2 - 1

        param_distance = np.linalg.norm(self.target_param - self.agent_param, ord=1)
        # We will sample the target's location randomly until it does not coincide with the agent's location
        while param_distance < 2.0:
            self.agent_param = self.np_random.random(self.num_qubits) * 2 - 1
            param_distance = np.linalg.norm(self.target_param - self.agent_param, ord=1)

        agent_observation, agent_state = self._get_agent_obs()
        info = self._get_info(agent_state, self.target_state)
        self.counter = 1

        return agent_observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3,  ... 11}) to the direction we walk in
        # the neural network output is the index of the neuron
        action_change = np.zeros(self.num_qubits)
        for i, action_element in enumerate(action):
            action_element = int(action_element)
            direction = self._action_to_direction[action_element]
            action_change[i] = direction * self.action_step

        # update the parameter once for each local coefficient J_i, 一次产生6个动作
        self.agent_param = np.clip(self.agent_param + action_change, -1, 1)
        agent_observation, agent_state = self._get_agent_obs()
        # ----------------[ reward design, option 1] -----------------------#
        normalized_euclidean_distance = (np.linalg.norm(agent_observation - self.target_observation)
                                         / np.sqrt(agent_observation.shape[0]))
        reward = -1 * normalized_euclidean_distance * 10  # data range (-1, 0)

        # --------------- [reward design, option 2]-----------------------#
        # # normalized distance ranged in [0,1]
        # if distance > -1 * threshold:
        #     reward = 0
        # else:
        #     reward = -1 * distance  # negative number

        # ----------------[reward design, sparse reward, the end of the episode give the reward]
        info = self._get_info(agent_state, self.target_state)
        if reward > -1 * 0.5 or self.counter >= self.maximum_counter:
            done = 1
        else:
            done = 0

        self.counter = 1 if done else self.counter + 1

        truncated = False

        return agent_observation, reward, done, truncated, info


def exact_E_rand_Js(L, Js, h):
    """
    For comparison: obtain ground state energy from exact diagonalization.
    Exponentially expensive in L, only works for small enough `L` <~ 20.
    """
    if L >= 20:
        warnings.warn("Large L: Exact diagonalization might take a long time!")
    # get single site operaors
    sx = sparse.csr_matrix(np.array([[0., 1.], [1., 0.]]))
    sz = sparse.csr_matrix(np.array([[1., 0.], [0., -1.]]))
    id = sparse.csr_matrix(np.eye(2))
    sx_list = []  # sx_list[i] = kron([id, id, ..., id, sx, id, .... id])
    sz_list = []
    for i_site in range(L):
        x_ops = [id] * L
        z_ops = [id] * L
        x_ops[i_site] = sx
        z_ops[i_site] = sz
        X = x_ops[0]
        Z = z_ops[0]
        for j in range(1, L):
            X = sparse.kron(X, x_ops[j], 'csr')
            Z = sparse.kron(Z, z_ops[j], 'csr')
        sx_list.append(X)
        sz_list.append(Z)
    H_x = sparse.csr_matrix((2 ** L, 2 ** L))
    H_zz = sparse.csr_matrix((2 ** L, 2 ** L))
    for i in range(L - 1):
        rand_J = Js[i]
        H_zz = H_zz + rand_J * sz_list[i] * sz_list[(i + 1) % L]
    for i in range(L):
        H_x = H_x + sx_list[i]
    H = -H_zz - h * H_x
    E, V = arp.eigsh(H, k=2, which='SA', return_eigenvectors=True, ncv=20)
    return V[:, 0]


def exact_E_rand_Js_probs(state, num_bit, vs):
    values = []
    for j in range(0, 3 ** num_bit):
        tmp = []
        for k in range(0, 2 ** num_bit):
            tmp.append(np.abs(np.inner(state.conj().T, vs[j][:, k])) ** 2)
        values.append(tmp)
    return values


def test_env():
    env = QIsingEnv()
    observation = env.reset()
    agent_observation, reward, done, info = env.step(6)

    print(agent_observation)
    print(reward)

# test_env()
