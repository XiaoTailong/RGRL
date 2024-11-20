#!/usr/bin/python
import random

import numpy as np

from abc import ABC
from ncon import ncon
import torch
import gym
from gym import spaces
import pygame
from representation import RepresentationNetwork
import pickle
import scipy.sparse as sparse
import scipy.sparse.linalg as arp
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Backbone(nn.Module):
    def __init__(self, x_dim, v_dim, r_dim):
        """
        A neural network with skip connections which converts the input observable and
        its corresponding expectation into a representation which is more dense than the input.
        """
        super(Backbone, self).__init__()
        # Final representation size
        self.r_dim = k = r_dim
        self.v_dim = v_dim
        self.linear1 = nn.Linear(v_dim, k)
        self.linear2 = nn.Linear(x_dim, k)
        self.linear3 = nn.Linear(3, k)
        self.linear4 = nn.Linear(2*k, k)
        self.linear5 = nn.Linear(k , k)
        # self.linear6 = nn.Linear(2*k, k)

    def forward(self, x, v):
        v1 = F.relu(self.linear1(v[:,0:self.v_dim]))
        x = F.relu(self.linear2(x))
        v2 = F.relu(self.linear3(v[:,self.v_dim:]))
        merge = torch.cat([v1,v2], dim=1)
        rv = F.relu(self.linear4(merge))
        rx = F.relu(self.linear5(x))

        return rx, rv


class XXZEnv(gym.Env):
    def __init__(self, target_param_id=None, agent_param_id=None,
                 maximum_counter=50,
                 r_dim=24,
                 n_views=50, action_step=0.05):
        self.num_bits = 3
        self.num_qubits = 50
        self.action_step = action_step
        self.reward_accum = 0


        self.target_param_id = target_param_id
        self.inital_param_id = agent_param_id

        self.n_views = n_views

        self.maximum_counter = maximum_counter
        self.counter = 1

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(r_dim,), dtype=np.float32)

        # three actions for J and three actions for P.
        self.action_space = spaces.Discrete(8)
        self._action_to_direction = {
            0: np.array([1, 0], dtype=np.int32) * 3,
            1: np.array([-1, 0], dtype=np.int32) * 3,
            # 2: np.array([0, 0], dtype=np.int32) * 3,
            2: np.array([0, 1], dtype=np.int32) * 3,
            3: np.array([0, -1], dtype=np.int32) * 3,
            4: np.array([1, 1], dtype=np.int32) * 3,
            5: np.array([1, -1], dtype=np.int32) * 3,
            6: np.array([-1, -1], dtype=np.int32) * 3,
            7: np.array([-1, 1], dtype=np.int32) * 3
        }

        self.model = Backbone(x_dim=2**self.num_bits, v_dim=2*4**self.num_bits, r_dim=r_dim)
        try:
            import os
            current_directory = os.path.dirname(os.path.abspath(__file__))
            self.model.load_state_dict(torch.load(
                current_directory + '/models/heisenberg_rep_model_num_qubits50num_measure_qubits3_r_dim24_num_known_mbases200model2_RMI1_RMI2', map_location='cpu'))
            self.model.to(device)
            self.model.eval()
            print("Total number of param in Model is ", sum(x.numel() for x in self.model.parameters()))
        except:
            print("No load")


        num_observables = 3 ** self.num_bits * (self.num_qubits - 2)
        self.observables = []
        for j in range(0, 3 ** self.num_bits):
            observable = np.load(
                'Heisenberg/float_observable' + str(self.num_bits) + str(j) + '.npy')
            self.observables.append(observable)

        index_observables = []
        combination_list = np.load(
            'Heisenberg/' + str(self.num_qubits) + 'qubit_' + str(self.num_bits) + 'combination_list.npy')

        for j in range(0, len(combination_list)):
            for i in range(0, 3 ** self.num_bits):
                tmp = np.concatenate((self.observables[i], np.array(combination_list[j])))
                index_observables.append(tmp)
        self.observables = np.array(index_observables)
        self.observables = self.observables.reshape((1, num_observables, -1))

        # for j in range(0, len(combination_list)):
        #     for i in range(0, 3 ** self.num_bits):
        #         tmp = np.concatenate((self.observables[i], np.array([combination_list[j][0]])))
        #         index_observables.append(tmp)
        #
        # self.observables = np.array(index_observables)
        # self.observables = self.observables.reshape((1, num_observables, -1))

        self.J_ps = np.linspace(0, 3, 64)
        self.deltas = np.linspace(0, 4, 64)

        # self.target_observation, self.target_state = self._get_target_obs()

    def calculate_fidelity(self, A, B):
        assert (len(A) == len(B))
        Al_temp = ncon((A[0], B[0]), ((-1, 1, -3), (-2, 1, -4)))
        for i in range(1, len(A)):
            Al_temp = ncon((Al_temp, A[i], B[i]), ((-1, -2, 1, 2), (1, 3, -3), (2, 3, -4)))
        return np.abs(np.squeeze(Al_temp).real)

    def _get_target_obs(self, basis_index):
        """
        generate the neural representation of target state
        :return: the neural representation of target state
        """
        v = torch.Tensor(self.observables).to(device)
        iJ = int(self.target_param_id[0])
        idelta = int(self.target_param_id[1])
        # print(iJ, idelta)
        target_file_name = ('Heisenberg/rand' + str(self.num_qubits) +'qubits_Jp'
                            + str(self.J_ps[iJ]) + '_delta' + str(self.deltas[idelta]))
        with open(target_file_name, "rb") as fp:
            target_tn_state = pickle.load(fp)

        # 问题在于，target的id不在441个当中。从而load错误。
        value = np.load(
                    'Heisenberg/prob3_' + str(self.num_qubits) + 'qubits_Jp'
                    + str(self.J_ps[iJ]) + '_delta' + str(self.deltas[idelta]) + '.npy')
        probs = value.reshape((1, -1, 2 ** self.num_bits))
        x = torch.Tensor(probs).to(device)

        x = x.float()
        v = v.float()
        # length of observables 48*(3**3)
        batch_size, m, *_ = v.size()
        context_x, context_v = x[:, basis_index], v[:, basis_index]
        b, m, *x_dims = context_x.shape
        _, _, *v_dims = context_v.shape
        context_x = context_x.view((-1, *x_dims))
        context_v = context_v.view((-1, *v_dims))

        with torch.no_grad():
            rx, rv = self.model(context_x, context_v)
            rx = rx.view((b, m, -1))
            rv = rv.view((b, m, -1))
            r = rx + rv
            neural_rep_target = torch.sum(r, dim=1)

        return neural_rep_target.squeeze().cpu().numpy(), target_tn_state

    def _get_agent_obs(self):
        """
        both the target and agent observation should output
        :return: returns the observation of the quantum ground state of the Hamiltonian
        """
        v = torch.Tensor(self.observables).to(device)
        iJ_agent = self.agent_param_id[0]
        idelta_agent = self.agent_param_id[1]

        agent_file_name = 'Heisenberg/rand' + str(50) + 'qubits_Jp' + str(self.J_ps[iJ_agent]) + '_delta' + str(
            self.deltas[idelta_agent])

        with open(agent_file_name, "rb") as fp:
            agent_tn_state = pickle.load(fp)

        # print(iJ_agent, idelta_agent)
        probs = np.load('Heisenberg/prob3_' + str(self.num_qubits) + 'qubits_Jp' +
                    str(self.J_ps[iJ_agent]) + '_delta' + str(self.deltas[idelta_agent]) + '.npy')

        probs = probs.reshape((1, -1, 2**self.num_bits))
        x = torch.FloatTensor(probs).to(device)
        
        # each time to randomly choose the views to enhance the robustness
        batch_size, m, *_ = v.size()
        indices = list(range(0, m))
        # the first 27 basis is important for representation
        representation_idx = indices[:27]
        rest_indices = list(range(27, m))
        np.random.shuffle(rest_indices)

        representation_idx = representation_idx + rest_indices[:self.n_views]
        context_x, context_v = x[:, representation_idx], v[:, representation_idx]

        b, m, *x_dims = context_x.shape
        _, _, *v_dims = context_v.shape
        context_x = context_x.view((-1, *x_dims))
        context_v = context_v.view((-1, *v_dims))

        with torch.no_grad():
            rx, rv = self.model(context_x, context_v)
            rx = rx.view((b, m, -1))
            rv = rv.view((b, m, -1))
            r = rx + rv
            neural_rep_agent = torch.sum(r, dim=1)

        return neural_rep_agent.squeeze().cpu().numpy(), agent_tn_state, representation_idx

    def _get_info(self, agent_state, target_state):
        """
        :return: returns the distance of the target and agent hamiltonian cofficients
        """
        quantum_fidelity = self.calculate_fidelity(agent_state, target_state)

        param_distance = np.sum(np.abs(self.agent_param_id - self.target_param_id))

        return {"param_J": self.agent_param_id[0], "param_delta": self.agent_param_id[1],
                "param_distance": param_distance,
                "quantum_fidelity": quantum_fidelity}

    def reset(self, seed=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        np.random.seed(seed)
        # ids = np.arange(1, 64, 3)
        small_change = np.random.randint(-1, 2, 2)
        small_change_iJ = small_change[0] * 3
        small_change_id = small_change[1] * 3
        self.agent_param_id = self.inital_param_id + np.array([small_change_iJ, small_change_id], dtype=np.int32)
        self.agent_param_id = np.clip(self.agent_param_id, 1, 61)
        agent_observation, agent_state, _ = self._get_agent_obs()
        return agent_observation

    def step(self, action):
        
        action = int(action)
        direction = self._action_to_direction[action]
        # update the parameter once for each local coefficient J_i
        self.agent_param_id = np.clip(self.agent_param_id + direction, 1, 61)
        agent_observation, agent_state, basis_index = self._get_agent_obs()

        target_observation, target_state = self._get_target_obs(basis_index)
        normalized_euclidean_distance = (np.linalg.norm(agent_observation - target_observation)
                                        )
        # reward = -1 * normalized_euclidean_distance
        reward = -1 * normalized_euclidean_distance * 0.1

        info = self._get_info(agent_state, target_state)
        if reward > -1e-5:
            done = 1
        else:
            done = 0
        return agent_observation, reward, done, info

def test_env():
    from random import choice
    J_ps = np.linspace(0, 3, 64)
    deltas = np.linspace(0, 4, 64)
    ids = np.arange(1, 64, 3)
    iJ = choice(ids)
    id = choice(ids)
    # print(ids[8], ids[20])
   
    env = XXZEnv(agent_param_id=np.array([iJ, id], dtype=np.int32), target_param_id=np.array([ids[8], ids[20]]))
    observation = env.reset()
    agent_observation, reward, done, info = env.step(3)
    print(observation.shape)
    # print(agent_observation)
    print(info["quantum_fidelity"])
    print(info["param_distance"])
    print(reward)

# test_env()
