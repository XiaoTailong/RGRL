#!/usr/bin/python
import random

import numpy as np

from abc import ABC
from ncon import ncon
import torch
import gym
from gym import spaces
import pygame
from gqn import GenerativeQueryNetwork
import pickle
import scipy.sparse as sparse
import scipy.sparse.linalg as arp
import warnings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class XXZEnv(gym.Env):
    def __init__(self, target_param_id=None, agent_param_id=None,
                 maximum_counter=50,
                 r_dim=64, h_dim=96, z_dim=64, L=2,
                 n_views=500, action_step=0.05):
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
                                            shape=(z_dim,), dtype=np.float32)

        # three actions for J and three actions for P.
        self.action_space = spaces.Discrete(3 * 2)
        self._action_to_direction = {
            0: np.array([1, 0], dtype=np.int32) * 3,
            1: np.array([-1, 0], dtype=np.int32) * 3,
            2: np.array([0, 0], dtype=np.int32) * 3,
            3: np.array([1, 1], dtype=np.int32) * 3,
            4: np.array([1, -1], dtype=np.int32) * 3,
            5: np.array([1, 0], dtype=np.int32) * 3,
        }

        self.model = GenerativeQueryNetwork(x_dim=2 ** self.num_bits,
                                            v_dim=4 ** self.num_bits * 2 + 1,
                                            r_dim=r_dim,
                                            h_dim=h_dim,
                                            z_dim=z_dim, L=L)
        try:
            import os
            current_directory = os.path.dirname(os.path.abspath(__file__))
            self.model.load_state_dict(torch.load(
                current_directory + '/models/XXZ_50qubit_partial3_'+str(r_dim)
                +'_'+str(h_dim)+'_'+str(z_dim)+'_softmax_cpu', map_location='cpu'))
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
                tmp = np.concatenate((self.observables[i], np.array([combination_list[j][0]])))
                index_observables.append(tmp)

        self.observables = np.array(index_observables)
        self.observables = self.observables.reshape((1, num_observables, -1))

        self.J_ps = np.linspace(0, 3, 64)
        self.deltas = np.linspace(0, 4, 64)

        self.target_observation, self.target_state = self._get_target_obs()

    def calculate_fidelity(self, A, B):
        assert (len(A) == len(B))
        Al_temp = ncon((A[0], B[0]), ((-1, 1, -3), (-2, 1, -4)))
        for i in range(1, len(A)):
            Al_temp = ncon((Al_temp, A[i], B[i]), ((-1, -2, 1, 2), (1, 3, -3), (2, 3, -4)))
        return np.abs(np.squeeze(Al_temp).real)

    def _get_target_obs(self):
        """
        generate the neural representation of target state
        :return: the neural representation of target state
        """
        v = torch.Tensor(self.observables).to(device)

        iJ = int(self.target_param_id[0])
        idelta = int(self.target_param_id[1])
        

        # print(iJ, idelta)
        target_file_name = 'Heisenberg/rand' + str(self.num_qubits) +'qubits_Jp' + str(self.J_ps[iJ]) + '_delta' + str(self.deltas[idelta])

        with open(target_file_name, "rb") as fp:
            target_tn_state = pickle.load(fp)

        # 问题在于，target的id不在441个当中。从而load错误。
        value = np.load(
                    'Heisenberg/prob3_' + str(self.num_qubits) + 'qubits_Jp' + str(self.J_ps[iJ]) + '_delta' + str(self.deltas[idelta]) + '.npy')
        probs = value.reshape((1, -1, 2 ** self.num_bits))
        x = torch.Tensor(probs).to(device)

        x = x.float()
        v = v.float()
        # length of observables 48*(3**3)
        
        b, m, *x_dims = x.shape
        _, _, *v_dims = v.shape
        x = x.view((-1, *x_dims))
        v = v.view((-1, *v_dims))

        with torch.no_grad():
            phi = self.model.representation(x, v)

            _, *phi_dims = phi.shape
            phi = phi.view((b, m, *phi_dims))
            # sum over n_views to obtain representations
            neural_rep_target = torch.mean(phi, dim=1)

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
        np.random.shuffle(indices)

        representation_idx = indices[:self.n_views]
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

        return neural_rep_agent.squeeze().cpu().numpy(), agent_tn_state

    def _get_info(self, agent_state, target_state):
        """
        :return: returns the distance of the target and agent hamiltonian cofficients
        """
        quantum_fidelity = self.calculate_fidelity(agent_state, target_state)

        param_distance = np.sum(np.abs(self.agent_param_id - self.target_param_id))

        return {"param_distance": param_distance,
                "quantum_fidelity": quantum_fidelity}

    def reset(self, seed=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        np.random.seed(seed)
        # ids = np.arange(1, 64, 3)
        small_change = np.random.randint(-2, 3, 2)
        small_change_iJ = small_change[0] * 3
        small_change_id = small_change[1] * 3
        self.agent_param_id = self.inital_param_id + np.array([small_change_iJ, small_change_id], dtype=np.int32)
        self.agent_param_id = np.clip(self.agent_param_id, 1, 61)
        agent_observation, agent_state = self._get_agent_obs()

        # self.counter = 1
        # info = self._get_info(agent_state, self.target_state)
        return agent_observation

    def step(self, action):
        
        action = int(action)
        direction = self._action_to_direction[action]
        # update the parameter once for each local coefficient J_i
        self.agent_param_id = np.clip(self.agent_param_id + direction, 1, 61)
        agent_observation, agent_state = self._get_agent_obs()
        normalized_euclidean_distance = np.linalg.norm(agent_observation - self.target_observation) / np.sqrt(agent_observation.shape[0])
        
        reward = -1 * normalized_euclidean_distance * 10

        info = self._get_info(agent_state, self.target_state)
        if reward > -1e-5:
            done = 1
        else:
            done = 0

        # self.counter = 1 if done else self.counter + 1

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

    # print(agent_observation)
    print(info["quantum_fidelity"])
    print(info["param_distance"])
    print(reward)

# test_env()
