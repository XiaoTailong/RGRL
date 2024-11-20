from abc import ABC

import torch
import gym
from gym import spaces
import pygame
from cgqn import GenerativeQueryNetwork

import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
from random import choice

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class QCVENV(gym.Env):
    def __init__(self, target_param=None, initial_param=None,
                 r_dim=32, h_dim=96, z_dim=32, L=2,
                 n_views=10):

        num_phi = 100
        num_train_states = 50
        kappa = 1
        var = 1

        self.phis = np.linspace(0, 1, num_phi + 1)[0:num_phi] * np.pi
        observables = np.array(self.phis).reshape((1, num_phi))

        scale = np.sqrt(sf.hbar)
        self.quad_axis = np.linspace(-6, 6, 100) * scale

        if target_param is None:
            # randomly generating the target parameters
            self.target_param = np.array([np.random.random() * 1, np.abs(np.random.random() * 3), np.random.random() * np.pi * 2])
        else:
            self.target_param = np.array(target_param)

        self.init_param = initial_param
        self.n_views = n_views

        self.counter = 1
        self.maximum_counter = 512

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(r_dim,),
                                            dtype=np.float32)

        # d_gate_para = [np.random.random() * var, np.abs(np.random.random() * 3), np.random.random() * np.pi * 2]
        self.action_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([1, 3, 2*np.pi]), dtype=np.float64)

        # continuous control
        # self._action_to_direction = {
        #     0: 1,
        #     1: -1,
        #     2: 0
        # }

        self.model = GenerativeQueryNetwork(x_dim=100,
                                            v_dim=1,
                                            r_dim=r_dim,
                                            h_dim=h_dim,
                                            z_dim=z_dim, L=L)

        try:
            import os
            current_directory = os.path.dirname(os.path.abspath(__file__))
            self.model.load_state_dict(torch.load(current_directory +
                                                  '/models/' + str(r_dim) + '_' + str(h_dim) + '_' + str(
                z_dim) + "_tanh_kerr_thermal_range_0_3_num_phi" + str(num_phi) + "_kappa" + str(
                kappa) + "_var" + str(var) + "_" + str(
                num_train_states) + "trainstates_output_cpu", map_location='cpu'))
            self.model.to(device)
            self.model.eval()
            print("Total number of param in Model is ", sum(x.numel() for x in self.model.parameters()))
        except:
            print("NO Load")

    def calculate_fidelity(self, state1, state2):
        xvec = np.linspace(-15, 15, 401)
        W1 = state1.wigner(mode=0, xvec=xvec, pvec=xvec)
        W2 = state2.wigner(mode=0, xvec=xvec, pvec=xvec)
        return np.sum(W1 * W2 * 30 / 400 * 30 / 400) * 4 * np.pi

    def generate_output_state_probs(self, d_gate_para, phis):
        #
        kappa = 1
        # phis = np.linspace(0, 1, 100 + 1) * np.pi
        prog = sf.Program(1)
        with prog.context as q:
            sf.ops.Thermal(d_gate_para[0])
            sf.ops.Dgate(d_gate_para[1], d_gate_para[2]) | q
            sf.ops.Kgate(kappa) | q
        eng = sf.Engine('fock', backend_options={"cutoff_dim": 10})
        state_return = eng.run(prog).state


        output_state_data = []
        for j in range(0, len(phis)):
            prog = sf.Program(1)
            with prog.context as q:
                sf.ops.Thermal(d_gate_para[0])
                sf.ops.Dgate(d_gate_para[1], d_gate_para[2]) | q
                sf.ops.Kgate(kappa) | q
                sf.ops.Rgate(phis[j]) | q

            eng = sf.Engine('fock', backend_options={"cutoff_dim": 10})
            state = eng.run(prog).state
            scale = np.sqrt(sf.hbar)
            xvec = np.linspace(-6, 6, 100) * scale
            pvec = np.linspace(-6, 6, 100) * scale
            prob = state.x_quad_values(0, xvec, pvec)
            output_state_data.append(prob)

        output_state_data = np.array(output_state_data)

        return state_return, output_state_data


    def _get_target_obs(self, phis_basis):

        v = torch.FloatTensor(self.phis).to(device)
        v = v.unsqueeze(dim=0)
        v = v.unsqueeze(dim=2)
        context_v = v[:, phis_basis]
        chosen_phi = self.phis[phis_basis]
        # cat_prob = []
        # 每次都要根据目标态和给定的观测量得到概率分布
        # for j in range(0, self.n_views):
        #     ideal_probs = self.generate_cat_homodyne_prob(self.target_cat, chosen_phi[j])
        #     cat_prob.append(ideal_probs)
        #     # print(ideal_probs.shape, flush=True)
        target_state, cat_prob = self.generate_output_state_probs(self.target_param, chosen_phi)

        agent_probs = np.expand_dims(np.array(cat_prob), axis=0)
        context_x = torch.FloatTensor(agent_probs).to(device)

        context_x = context_x.float()
        context_v = context_v.float()

        b, m, *x_dims = context_x.shape
        _, _, *v_dims = context_v.shape

        x = context_x.view((-1, *x_dims))
        v = context_v.view((-1, *v_dims))

        with torch.no_grad():
            phi = self.model.representation(x, v)
            _, *phi_dims = phi.shape
            phi = phi.view((b, m, *phi_dims))
            # sum over n_views to obtain representations
            neural_rep_target = torch.mean(phi, dim=1)

        return neural_rep_target.squeeze().cpu().numpy(), target_state

    def _get_agent_obs(self, agent_param):
        """
        随机挑选30个observables
        直接将representation axis传进去，每个episode就不用动了。
        """
        v = torch.FloatTensor(self.phis).to(device)
        v = v.unsqueeze(dim=0)
        v = v.unsqueeze(dim=2)

        batch_size, m, *_ = v.size()
        indices = list(range(0, m))
        np.random.shuffle(indices)
        representation_idx = indices[:self.n_views]

        #
        context_v = v[:, representation_idx]
        chosen_phis = self.phis[representation_idx]
        # 本质上就是改变初始gate参数
        agent_state, agent_prob = self.generate_output_state_probs(agent_param, chosen_phis)

        agent_probs = np.expand_dims(np.array(agent_prob), axis=0)

        context_x = torch.FloatTensor(agent_probs).to(device)

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

        return neural_rep_agent.squeeze().cpu().numpy(), agent_state, representation_idx

    def _get_info(self, agent_state, target_state):
        """
        :return: returns the distance of the target and agent hamiltonian cofficients
        """
        quantum_fidelity = self.calculate_fidelity(agent_state, target_state)
        param_distance1 = np.linalg.norm(self.agent_param - self.target_param, ord=1)
        param_distance2 = np.linalg.norm(-1 * self.agent_param - self.target_param, ord=1)
        if param_distance1 < param_distance2:
            param_distance = param_distance1
        else:
            param_distance = param_distance2

        return {"quantum_fidelity": np.real(quantum_fidelity),
                "param_distance": param_distance}

    def reset(self, seed=None):
        super().reset(seed=seed)

        # indices = list(range(0, 300))
        # idx = choice(indices)
        # self.representation_idx = [idx, (idx + 100) % 300, (idx + 200) % 300]

        agent_small_change = np.random.uniform(-0.1, 0.1, 3)
        # intial param is fixed, and agent small change is varied for each time
        self.agent_param = self.init_param + agent_small_change

        initial_observation, agent_state, representation_idx = self._get_agent_obs(self.agent_param)
        target_observation, target_state = self._get_target_obs(representation_idx)

        self.counter = 1
        info = self._get_info(agent_state, target_state)
        # 每一次reset，都会对representation index进行重新选取

        return initial_observation, info

    def step(self, action):

        # 0.3 + (0.01-0.3)*1/512
        # action_step = (self.action_step +
        #                (0.01 - self.action_step) * self.counter / self.maximum_counter)
        # initial cat for
        # action_change = np.zeros(3)
        # for i, action_element in enumerate(action):
        #     action_element = int(action_element)
        #     direction = self._action_to_direction[action_element]
        #     action_change[i] = direction * action_step
        # self.agent_param = np.clip(self.agent_param + action_change, -2, 2)
        self.agent_param = np.array(action)

        agent_observation, agent_state, representation_idx = self._get_agent_obs(self.agent_param)
        target_observation, target_state = self._get_target_obs(representation_idx)

        normalized_euclidean_distance = (np.linalg.norm(agent_observation
                                                        - target_observation)
                                         / np.sqrt(agent_observation.shape[0]))
        reward = -1 * normalized_euclidean_distance  # data range (-1, 0)

        try:
            info = self._get_info(agent_state, target_state)
        except:
            info["quantum_fidelity"] = np.real(0)

        if reward == 0:
            done = 1
        else:
            done = 0

        truncated = False
        self.counter += 1

        return agent_observation, reward, done, truncated, info


def test_env():
    agent_param = [np.random.random() * 1, np.abs(np.random.random() * 3), np.random.random() * np.pi * 2]
    target_param = [np.random.random() * 1, np.abs(np.random.random() * 3), np.random.random() * np.pi * 2]

    env = QCVENV(target_param=target_param, initial_param=agent_param)
    observation = env.reset()[0],
    agent_observation, reward, done, _, info = env.step([0.5, 0.1, np.pi])

    print(reward)
    print(info["quantum_fidelity"])

# test_env()