from main_cgate_discrete import Agent
from QENV_cgate_discrete_fix_epi import QCVENV

import argparse
import os
import random
import time
from distutils.util import strtobool
# from torch.distributions.normal import Normal
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from torch.utils.tensorboard import SummaryWriter

# define the RL env
random.seed(36)

torch.manual_seed(36)

torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="QCV--v0",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--action-step", type=float, default=0.01,
                        help="the learning rate of the optimizer")
    parser.add_argument("--agent-random", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="random agent or not!")
    parser.add_argument("--target-case", type=int, default=1, help="the case of target CV state")
    parser.add_argument("--n-views", type=int, default=20,
                        help="the number of views for measuring")
    parser.add_argument("--seed", type=int, default=42,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=50000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


args = parse_args()

for k in args.__dict__:
    print(k + ": " + str(args.__dict__[k]), flush=True)

np.random.seed(args.seed)
run_name = f"{args.exp_name}_seed{args.seed}_as{args.action_step}_nv{args.n_views}_90"


initial_param = np.array([2.856542, 0.8995853])
print("initial_param:", initial_param, flush=True)

target_param = np.array([2.18552157, 3.78005358])
print("target_param:", target_param, flush=True)

envs = gym.vector.SyncVectorEnv(
        [lambda: QCVENV(target_param=target_param, initial_param=initial_param,
                 action_step=args.action_step,
                 n_views=args.n_views)] * args.num_envs
    )
envs.action_space.seed(36)
envs.observation_space.seed(36)

agent = Agent(envs)
model_name = "main_cgate_discrete_seed36_target_case1_iniTrue_nv3_as0.03"

agent.load_state_dict(torch.load('./models/' + model_name, map_location='cpu'))
agent.to(device)
agent.eval()

# ALGO Logic: Storage setup
obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)


# TRY NOT TO MODIFY: start the game
global_step = 0
num_repetitions = 200

actions_save = torch.zeros((num_repetitions, 512, 2)).to(device)
rewards_save = torch.zeros((num_repetitions, 512)).to(device)
fidelity_save = torch.zeros((num_repetitions, 512)).to(device)
distance_save = torch.zeros((num_repetitions, 512)).to(device)
observation_save = torch.zeros((num_repetitions, 512, 32)).to(device)

for k in range(num_repetitions):

    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    # next_done = torch.zeros(args.num_envs).to(device)
    for step in range(0, 512):  # policy roleout 128
        global_step += 1
        observation_save[k, step, :] = next_obs[0]
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(next_obs)
        actions_save[k, step, ] = action[0]
        next_obs, reward, done, _, info = envs.step(action.cpu().numpy())
        rewards_save[k, step] = torch.tensor(reward).to(device).view(-1)
        fidelity_save[k, step] = torch.tensor(info["quantum_fidelity"][0]).to(device).view(-1)
        distance_save[k, step] = torch.tensor(info["param_distance"][0]).to(device).view(-1)

        next_obs = torch.Tensor(next_obs).to(device)

        if global_step % 512 == 0:
            print("global step: {}, fidelity: {}, reward: {}".format(global_step,
                                                                     info["quantum_fidelity"][0],
                                                                     reward[0]), flush=True)


# save actions, info, rewards
np.save('./exp/'+run_name+'_obs_fix_epi.npy', observation_save.cpu().numpy())
np.save('./exp/'+run_name+'_action_fix_epi.npy', actions_save.cpu().numpy())
np.save('./exp/'+run_name+'_reward_fix_epi.npy', rewards_save.cpu().numpy())
np.save('./exp/'+run_name+'_fidelity_fix_epi.npy', fidelity_save.cpu().numpy())
np.save('./exp/'+run_name+'_distance_fix_epi.npy', distance_save.cpu().numpy())



