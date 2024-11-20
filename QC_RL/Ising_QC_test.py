import numpy as np
import torch
import argparse
import torch.nn as nn
from distutils.util import strtobool
import os
import random
import gym
# import pandas as pd
from QENV_small_range_nodone_rfixed_multidiscrete_fixed_epi import QIsingEnv
from main_small_range_nodone_case_rfixed_multidiscrete import Agent
##############################################

# define the RL env
random.seed(1317)

torch.manual_seed(42)

torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--action-step", type=float, default=0.05,
                        help="the learning rate of the optimizer")
    parser.add_argument("--n-views", type=int, default=20,
                        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=42,
        help="seed of the experiment")
    parser.add_argument("--agent-random", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="random agent or not!")
    parser.add_argument("--total-timesteps", type=int, default=100000,
        help="total timesteps of the experiments")
    parser.add_argument("--maximum-counter", type=int, default=50,
                        help="maximum length of a enviroment")
    parser.add_argument("--target-index", type=int, default=3,
                        help="the target parameters index")
    parser.add_argument("--target-case", type=int, default=1,
                        help="case of target parameters")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=512,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="the number of mini-batches")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


args = parse_args()

np.random.seed(args.seed)

run_name = f"{args.exp_name}_seed{args.seed}_target{args.target_case}_ini{args.agent_random}_as{args.action_step}_nv{args.n_views}"

if args.target_case == 1:
    target_param = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
elif args.target_case == 2:
    target_param = np.array([0.8, -0.8, 0.8, -0.8, 0.8, -0.8])
elif args.target_case == 3:
    target_param = np.array([0.8, 0.8, 0.8, -0.8, -0.8, -0.8])
elif args.target_case == 4:
    target_param = np.load("./models/new_model/target_Js/Ising_6qubit_random_targetJ_{}.npy"
                           .format(args.target_index))
else:
    raise ValueError("Not supported case!")

print("target param:", target_param, flush=True)

if args.agent_random:
    if args.target_case == 1:
        agent_param = np.array([-0.25091976, 0.90142861, 0.46398788, 0.19731697, -0.68796272, -0.68801096])
    elif args.target_case == 2:
        agent_param = np.array([-0.25091976, 0.90142861, 0.46398788, 0.19731697, -0.68796272, -0.68801096])
    elif args.target_case == 3:
        agent_param = np.array([-0.88383278, 0.73235229, 0.20223002, 0.41614516, -0.95883101, 0.9398197])
    elif args.target_case == 4:
        agent_param = np.array([-0.88383278, 0.73235229, 0.20223002, 0.41614516, -0.95883101, 0.9398197])
else:
    agent_param = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)

print("inital param:", agent_param, flush=True)

envs = gym.vector.SyncVectorEnv(
    [lambda: QIsingEnv(target_param=target_param, agent_param=agent_param,
                       maximum_counter=args.maximum_counter,
                       n_views=args.n_views,
                       action_step=args.action_step)] * 1)
envs.action_space.seed(42)
envs.observation_space.seed(42)

agent = Agent(envs)
model_name = f"main_small_range_nodone_case_rfixed_multidiscrete_01_seed1001_target{args.target_case}_ini{args.agent_random}_as{args.action_step}_nv{args.n_views}"

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

actions_save = torch.zeros((num_repetitions, 512, 6)).to(device)
rewards_save = torch.zeros((num_repetitions, 512)).to(device)
fidelity_save = torch.zeros((num_repetitions, 512)).to(device)
distance_save = torch.zeros((num_repetitions, 512)).to(device)
observation_save = torch.zeros((num_repetitions, 512, 32)).to(device)

for k in range(num_repetitions):
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    # next_done = torch.zeros(args.num_envs).to(device)
    for step in range(0, 100):  # policy roleout 128
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


# save actions, info, rewards
np.save('./exp/'+run_name+'_obs', observation_save.cpu().numpy())
np.save('./exp/'+run_name+'_action', actions_save.cpu().numpy())
np.save('./exp/'+run_name+'_reward', rewards_save.cpu().numpy())
np.save('./exp/'+run_name+'_fidelity', fidelity_save.cpu().numpy())
np.save('./exp/'+run_name+'_distance', distance_save.cpu().numpy())



