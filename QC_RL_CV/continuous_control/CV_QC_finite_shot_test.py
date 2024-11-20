import numpy as np
import torch
import argparse
import torch.nn as nn
from distutils.util import strtobool
import os
import random
import gym
# import pandas as pd
from QENV_CV_alpha_01_decay_finite_shot import QCVENV
from main_CV_alpha_01_decay_case1 import Agent
##############################################

# define the RL env
random.seed(42)
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
    parser.add_argument("--sampling-var", type=float, default=1.0,
                        help="the finite shot noise variance")
    parser.add_argument("--seed", type=int, default=42,
        help="seed of the experiment")
    parser.add_argument("--agent-case", type=int, default=1,
                        help="agent case for the experiment!")
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

for k in args.__dict__:
    print(k + ": " + str(args.__dict__[k]), flush=True)

np.random.seed(args.seed)
run_name = f"{args.exp_name}_seed{args.seed}_agentcase{args.agent_case}_as{args.action_step}_nv{args.n_views}_fs{args.sampling_var}"

if args.agent_case == 1:
    # random
    initial_param = np.array([0, 0])
    target_param = np.array([-0.50183952, 1.80285723])
elif args.agent_case == 2:
    initial_param = np.array([0.5, -0.34])
    target_param = np.array([-0.50183952, 1.80285723])
elif args.agent_case == 3:
    initial_param = np.array([0.95, 1.1])
    target_param = np.array([-0.50183952, 1.80285723])
elif args.agent_case == 4:
    initial_param = np.array([1.5, -1.5])
    target_param = np.array([-0.50183952, 1.80285723])
else:
    raise ValueError("Not supported case!")

print("initial_param:", initial_param, flush=True)
print("target_param:", target_param, flush=True)

envs = gym.vector.SyncVectorEnv(
        [lambda: QCVENV(target_param=target_param, initial_param=initial_param,
                 action_step=args.action_step, sampling_var=args.sampling_var,
                 n_views=args.n_views)] * args.num_envs
    )
envs.action_space.seed(42)
envs.observation_space.seed(42)

agent = Agent(envs)

model_name = "main_CV_alpha_01_decay_case1_finite_shot_seed42_agent_case{}_nv3_as0.3_fs{}".format(args.agent_case,
                                                                                                  args.sampling_var)


agent.load_state_dict(torch.load('./models/' + model_name, map_location='cpu'))
agent.to(device)
agent.eval()

# ALGO Logic: Storage setup
obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)


# TRY NOT TO MODIFY: start the game
global_step = 0
num_repetitions = 500

actions_save = torch.zeros((num_repetitions, 512, 2)).to(device)
rewards_save = torch.zeros((num_repetitions, 512)).to(device)
fidelity_save = torch.zeros((num_repetitions, 512)).to(device)
distance_save = torch.zeros((num_repetitions, 512)).to(device)
observation_save = torch.zeros((num_repetitions, 512, 24)).to(device)

for k in range(num_repetitions):
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    # next_done = torch.zeros(args.num_envs).to(device)
    for step in range(0, 50):  # policy roleout 128
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

        if global_step % 500 == 0:
            print("global step: {}, fidelity: {}, reward: {}".format(global_step,
                                                                     info["quantum_fidelity"][0],
                                                                     reward[0]), flush=True)


# save actions, info, rewards
np.save('./exp/'+run_name+'_obs_fixed_each_epi.npy', observation_save.cpu().numpy())
np.save('./exp/'+run_name+'_action_fixed_each_epi.npy', actions_save.cpu().numpy())
np.save('./exp/'+run_name+'_reward_fixed_each_epi.npy', rewards_save.cpu().numpy())
np.save('./exp/'+run_name+'_fidelity_fixed_each_epi.npy', fidelity_save.cpu().numpy())
np.save('./exp/'+run_name+'_distance_fixed_each_epi.npy', distance_save.cpu().numpy())



