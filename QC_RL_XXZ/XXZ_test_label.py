import numpy as np
import torch
import argparse
import torch.nn as nn
from distutils.util import strtobool
import os
import random
import gym
# import pandas as pd
from XXZ_ENV_small_change_model1_rfixed_label import XXZEnv
from main_small_change_model1_rfixed_label import Agent

##############################################



torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--n-views", type=int, default=20,
                        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed of the experiment")
    parser.add_argument("--target-case", type=int, default=1,
                        help="case of target parameters")
    parser.add_argument("--num-steps", type=int, default=512,
                        help="the number of steps to run in each environment per policy rollout")
    args = parser.parse_args()
    # fmt: on
    return args


args = parse_args()

np.random.seed(args.seed)

# define the RL env
random.seed(args.seed)
torch.manual_seed(args.seed)


run_name = f"{args.exp_name}_seed{args.seed}_target{args.target_case}_nv{args.n_views}_action_complete"

ids = np.arange(1, 64, 3)

if args.target_case == 1:
    # 52, 7 这里才是要保持一致的
    initial_agent_id = np.array([ids[8], ids[18]], dtype=np.int32)
    target_param_id = np.array([52, 7], dtype=np.int32)
elif args.target_case == 2:
    target_param_id = np.array([7, 7], dtype=np.int32)
    initial_agent_id = np.array([ids[7], ids[16]], dtype=np.int32)
elif args.target_case == 3:
    target_param_id = np.array([ids[8], ids[18]], dtype=np.int32)
    initial_agent_id = np.array([52, 7], dtype=np.int32)
elif args.target_case == 4:
    # 会比较难学习
    target_param_id = np.array([ids[5], ids[5]], dtype=np.int32)
    initial_agent_id = np.array([ids[19], ids[19]], dtype=np.int32)
elif args.target_case == 5:
    target_param_id = np.array([ids[6], ids[18]], dtype=np.int32)
    initial_agent_id = np.array([ids[1], ids[1]], dtype=np.int32)
elif args.target_case == 6:
    # 会比较难学习
    target_param_id = np.array([ids[17], ids[6]], dtype=np.int32)
    initial_agent_id = np.array([ids[2], ids[20]], dtype=np.int32)
elif args.target_case == 7:
    target_param_id = np.array([ids[12], ids[16]], dtype=np.int32)
    initial_agent_id = np.array([ids[19], ids[4]], dtype=np.int32)
else:
    raise ValueError("Not supported setting!")

print("target_param_id:", target_param_id, flush=True)
print("initial_param_id:", initial_agent_id, flush=True)

env = XXZEnv(target_param_id=target_param_id, agent_param_id=initial_agent_id,
             n_views=args.n_views)

env.action_space.seed(args.seed)
env.observation_space.seed(args.seed)

agent = Agent(env)

model_name = f"main_small_change_model1_rfixed_label_42_{args.target_case}_ns512_nv{args.n_views}_action_complete"

agent.load_state_dict(torch.load('./models/' + model_name, map_location='cpu'))
agent.to(device)
agent.eval()

# ALGO Logic: Storage setup
obs = torch.zeros((args.num_steps, env.observation_space.shape[0])).to(device)
actions = torch.zeros((args.num_steps,) + env.action_space.shape).to(device)
rewards = torch.zeros((args.num_steps,)).to(device)

# TRY NOT TO MODIFY: start the game
global_step = 0
num_repetitions = 500

actions_save = torch.zeros((num_repetitions, 512, 2)).to(device)
rewards_save = torch.zeros((num_repetitions, 512)).to(device)
fidelity_save = torch.zeros((num_repetitions, 512)).to(device)
distance_save = torch.zeros((num_repetitions, 512)).to(device)
observation_save = torch.zeros((num_repetitions, 512, 24)).to(device)

for k in range(num_repetitions):
    next_obs = torch.Tensor(env.reset()).to(device)
    # next_done = torch.zeros(args.num_envs).to(device)
    for step in range(0, 512):  # policy roleout 128
        global_step += 1
        observation_save[k, step, :] = next_obs
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(next_obs)
            # print(action.shape)
        actions_save[k, step,] = action
        next_obs, reward, done, info = env.step(action.cpu().numpy())
        rewards_save[k, step] = torch.tensor(reward).to(device).view(-1)
        fidelity_save[k, step] = torch.tensor(info["quantum_fidelity"]).to(device).view(-1)
        distance_save[k, step] = torch.tensor(info["param_distance"]).to(device).view(-1)

        next_obs = torch.Tensor(next_obs).to(device)

        if global_step % 512 == 0:
            print("global step: {}, fidelity: {}, reward: {}".format(global_step,
                                                                     info["quantum_fidelity"],
                                                                     reward), flush=True)

# save actions, info, rewards
np.save('./exp/' + run_name + '_obs_case', observation_save.cpu().numpy())
np.save('./exp/' + run_name + '_action_case', actions_save.cpu().numpy())
np.save('./exp/' + run_name + '_reward_case', rewards_save.cpu().numpy())
np.save('./exp/' + run_name + '_fidelity_case', fidelity_save.cpu().numpy())
np.save('./exp/' + run_name + '_distance_case', distance_save.cpu().numpy())
