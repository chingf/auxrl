import sys
import logging
import pickle
import yaml
import matplotlib.cm as cm
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import shortuuid
import torch
from sklearn.decomposition import PCA

from acme import specs
from acme import wrappers

from auxrl.Agent import Agent
from auxrl.networks.Network import Network
from auxrl.environments.AlternatingT import Env as Env
from auxrl.utils import run_train_episode, run_eval_episode
from model_parameters.gridworld import *

import torch

## Experiment Parameters
generic_exp_name = sys.argv[1] # altT
network_yaml = sys.argv[2] # dm
internal_dim = int(sys.argv[3]) # 32
source_episode = int(sys.argv[4]) # 60
selected_fnames, _, _ = altT()
selected_fnames = [f'{generic_exp_name}_{f}' for f in selected_fnames]

# Set up paths
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if 'SLURM_JOBID' in os.environ.keys():
    engram_dir = '/mnt/smb/locker/aronov-locker/Ching/rl/' # Axon Path
else:
    engram_dir = '/home/cf2794/engram/Ching/rl/' # Cortex Path
engram_dir = './'
exp_name = f'{generic_exp_name}_{network_yaml}_dim{internal_dim}'
latents_dir = f'{engram_dir}latents/{exp_name}/'
nnets_dir = f'{engram_dir}nnets/{exp_name}/'
pickle_dir = f'{engram_dir}pickles/{exp_name}/'
analysis_dir = f'{engram_dir}analysis/{exp_name}/'
os.makedirs(analysis_dir, exist_ok=True)

# Collecting low-dim representations
repr_dict = {
    'model': [],
    'iteration': [],
    'timestep': [],
    'latents': [],
    'x': [],
    'y': [],
    'last_reward_loc': [],
    'reward_loc': [],
    'final_reward': [],
    'condn_label': [],
    'maze_side': [],
    }

# Iterate through models
for model_name in os.listdir(nnets_dir):
    string_split = model_name.rfind('_')
    fname = model_name[:string_split]
    if (selected_fnames != None) and (fname not in selected_fnames):
        continue
    iteration = int(model_name[string_split+1:])
    model_nnet_dir = f'{nnets_dir}{model_name}/'
    if not os.path.exists(f'{model_nnet_dir}network_ep{source_episode}.pth'):
        continue
    print(f'Processing {model_name}')

    # Initialize necessary objects and load network
    with open(f'{engram_dir}params/{exp_name}/{fname}.yaml', 'r') as f:
        parameters = yaml.safe_load(f)
    parameters['fname'] = f'{exp_name}/{model_name}'
    parameters['internal_dim'] = internal_dim
    env = Env(**parameters['dset_args'])
    env_spec = specs.make_environment_spec(env)
    midwidth = env._width//2
    network = Network(env_spec, device=device, **parameters['network_args'])
    agent = Agent(env_spec, network, device=device, **parameters['agent_args'])
    agent.load_network(model_nnet_dir, source_episode, False)

    # Get latents
    with torch.no_grad():
        timestep = env.reset()
        episode_steps = 0
        episode_return = 0
        n_total_steps = 100

        latents = []; xs = []; ys = []; last_reward_locs = []; reward_locs = [];
        condn_labels = []; maze_side = [];

        while not timestep.last():
            if episode_steps >= n_total_steps: # Stop at some number of steps
                break
            action, latent = agent.select_action(
                timestep.observation, force_greedy=True, return_latent=True)
            x, y = env._state
            latents.append(latent.cpu().numpy())
            xs.append(x)
            ys.append(y)
            last_reward_locs.append(env._last_reward_loc)
            reward_locs.append(env._reward_loc)
            condn_labels.append(env._space_label[x, y])
            maze_side.append(x - midwidth)

            timestep = env.step(action)
            episode_steps += 1
            episode_return += timestep.reward

        repr_dict['model'].extend([fname[len(generic_exp_name)+1:]]*n_total_steps)
        repr_dict['iteration'].extend([iteration]*n_total_steps)
        repr_dict['timestep'].extend(range(n_total_steps))
        repr_dict['latents'].extend(latents)
        repr_dict['x'].extend(xs)
        repr_dict['y'].extend(ys)
        repr_dict['last_reward_loc'].extend(last_reward_locs)
        repr_dict['reward_loc'].extend(reward_locs)
        repr_dict['condn_label'].extend(condn_labels)
        repr_dict['maze_side'].extend(maze_side)
        repr_dict['final_reward'].extend([episode_return]*n_total_steps)

repr_df = pd.DataFrame(repr_dict)
with open(f'{analysis_dir}representation_df_ep{source_episode}.p', 'wb') as f:
    pickle.dump(repr_df, f)
