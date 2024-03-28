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
import re
import time
import shortuuid
from flatten_dict import flatten
from flatten_dict import unflatten
import torch
from sklearn.decomposition import PCA

from acme import specs
from acme import wrappers

from auxrl.environments.GridWorld import Env as Env
from auxrl.utils import run_train_episode, run_eval_episode
from model_parameters.gridworld import *

import torch
#torch.cuda.is_available = lambda : False

## Arguments
internal_dim = 10 #int(sys.argv[1])
generic_exp_name = 'iqn' #str(sys.argv[2]) #'gridworld8x8'
network_yaml = 'iqn' #str(sys.argv[3]) #'dm'
source_episode = 350 #int(sys.argv[4]) #250
selected_fnames = None
random_net = False

if 'iqn' in generic_exp_name:
    from auxrl.IQNAgent import Agent
    from auxrl.networks.IQNNetwork import Network
else:
    from auxrl.Agent import Agent
    from auxrl.networks.Network import Network

# Set up paths
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if 'SLURM_JOBID' in os.environ.keys():
    engram_dir = '/mnt/smb/locker/aronov-locker/Ching/rl/' # Axon Path
else:
    engram_dir = '/home/cf2794/engram/Ching/rl/' # Cortex Path
exp_name = f'{generic_exp_name}_{network_yaml}_dim{internal_dim}'
latents_dir = f'{engram_dir}latents/{exp_name}/'
nnets_dir = f'{engram_dir}nnets/{exp_name}/'
pickle_dir = f'{engram_dir}pickles/{exp_name}/'
analysis_dir = f'{engram_dir}analysis/{exp_name}/'
os.makedirs(analysis_dir, exist_ok=True)

quantile_dict = {
    'model': [],
    'iteration': [],
    'x': [],
    'y': [],
    'distance from goal': [],
    'quadrant': [],
    'goal state': [],
    'quantile': [],
    'action': [],
    'quantile val': []
    }

# Helper functions
def manhattan_dist(xy, xy2):
    x, y = xy
    x2, y2 = xy2
    return abs(x-x2) + abs(y-y2)

# Iterate through models
for model_name in os.listdir(nnets_dir):
    string_split = model_name.rfind('_')
    fname = model_name[:string_split]
    if (selected_fnames != None) and (fname not in selected_fnames): continue
    if not fname.startswith(generic_exp_name): continue
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
    network = Network(
        env_spec, device=device, random_quantiles=False, **parameters['network_args'])
    agent = Agent(env_spec, network, device=device, **parameters['agent_args'])
    with open(f'{model_nnet_dir}goal.txt', 'r') as goalfile: # Setting of goal
        goal_state = str(goalfile.read())
        goal_state = [int(goal_state[1]), int(goal_state[-2])]
    agent.load_network(
        model_nnet_dir, source_episode, False, shuffle=random_net)
    shuffle_obs = env._shuffle_obs
    if shuffle_obs:
        with open(f'{model_nnet_dir}shuffle_indices.txt', 'r') as f:
            shuffle_indices = f.read()
        shuffle_indices = re.split('\[|\]|\s|\n', shuffle_indices)
        shuffle_indices = [int(i) for i in shuffle_indices if i != '']
        shuffle_indices = np.array(shuffle_indices)
        env._shuffle_indices = shuffle_indices

    # Get latents
    all_possib_inp = [] 
    quadrant = [] # which quadrant
    x = []
    y = []
    dist_from_goal = []
    maze_width = env._layout_dims[0]
    maze_height = env._layout_dims[1]
    for _x in range(maze_width):
        for _y in range(maze_height):
            if env._layout[_x, _y] != -1:
                env._start_state = env._state = (_x, _y)
                obs = env.get_obs()
                all_possib_inp.append(obs)
                _quadrant = 0 if _x < maze_width//2 else 2
                _quadrant += (0 if _y < maze_height//2 else 1)
                quadrant.append(_quadrant)
                x.append(_x)
                y.append(_y)
                dist_from_goal.append(manhattan_dist((_x, _y), goal_state))
    with torch.no_grad():
        all_possib_inp = np.array(all_possib_inp)
        latents = agent._network.encoder(
            torch.tensor(all_possib_inp).float().to(device),
            save_conv_activity=True)
        conv_activity = agent._network.encoder._prev_conv_activity.cpu().numpy()
    n_states = latents.shape[0]
    quantile_vals, quantiles = agent._network.Q(latents)
    n_states, n_quantiles, n_actions = quantile_vals.shape
    quantiles = quantiles[0,:,0] # Should be the same for each state
    latents = latents.cpu().numpy() # (states, latent_dim)
    for a in range(n_actions):
        for q_idx in range(quantiles.shape[0]):
            quantile = quantiles[q_idx].item()
            _vals = quantile_vals[:, q_idx, a]
            quantile_dict['model'].extend([fname]*n_states)
            quantile_dict['iteration'].extend([iteration]*n_states)
            quantile_dict['x'].extend(x)
            quantile_dict['y'].extend(y)
            quantile_dict['distance from goal'].extend(dist_from_goal)
            quantile_dict['quadrant'].extend(quadrant)
            quantile_dict['goal state'].extend([goal_state]*n_states)
            quantile_dict['quantile'].extend([quantile]*n_states)
            quantile_dict['action'].extend([a]*n_states)
            quantile_dict['quantile val'].extend(_vals.detach().numpy().tolist())

quantile_df = pd.DataFrame(quantile_dict)
suffix = 'randomnet_' if random_net else ''
with open(f'{analysis_dir}{suffix}quantile_df_ep{source_episode}.p', 'wb') as f:
    pickle.dump(quantile_df, f)
