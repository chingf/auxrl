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
from flatten_dict import flatten
from flatten_dict import unflatten
import torch
from sklearn.decomposition import PCA

from acme import specs
from acme import wrappers

from auxrl.Agent import Agent
from auxrl.networks.Network import Network
from auxrl.environments.GridWorld import Env as Env
from auxrl.utils import run_train_episode, run_eval_episode
from model_parameters.gridworld import selected_models

import torch
#torch.cuda.is_available = lambda : False

## Arguments
internal_dim = 8
generic_exp_name = 'gridworld8x8'
network_yaml = 'dm'
source_episode = 250
selected_fnames, _, _ = selected_models(include_pos_sample_only=True)
selected_fnames = [f'{generic_exp_name}_{f}' for f in selected_fnames]
random_net = False

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

## Helper functions
def get_gini(var_ratio):
    total = 0
    for i, xi in enumerate(var_ratio[:-1], 1):
        total += np.sum(np.abs(xi - var_ratio[i:]))
    return total / (len(var_ratio)**2 * np.mean(var_ratio))

def get_auc(var_ratio):
    var_curve = np.cumsum(var_ratio)
    var_auc = np.trapz(var_curve, dx=1/var_curve.size)
    return var_auc

def get_entro(var_ratio):
    var_entro = -np.sum(var_ratio * np.log(var_ratio)) / np.log(2)
    return var_entro

def get_n_components(var_ratio):
    var_curve = np.cumsum(var_ratio)
    return np.argwhere(var_curve > 0.9)[0].item()

# Collecting dimensionality measures
dim_dict = {
    'model': [],
    'iteration': [],
    'gini': [],
    'auc': [],
    'entro': [],
    'n_components': []
    }

# Collecting low-dim representations
repr_dict = {
    'model': [],
    'iteration': [],
    'latents': [],
    'x': [],
    'y': [],
    'quadrant': [],
    'goal_state': [],
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
    network = Network(env_spec, device=device, **parameters['network_args'])
    agent = Agent(env_spec, network, device=device, **parameters['agent_args'])
    with open(f'{model_nnet_dir}goal.txt', 'r') as goalfile: # Setting of goal
        goal_state = str(goalfile.read())
        goal_state = [int(goal_state[1]), int(goal_state[-2])]
    agent.load_network(
        model_nnet_dir, source_episode, False, shuffle=random_net)

    # Get latents
    all_possib_inp = [] 
    quadrant = [] # which quadrant
    x = []
    y = []
    maze_width = env._layout_dims[0]
    maze_height = env._layout_dims[1]
    for _x in range(maze_width):
        for _y in range(maze_height):
            if env._layout[_x, _y] != -1:
                env._start_state = env._state = (_x, _y)
                all_possib_inp.append(env.get_obs())
                _quadrant = 0 if _x < maze_width//2 else 2
                _quadrant += (0 if _y < maze_height//2 else 1)
                quadrant.append(_quadrant)
                x.append(_x)
                y.append(_y)
    with torch.no_grad():
        all_possib_inp = np.array(all_possib_inp)
        latents = agent._network.encoder(
            torch.tensor(all_possib_inp).float().to(device)).cpu().numpy()
    n_states = latents.shape[0]
    pca = PCA()
    reduced_latents = pca.fit_transform(latents)
    reduced_latents = reduced_latents[:, :3]
    var_ratio = pca.explained_variance_ratio_

    repr_dict['model'].extend([fname]*n_states)
    repr_dict['iteration'].extend([iteration]*n_states)
    repr_dict['latents'].extend(latents.tolist())
    repr_dict['x'].extend(x)
    repr_dict['y'].extend(y)
    repr_dict['quadrant'].extend(quadrant)
    repr_dict['goal_state'].extend([goal_state]*n_states)
        
    dim_dict['model'].append(fname)
    dim_dict['iteration'].append(iteration)
    dim_dict['gini'].append(get_gini(var_ratio))
    dim_dict['auc'].append(get_auc(var_ratio))
    dim_dict['entro'].append(get_entro(var_ratio))
    dim_dict['n_components'].append(get_n_components(var_ratio))

repr_df = pd.DataFrame(repr_dict)
dim_df = pd.DataFrame(dim_dict)
suffix = 'randomnet_' if randomnet else ''
with open(f'{analysis_dir}{suffix}representation_df.p', 'wb') as f:
    pickle.dump(repr_df, f)
with open(f'{analysis_dir}{suffix}dimensionality_df.p', 'wb') as f:
    pickle.dump(dim_df, f)