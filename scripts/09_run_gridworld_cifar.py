import sys
import logging
import pickle
import yaml
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from flatten_dict import flatten
from flatten_dict import unflatten
import torch

from acme import specs
from acme import wrappers

from auxrl.Agent import Agent
from auxrl.networks.Network import Network
from auxrl.environments.GridWorld import Env as Env, ObservationType
from auxrl.utils import run_train_episode, run_eval_episode
from model_parameters.gridworld import *

# Command-line args
job_idx = int(sys.argv[1])
n_jobs = int(sys.argv[2])
nn_yaml = sys.argv[3]
internal_dim = int(sys.argv[4])
epsilon = float(sys.argv[5]) # 1.0

# Experiment Parameters
load_function = test_full
fname_prefix = f'cifar_gridworld_8x8'

# Less used params
shuffle = False
observation_type = ObservationType.CIFAR
if epsilon < 1.0:
    fname_prefix += f'_eps{epsilon}'
n_iters = 30
if epsilon < 0.4:
    n_episodes = 1501
elif epsilon < 0.6:
    n_episodes = 1201
elif epsilon < 0.8:
    n_episodes = 901
else:
    n_episodes = 601
random_seed = True
size_maze = 8
fname_suffix = ''
n_cpu_jobs = 56
eval_every = 1
save_net_every = 50
continual_transfer = False

# GPU selection
try:
    n_gpus = (len(os.environ['CUDA_VISIBLE_DEVICES'])+1)/2
except:
    n_gpus = 0
if n_gpus > 1:
    device_num = str(job_idx % n_gpus)
    my_env = os.environ
    my_env["CUDA_VISIBLE_DEVICES"] = device_num

# Make directories
if os.environ['USER'] == 'chingfang':
    engram_dir = '/Volumes/aronov-locker/Ching/rl/' # Local Path
elif 'SLURM_JOBID' in os.environ.keys():
    engram_dir = '/mnt/smb/locker/aronov-locker/Ching/rl/' # Axon Path
else:
    engram_dir = '/home/cf2794/engram/Ching/rl/' # Cortex Path
exp_dir = f'{fname_prefix}_{nn_yaml}_dim{internal_dim}{fname_suffix}/'
for d in ['pickles/', 'nnets/', 'figs/', 'params/']:
    os.makedirs(f'{engram_dir}{d}{exp_dir}', exist_ok=True)
pickle_dir = f'{engram_dir}pickles/{exp_dir}/'
param_dir = f'{engram_dir}params/{exp_dir}/'

def gpu_parallel(job_idx):
    for _arg in split_args[job_idx]:
        run(_arg)

def cpu_parallel():
    job_results = Parallel(n_jobs=n_cpu_jobs)(delayed(run)(arg) for arg in args)

def run(arg):
    _fname, loss_weights, param_update, i = arg
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    fname = f'{_fname}_{i}'
    fname_nnet_dir = f'{engram_dir}nnets/{exp_dir}/{fname}/'
    fname_fig_dir = f'{engram_dir}figs/{exp_dir}/{fname}/'
    for _dir in [fname_nnet_dir, fname_fig_dir]:
        os.makedirs(_dir, exist_ok=True)

    saved_epoch = int(save_net_every * (n_episodes//save_net_every))
    net_exists = f'network_ep{saved_epoch}.pth' in os.listdir(fname_nnet_dir)
    if net_exists:
        print(f'Skipping {fname}')
        return
    else:
        print(f'Running {fname}')

    parameters = {
        'fname': fname,
        'n_episodes': n_episodes,
        'n_test_episodes': 10,
        'continual_transfer': continual_transfer,
        'agent_args': {
            'loss_weights': loss_weights, 'lr': 1e-4,
            'replay_capacity': 20_000, 'epsilon': epsilon,
            'batch_size': 64, 'target_update_frequency': 1000,
            'train_seq_len': 1},
        'network_args': {'latent_dim': internal_dim, 'network_yaml': nn_yaml},
        'dset_args': {
            'layout': size_maze, 'observation_type': observation_type}
        }
    parameters = flatten(parameters)
    parameters.update(flatten(param_update))
    parameters = unflatten(parameters)
    with open(f'{param_dir}{_fname}.yaml', 'w') as outfile:
        yaml.dump(parameters, outfile, default_flow_style=False)
    if random_seed: np.random.seed(i)
    env = Env(**parameters['dset_args'])
    env_spec = specs.make_environment_spec(env)
    network = Network(env_spec, device=device, **parameters['network_args'])
    agent = Agent(env_spec, network, device=device, **parameters['agent_args'])

    with open(f'{fname_nnet_dir}goal.txt', 'w') as goalfile:
        goalfile.write(str(env._goal_state))
    with open(f'{fname_nnet_dir}image_indices.txt', 'w') as goalfile:
        goalfile.write(str(env._cifar_images_indices))

    result = {}
    result['episode'] = []
    result['step'] = []
    result['train_loss'] = []
    result['mf_loss'] = []
    result['neg_random_loss'] = []
    result['neg_neighbor_loss'] = []
    result['pos_sample_loss'] = []
    result['train_score'] = []
    result['valid_score'] = []
    result['train_steps_per_ep'] = []
    result['valid_steps_per_ep'] = []
    result['model'] = []
    result['model_iter'] = []

    sec_per_step_SUM = 0.
    sec_per_step_NUM = 0.
    goal_reset_episode = n_episodes//2 + 1

    for episode in range(n_episodes):
        if continual_transfer and (episode == goal_reset_episode):
            env.reset(reset_goal=True)
            agent._replay_buffer.flush()
        start = time.time()
        losses, score, steps_per_episode = run_train_episode(env, agent)
        end = time.time()
        sec_per_step_SUM += (end-start)
        sec_per_step_NUM += steps_per_episode
        result['episode'].append(episode)
        result['step'].append(sec_per_step_NUM)
        result['train_loss'].append(losses[4])
        result['mf_loss'].append(losses[3])
        result['neg_random_loss'].append(losses[2])
        result['neg_neighbor_loss'].append(losses[1])
        result['pos_sample_loss'].append(losses[0])
        result['train_score'].append(score)
        result['train_steps_per_ep'].append(steps_per_episode)
        result['model'].append(_fname)
        result['model_iter'].append(i)
        if episode % eval_every == 0:
            sec_per_step = sec_per_step_SUM/sec_per_step_NUM
            print(f'[TRAIN SUMMARY] {500*sec_per_step} sec/ 500 steps')
            print(f'{sec_per_step_NUM} training steps elapsed.')
            score, steps_per_episode = run_eval_episode(
                env, agent, parameters['n_test_episodes'])
            result['valid_score'].append(score)
            result['valid_steps_per_ep'].append(steps_per_episode)
            # Save plots tracking training progress
            fig, axs = plt.subplots(3, 2, figsize=(7, 10))
            loss_keys = [
                'train_loss', 'mf_loss', 'neg_random_loss',
                'neg_neighbor_loss', 'pos_sample_loss']
            for key_idx, loss_key in enumerate(loss_keys):
                ax = axs[key_idx%3][key_idx//3]
                ax.plot(result[loss_key])
                ax.set_ylabel(loss_key)
            plt.tight_layout()
            plt.savefig(f'{fname_fig_dir}train_losses.png')
            plt.figure()
            plt.plot(result['train_score'])
            plt.ylabel('Training Score'); plt.xlabel('Training Episodes')
            plt.tight_layout()
            plt.savefig(f'{fname_fig_dir}train_scores.png')
            plt.figure()
            plt.plot(
                result['episode'][::eval_every],
                result['valid_score'][::eval_every])
            plt.ylabel('Validation Score'); plt.xlabel('Training Episodes')
            plt.tight_layout()
            plt.savefig(f'{fname_fig_dir}valid_scores.png')
            plt.figure()
            plt.plot(
                result['episode'][::eval_every],
                result['valid_steps_per_ep'][::eval_every])
            plt.ylabel('Validation Steps to Goal'); plt.xlabel('Training Episodes')
            plt.tight_layout()
            plt.savefig(f'{fname_fig_dir}valid_steps.png')
            plt.close('all')
        else:
            result['valid_score'].append(None)
            result['valid_steps_per_ep'].append(None)
        if episode % save_net_every == 0:
            agent.save_network(fname_nnet_dir, episode)

    # Save pickle
    with open(f'{pickle_dir}{fname}.p', 'wb') as f:
        pickle.dump(result, f)

# Load model parameters
fname_grid, loss_weights_grid, param_updates = load_function()
assert(len(fname_grid) == len(loss_weights_grid))
assert(len(fname_grid) == len(param_updates))
fname_grid = [f'{fname_prefix}_{f}' for f in fname_grid]

# Collect argument combination
iters = np.arange(n_iters)
args = []
for arg_idx in range(len(fname_grid)):
    for i in iters:
        fname = fname_grid[arg_idx]
        loss_weights = loss_weights_grid[arg_idx]
        param_update = param_updates[arg_idx]
        args.append([fname, loss_weights, param_update, i])
split_args = np.array_split(args, n_jobs)

import time
start = time.time()
# Run relevant parallelization script
if job_idx == -1:
    cpu_parallel()
else:
    gpu_parallel(job_idx)
end = time.time()

print(f'ELAPSED TIME: {end-start} seconds')
