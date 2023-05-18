import sys
import logging
import pickle
import yaml
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import shortuuid
from flatten_dict import flatten
from flatten_dict import unflatten
import torch

from acme import specs
from acme import wrappers

from auxrl.Agent import Agent
from auxrl.networks.Network import Network
from auxrl.environments.GridWorld import Env as Env
from auxrl.utils import run_train_episode, run_eval_episode

# Experiment Parameters
job_idx = int(sys.argv[1])
n_jobs = int(sys.argv[2])
nn_yaml = sys.argv[3]
internal_dim = int(sys.argv[4])
try:
    n_gpus = (len(os.environ['CUDA_VISIBLE_DEVICES'])+1)/2
except:
    n_gpus = 0
if n_gpus > 1:
    device_num = str(job_idx % n_gpus)
    my_env = os.environ
    my_env["CUDA_VISIBLE_DEVICES"] = device_num
fname_prefix = 'gridtestTD'
fname_suffix = ''
n_episodes = 201
n_cpu_jobs = 28
eval_every = 1
save_net_every = 50
epsilon = 1.
size_maze = 6
continual_transfer = False

# Make directories
#engram_dir = '/home/cf2794/engram/Ching/rl/' # Cortex Path
engram_dir = '/mnt/smb/locker/aronov-locker/Ching/rl/' # Axon Path
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
    fname_pickle_dir = f'{engram_dir}pickles/{exp_dir}/{fname}/'
    for _dir in [fname_nnet_dir, fname_fig_dir, fname_pickle_dir]:
        os.makedirs(_dir, exist_ok=True)
    parameters = {
        'fname': fname,
        'n_episodes': n_episodes,
        'n_test_episodes': 5,
        'continual_transfer': continual_transfer,
        'agent_args': {
            'loss_weights': loss_weights, 'lr': 1e-4,
            'replay_capacity': 20_000, 'epsilon': epsilon,
            'batch_size': 64, 'target_update_frequency': 1000,
            'train_seq_len': 1},
        'network_args': {'latent_dim': internal_dim, 'network_yaml': nn_yaml},
        'dset_args': {'layout': size_maze}
        }
    parameters = flatten(parameters)
    parameters.update(flatten(param_update))
    parameters = unflatten(parameters)
    with open(f'{param_dir}{_fname}.yaml', 'w') as outfile:
        yaml.dump(parameters, outfile, default_flow_style=False)
    env = Env(**parameters['dset_args'])
    #env = wrappers.SinglePrecisionWrapper(env)
    env_spec = specs.make_environment_spec(env)
    network = Network(env_spec, device=device, **parameters['network_args'])
    agent = Agent(env_spec, network, device=device, **parameters['agent_args'])

    os.makedirs(fname_nnet_dir, exist_ok=True)
    with open(f'{fname_nnet_dir}goal.txt', 'w') as goalfile:
        goalfile.write(str(env._goal_state))

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
            for i, loss_key in enumerate(loss_keys):
                ax = axs[i%3][i//3]
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
    unique_id = shortuuid.uuid()
    with open(f'{pickle_dir}{unique_id}.p', 'wb') as f:
        pickle.dump(result, f)

# Labels assigned to each network
fname_grid = [
    'mb_-3'
    'gamma0.25v2_-2',
#    'gamma0.25v2',
#    'gamma0.5v2_e-2',
#    'gamma0.8v2_e-2',
#    'gamma0.5v2',
#    'gamma0.v2',
#    'gamma0.8v2',
    ]

# Typical loss weights:
# MB: [1E-2, 1E-1, 1E-1, 1] Neigh: [1E-2, 1E-2, 0, 1]
loss_weights_grid = [
    [1E-3, 1E-1, 1E-1, 1],
    [1E-2, 1E-1, 1E-1, 1],
#    [1E-2, 1E-1, 1E-1, 1],
    ]

# If you want latents to predict future latents
# {'agent_args': {'pred_len': 10, 'pred_gamma': 0.93}},
# If you wanted latents to predict observations:
# {'yaml_mods': {'trans-pred': {'predict_z': False}}}
param_updates = [
        {},
        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.25}},
#        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.5}},
#        {'agent_args': {'pred_TD': True, 'pred_len': 2, 'pred_gamma': 0.8}},
    ]

fname_grid = [f'{fname_prefix}_{f}' for f in fname_grid]
iters = np.arange(5)
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
