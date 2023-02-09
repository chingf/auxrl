import sys
import logging
import pickle
import re
import yaml
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
from joblib import hash, dump, load
import os

from deer.default_parser import process_args
from deer.agent import NeuralAgent
from deer.learning_algos.CRAR_torch import CRAR
import deer.controllers as bc
from deer.environments.Foraging import MyEnv as Env

from deer.policies import EpsilonGreedyPolicy

# Experiment Parameters
job_idx = int(sys.argv[1])
n_jobs = int(sys.argv[2])
nn_yaml = sys.argv[3]
internal_dim = int(sys.argv[4])
device_num = sys.argv[5]
if int(device_num) >= 0:
    my_env = os.environ
    my_env["CUDA_VISIBLE_DEVICES"] = device_num
fname_prefix = 'transfer_sr'
fname_suffix = ''
epochs = 30
source_prefix = 'sr'
source_suffix = ''
source_epoch = 30
policy_eps = 1. 
encoder_only = True
higher_dim_obs = True
size_maze = 6

# Make directories
engram_dir = '/home/cf2794/engram/Ching/rl/' # Cortex Path
#engram_dir = '/mnt/smb/locker/aronov-locker/Ching/rl/' # Axon Path
exp_dir = f'{fname_prefix}_{nn_yaml}_dim{internal_dim}{fname_suffix}/'
source_dir = f'{source_prefix}_{nn_yaml}_dim{internal_dim}{source_suffix}/'
for d in ['pickles/', 'nnets/', 'scores/', 'figs/', 'params/']:
    os.makedirs(f'{engram_dir}{d}{exp_dir}', exist_ok=True)

def gpu_parallel(job_idx):
    results_dir = f'{engram_dir}pickles/{exp_dir}'
    results = {}
    results['dimensionality_tracking'] = []
    results['dimensionality_variance_ratio'] = []
    results['valid_scores'] = []
    results['valid_steps'] = []
    results['iteration'] = []
    results['valid_eps'] = []
    results['training_eps'] = []
    results['epochs'] = []
    results['fname'] = []
    results['loss_weights'] = []
    for _arg in split_args[job_idx]:
        fname, loss_weights, result = run_env(_arg)
        for key in result.keys():
            results[key].append(result[key])
            results['fname'].append(fname)
            results['loss_weights'].append(loss_weights)
    with open(f'{results_dir}results_{job_idx}.p', 'wb') as f:
        pickle.dump(results, f)

def cpu_parallel():
    results_dir = f'{engram_dir}pickles/{exp_dir}'
    os.makedirs(results_dir, exist_ok=True)
    results = {}
    results['dimensionality_tracking'] = []
    results['dimensionality_variance_ratio'] = []
    results['valid_scores'] = []
    results['valid_steps'] = []
    results['iteration'] = []
    results['valid_eps'] = []
    results['training_eps'] = []
    results['epochs'] = []
    results['fname'] = []
    results['loss_weights'] = []
    job_results = Parallel(n_jobs=56)(delayed(run_env)(arg) for arg in args)
    for job_result in job_results:
        fname, loss_weights, result = job_result
        for key in result.keys():
            results[key].append(result[key])
        results['fname'].append(fname)
        results['loss_weights'].append(loss_weights)
    with open(f'{results_dir}results_0.p', 'wb') as f:
        pickle.dump(results, f)

def run_env(arg):
    _fname, network_file, loss_weights, i = arg
    nnet_dir = f'{engram_dir}nnets/{source_dir}'
    if network_file is None:
        set_network = None
    else:
        network_file_options = [
            s for s in os.listdir(nnet_dir) if \
            (re.search(f"^({network_file})_\\d+", s) != None)]
        network_file_idx = np.random.choice(len(network_file_options))
        network_file_path = f'{source_dir}{network_file_options[network_file_idx]}'
        set_network = [f'{network_file_path}', source_epoch, encoder_only]
    fname = f'{exp_dir}{_fname}_{i}'
    encoder_type = 'variational' if loss_weights[-1] > 0 else 'regular'
    parameters = {
        'nn_yaml': nn_yaml,
        'higher_dim_obs': higher_dim_obs,
        'internal_dim': internal_dim,
        'fname': fname,
        'steps_per_epoch': 1000,
        'epochs': epochs,
        'steps_per_test': 1000,
        'period_btw_summary_perfs': 1,
        'encoder_type': encoder_type,
        'frame_skip': 2,
        'learning_rate': 1*1E-4,
        'learning_rate_decay': 1.0,
        'discount': 0.9,
        'epsilon_start': 1.0,
        'epsilon_min': 1.0,
        'epsilon_decay': 1000,
        'update_frequency': 1,
        'replay_memory_size': 100000, #50000
        'batch_size': 64,
        'freeze_interval': 1000,
        'deterministic': False,
        'loss_weights': loss_weights,
        'foraging_give_rewards': True,
        'size_maze': size_maze,
        'pred_len': 1,
        'pred_gamma': 0.
        }
    with open(f'{engram_dir}params/{_fname}.yaml', 'w') as outfile:
        yaml.dump(parameters, outfile, default_flow_style=False)
    rng = np.random.RandomState()
    env = Env(
        rng, reward=parameters['foraging_give_rewards'],
        higher_dim_obs=parameters['higher_dim_obs'], plotfig=False,
        size_maze=parameters['size_maze']
        )
    learning_algo = CRAR(
        env, parameters['freeze_interval'], parameters['batch_size'], rng,
        internal_dim=parameters['internal_dim'],
        lr=parameters['learning_rate'], nn_yaml=parameters['nn_yaml'],
        double_Q=True, loss_weights=parameters['loss_weights'],
        encoder_type=parameters['encoder_type'],
        pred_len=parameters['pred_len'], pred_gamma=parameters['pred_gamma']
        )
    print(f'DEVICE USED: {learning_algo.device}')
    train_policy = EpsilonGreedyPolicy(
        learning_algo, env.nActions(), rng, policy_eps)
    test_policy = EpsilonGreedyPolicy(learning_algo, env.nActions(), rng, 0.)
    agent = NeuralAgent(
        env, learning_algo, parameters['replay_memory_size'], 1,
        parameters['batch_size'], rng,
        train_policy=train_policy, test_policy=test_policy,
        save_dir=engram_dir
        )
    if set_network is not None:
        agent.setNetwork(
            f'{set_network[0]}/fname', nEpoch=set_network[1],
            encoder_only=set_network[2]
            )
    agent.run(10, 500)
    agent.attach(bc.VerboseController( evaluate_on='epoch', periodicity=1))
    agent.attach(bc.LearningRateController(
        initial_learning_rate=parameters['learning_rate'],
        learning_rate_decay=parameters['learning_rate_decay'],
        periodicity=1))
    agent.attach(bc.TrainerController(
        evaluate_on='action', periodicity=parameters['update_frequency'],
        show_episode_avg_V_value=True, show_avg_Bellman_residual=True))
    best_controller = bc.FindBestController(
        validationID=Env.VALIDATION_MODE, testID=None,
        unique_fname=fname)
    agent.attach(best_controller)
    agent.attach(bc.InterleavedTestEpochController(
        id=Env.VALIDATION_MODE, epoch_length=parameters['steps_per_test'],
        periodicity=1, show_score=True, summarize_every=5, unique_fname=fname))
    if set_network is not None:
        agent.setNetwork(
            f'{set_network[0]}/fname', nEpoch=set_network[1],
            encoder_only=set_network[2]
            )
    if freeze_encoder:
        parameter_lists = [
            agent._learning_algo.crar.encoder.parameters(),
            agent._learning_algo.crar_target.encoder.parameters(),
            agent._learning_algo.crar.transition.parameters(),
            agent._learning_algo.crar_target.transition.parameters(),
            ]
        for parameter_list in parameter_lists:
            for p in parameter_list:
                p.requires_grad = False
    agent.run(parameters['epochs'], parameters['steps_per_epoch'])

    result = {
        'dimensionality_tracking': env._dimensionality_tracking[-1],
        'dimensionality_variance_ratio': env._dimensionality_variance_ratio,
        'valid_scores':  best_controller._validationScores,
        'valid_steps':  best_controller._validationSteps, 'iteration': i,
        'valid_eps': best_controller._validationEps,
        'epochs': best_controller._epochNumbers, 'training_eps': agent.n_eps
        }
    return _fname, loss_weights, result

# load user-defined parameters

fname_grid = [
    'entro', 'mb',
    'sr_5_0.9', 'sr_10_0.9',
    'mf']
network_files = [f'{source_prefix}_{f}' for f in fname_grid]
fname_grid.append('clean')
network_files.append(None)
loss_weights_grid = [[0., 0., 0., 1., 0.]] * len(fname_grid)
fname_grid = [f'{fname_prefix}_{f}' for f in fname_grid]
freeze_encoder = False
iters = np.arange(60)
args = []
for i in iters:
    for j in range(len(fname_grid)):
        fname = fname_grid[j]
        network_file = network_files[j]
        loss_weights = loss_weights_grid[j]
        args.append([fname, network_file, loss_weights, i])
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
