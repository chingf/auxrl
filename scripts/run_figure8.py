import sys
import logging
import pickle
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
from deer.environments.Figure8 import MyEnv as Env

from deer.policies import EpsilonGreedyPolicy, FixedFigure8Policy

# Experiment Parameters
job_idx = int(sys.argv[1])
n_jobs = int(sys.argv[2])
nn_yaml = sys.argv[3]
internal_dim = int(sys.argv[4])
n_gpus = (len(os.environ['CUDA_VISIBLE_DEVICES'])+1)/2
if n_gpus > 1:
    device_num = str(job_idx % n_gpus)
    my_env = os.environ
    my_env["CUDA_VISIBLE_DEVICES"] = device_num
fname_prefix = 'noisy_altT_eps0.5_volweight'
fname_suffix = ''
epochs = 41
policy_eps = 0.5
higher_dim_obs = True

# Make directories
#engram_dir = '/home/cf2794/engram/Ching/rl/' # Cortex Path
engram_dir = '/mnt/smb/locker/aronov-locker/Ching/rl/' # Axon Path
exp_dir = f'{fname_prefix}_{nn_yaml}_dim{internal_dim}{fname_suffix}/'
for d in ['pickles/', 'nnets/', 'figs/', 'latents/']:
    os.makedirs(f'{engram_dir}{d}{exp_dir}', exist_ok=True)

def gpu_parallel(job_idx):
    results_dir = f'{engram_dir}pickles/{exp_dir}'
    results = {}
    results['separability_matrix'] = []
    results['separability_slope'] = []
    results['separability_tracking'] = []
    results['dimensionality_tracking'] = []
    results['valid_scores'] = []
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
    results = {}
    results['separability_matrix'] = []
    results['separability_slope'] = []
    results['separability_tracking'] = []
    results['dimensionality_tracking'] = []
    results['valid_scores'] = []
    results['epochs'] = []
    results['fname'] = []
    results['loss_weights'] = []
    job_results = Parallel(n_jobs=40)(delayed(run_env)(arg) for arg in args)
    for job_result in job_results:
        fname, loss_weights, result = job_result
        for key in result.keys():
            results[key].append(result[key])
        results['fname'].append(fname)
        results['loss_weights'].append(loss_weights)
    with open(f'{results_dir}results_0.p', 'wb') as f:
        pickle.dump(results, f)

def run_env(arg):
    _fname, loss_weights, param_update, i = arg
    fname = f'{exp_dir}{_fname}_{i}'
    encoder_type = 'variational' if loss_weights[-1] > 0 else 'regular'
    parameters = {
        'figure8_give_rewards': True,
        'nn_yaml': nn_yaml,
        'higher_dim_obs': higher_dim_obs,
        'internal_dim': internal_dim,
        'fname': fname,
        'steps_per_epoch': 2000,
        'epochs': epochs,
        'steps_per_test': 500,
        'mem_len': 2,
        'train_len': 10,
        'encoder_type': encoder_type,
        'frame_skip': 2,
        'show_rewards': False,
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
        'yaml_mods': {},
        'volume_weight': 1E-4
        }
    parameters.update(param_update)
    with open(f'{engram_dir}params/{_fname}.yaml', 'w') as outfile:
        yaml.dump(parameters, outfile, default_flow_style=False)
    rng = np.random.RandomState()
    env = Env(
        give_rewards=parameters['figure8_give_rewards'],
        higher_dim_obs=parameters['higher_dim_obs'],
        show_rewards=parameters['show_rewards'],
        )
    learning_algo = CRAR(
        env, parameters['freeze_interval'], parameters['batch_size'], rng,
        internal_dim=parameters['internal_dim'], lr=parameters['learning_rate'],
        nn_yaml=parameters['nn_yaml'], yaml_mods=parameters['yaml_mods'],
        double_Q=True, loss_weights=parameters['loss_weights'],
        encoder_type=parameters['encoder_type'], mem_len=parameters['mem_len'],
        train_len=parameters['train_len'],
        volume_weight=parameters['volume_weight']
        )
    if parameters['figure8_give_rewards']:
        train_policy = EpsilonGreedyPolicy(
            learning_algo, env.nActions(), rng, epsilon=policy_eps)
        test_policy = EpsilonGreedyPolicy(
            learning_algo, env.nActions(), rng, 0.)
    else:
        train_policy = FixedFigure8Policy.FixedFigure8Policy(
            learning_algo, env.nActions(), rng, epsilon=policy_eps,
            height=env.HEIGHT, width=env.WIDTH)
        test_policy = FixedFigure8Policy.FixedFigure8Policy(
            learning_algo, env.nActions(), rng,
            height=env.HEIGHT, width=env.WIDTH)
    agent = NeuralAgent(
        env, learning_algo, parameters['replay_memory_size'], 1,
        parameters['batch_size'], rng, save_dir=engram_dir,
        train_policy=train_policy, test_policy=test_policy)
    agent.run(10, 500)
    agent.attach(bc.TrainerController(
        evaluate_on='action',  periodicity=parameters['update_frequency'],
        show_episode_avg_V_value=True, show_avg_Bellman_residual=True))
    best_controller = bc.FindBestController(
        validationID=Env.VALIDATION_MODE, testID=None, unique_fname=fname,
        savefrequency=5)
    agent.attach(best_controller)
    agent.attach(bc.InterleavedTestEpochController(
        id=Env.VALIDATION_MODE, epoch_length=parameters['steps_per_test'],
        periodicity=1, show_score=True, summarize_every=10, unique_fname=fname))
    agent.run(parameters['epochs'], parameters['steps_per_epoch'])

    result = {
        'separability_matrix':  env._separability_matrix,
        'separability_slope': env._separability_slope[-1], 
        'separability_tracking': [s[-1] for s in env._separability_tracking],
        'dimensionality_tracking': env._dimensionality_tracking[-1],
        'valid_scores':  best_controller._validationScores,
        'epochs': best_controller._epochNumbers,
        }
    return _fname, loss_weights, result

fname_grid = [
    'mf',
    'mb',
#    'entropy',
    ]
loss_weights_grid = [
    [0, 0, 0, 1, 0], 
    [1E-1, 1E-2, 1E-2, 1, 0],
#    [0, 1E-2, 1E-2, 1, 0],
    ]
param_updates = [{}]*len(fname_grid)
fname_grid = [f'{fname_prefix}_{f}' for f in fname_grid]
iters = np.arange(32)
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

