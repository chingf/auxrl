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
net_type = 'simplest'
internal_dim = 5
fname_prefix = 'figure8'
fname_suffix = ''
epochs = 80
policy_eps = 1.
higher_dim_obs = False
foraging_give_rewards = True
expand_tcm = False

# Make directories
nn_yaml = f'network_{net_type}.yaml'
engram_dir = '/home/cf2794/engram/Ching/rl/'
exp_dir = f'{fname_prefix}_{net_type}_dim{internal_dim}{fname_suffix}/'
for d in ['pickles/', 'nnets/', 'figs/', 'params/']:
    os.makedirs(f'{engram_dir}{d}{exp_dir}', exist_ok=True)

def gpu_parallel(arg_idx):
    results_dir = f'pickles/{exp_dir}'
    results = {}
    results['separability_matrix'] = []
    results['separability_slope'] = []
    results['separability_tracking'] = []
    results['dimensionality_tracking'] = []
    results['valid_scores'] = []
    results['fname'] = []
    results['loss_weights'] = []
    for _arg in split_args[job_idx]:
        fname, loss_weights, result = run_env(args[i])
        for key in result.keys():
            results[key].append(result[key])
        results['fname'].append(fname)
        results['loss_weights'].append(loss_weights)
    with open(f'{results_dir}results_{arg_idx}.p', 'wb') as f:
        pickle.dump(results, f)

def cpu_parallel():
    results_dir = f'{engram_dir}pickles/{exp_dir}'
    results = {}
    results['separability_matrix'] = []
    results['separability_slope'] = []
    results['separability_tracking'] = []
    results['dimensionality_tracking'] = []
    results['valid_scores'] = []
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
    _fname, loss_weights, i = arg
    fname = f'{exp_dir}{_fname}_{i}'
    encoder_type = 'variational' if loss_weights[-1] > 0 else 'regular'
    parameters = {
        'figure8_give_rewards': True,
        'nn_yaml': nn_yaml,
        'higher_dim_obs': higher_dim_obs,
        'internal_dim': internal_dim,
        'fname': fname,
        'steps_per_epoch': 2500,
        'epochs': epochs,
        'steps_per_test': 1000,
        'period_btw_summary_perfs': 1,
        'nstep': 15,
        'nstep_decay': 1.,
        'expand_tcm': expand_tcm,
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
        'loss_weights': loss_weights
        }
    with open(f'params/{_fname}.yaml', 'w') as outfile:
        yaml.dump(parameters, outfile, default_flow_style=False)
    rng = np.random.RandomState()
    env = figure8_env(
        give_rewards=parameters['figure8_give_rewards'],
        higher_dim_obs=parameters['higher_dim_obs'],
        show_rewards=parameters['show_rewards'], plotfig=False
        )
    learning_algo = CRAR(
        env, parameters['freeze_interval'], parameters['batch_size'], rng,
        high_int_dim=False, internal_dim=parameters['internal_dim'],
        lr=parameters['learning_rate'], nn_yaml=parameters['nn_yaml'],
        double_Q=True, loss_weights=parameters['loss_weights'],
        nstep=parameters['nstep'], nstep_decay=parameters['nstep_decay'],
        encoder_type=parameters['encoder_type'],
        expand_tcm=parameters['expand_tcm']
        )
    if parameters['figure8_give_rewards']:
        train_policy = EpsilonGreedyPolicy(
            learning_algo, env.nActions(), rng, 0.2, consider_valid_transitions=False
            )
        test_policy = EpsilonGreedyPolicy(
            learning_algo, env.nActions(), rng, 0.
            )
    else:
        train_policy = FixedFigure8Policy.FixedFigure8Policy(
            learning_algo, env.nActions(), rng, epsilon=0.2,
            height=env.HEIGHT, width=env.WIDTH
            )
        test_policy = FixedFigure8Policy.FixedFigure8Policy(
            learning_algo, env.nActions(), rng,
            height=env.HEIGHT, width=env.WIDTH
            )
    agent = NeuralAgent(
        env, learning_algo, parameters['replay_memory_size'], 1,
        parameters['batch_size'], rng,
        train_policy=train_policy, test_policy=test_policy)
    agent.run(10, 500)
    agent.attach(bc.VerboseController( evaluate_on='epoch', periodicity=1))
    agent.attach(bc.LearningRateController(
        initial_learning_rate=parameters['learning_rate'],
        learning_rate_decay=parameters['learning_rate_decay'],
        periodicity=1))
    agent.attach(bc.TrainerController(
        evaluate_on='action',  periodicity=parameters['update_frequency'],
        show_episode_avg_V_value=True, show_avg_Bellman_residual=True))
    best_controller = bc.FindBestController(
        validationID=figure8_env.VALIDATION_MODE, testID=None, unique_fname=fname)
    agent.attach(best_controller)
    agent.attach(bc.InterleavedTestEpochController(
        id=figure8_env.VALIDATION_MODE, epoch_length=parameters['steps_per_test'],
        periodicity=1, show_score=True, summarize_every=10, unique_fname=fname))
    agent.run(parameters['epochs'], parameters['steps_per_epoch'])

    result = {
        'separability_matrix':  env._separability_matrix,
        'separability_slope': env._separability_slope[-1], 
        'separability_tracking': [s[-1] for s in env._separability_tracking],
        'dimensionality_tracking': env._dimensionality_tracking[-1],
        'valid_scores':  best_controller._validationScores
        }
    return _fname, loss_weights, result

job_idx = int(sys.argv[1])
n_jobs = int(sys.argv[2])
fname_grid = ['entro', 'mb', 'mb_only', 'mf']
loss_weights_grid = [
    [0, 1E-1, 1E-1, 1, 0],
    [1E-2, 1E-1, 1E-1, 1, 0],
    [1E-2, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
    ]
fname_grid = [f'{fname_prefix}_{f}' for f in fname_grid]
iters = np.arange(20)
args = []
for fname, loss_weights in zip(fname_grid, loss_weights_grid):
    for i in iters:
        args.append([fname, loss_weights, i])
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

