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
from simple_maze_env import MyEnv as simple_maze_env
import deer.experiment.base_controllers as bc

from deer.policies import EpsilonGreedyPolicy

def gpu_parallel(job_idx):
    results_dir = 'pickles/foraging_rewardless_results/'
    os.makedirs(results_dir, exist_ok=True)
    results = {}
    results['dimensionality_tracking'] = []
    results['valid_scores'] = []
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

def run_env(arg):
    _fname, loss_weights, i = arg
    fname = f'{_fname}_{i}'
    encoder_type = 'variational' if loss_weights[-1] > 0 else 'regular'
    parameters = {
        'nn_yaml': 'network_simple.yaml',
        'higher_dim_obs': True,
        'internal_dim': 10,
        'fname': fname,
        'steps_per_epoch': 1000,
        'epochs': 40,
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
        'foraging_give_rewards': False
        }
    with open(f'params/{_fname}.yaml', 'w') as outfile:
        yaml.dump(parameters, outfile, default_flow_style=False)
    rng = np.random.RandomState()
    env = simple_maze_env(
        rng, reward=parameters['foraging_give_rewards'],
        higher_dim_obs=parameters['higher_dim_obs'], plotfig=False
        )
    learning_algo = CRAR(
        env, parameters['freeze_interval'], parameters['batch_size'], rng,
        high_int_dim=False, internal_dim=parameters['internal_dim'],
        lr=parameters['learning_rate'], nn_yaml=parameters['nn_yaml'],
        double_Q=True, loss_weights=parameters['loss_weights'],
        encoder_type=parameters['encoder_type']
        )
    test_policy = EpsilonGreedyPolicy(learning_algo, env.nActions(), rng, 0.)
    agent = NeuralAgent(
        env, learning_algo, parameters['replay_memory_size'], 1,
        parameters['batch_size'], rng,
        test_policy=test_policy)
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
        validationID=simple_maze_env.VALIDATION_MODE, testID=None, unique_fname=fname)
    agent.attach(best_controller)
    agent.attach(bc.InterleavedTestEpochController(
        id=simple_maze_env.VALIDATION_MODE, epoch_length=parameters['steps_per_test'],
        periodicity=1, show_score=True, summarize_every=1, unique_fname=fname))
    agent.run(parameters['epochs'], parameters['steps_per_epoch'])

    result = {
        'dimensionality_tracking': env._dimensionality_tracking[-1],
        'valid_scores':  best_controller._validationScores
        }
    return _fname, loss_weights, result


# load user-defined parameters
job_idx = int(sys.argv[1])
n_jobs = int(sys.argv[2])
fname_grid = ['foraging_rewardless']
loss_weights_grid = [
    [1E-2, 1E-3, 1E-3, 0, 0, 1E-2, 0., 0]
    ]
iters = np.arange(5)
args = []
for fname, loss_weights in zip(fname_grid, loss_weights_grid):
    for i in iters:
        args.append([fname, loss_weights, i])
split_args = np.array_split(args, n_jobs)

# Run relevant parallelization script
gpu_parallel(job_idx)

