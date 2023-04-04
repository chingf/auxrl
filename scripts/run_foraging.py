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
from deer.environments.Foraging import MyEnv as Env

from deer.policies import EpsilonGreedyPolicy

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
fname_prefix = 'continual6x6_pt1'
fname_suffix = ''
epochs = 61
policy_eps = 1.
higher_dim_obs = True
foraging_give_rewards = True
size_maze = 6 + 2

# Make directories
#engram_dir = '/home/cf2794/engram/Ching/rl/' # Cortex Path
engram_dir = '/mnt/smb/locker/aronov-locker/Ching/rl/' # Axon Path
exp_dir = f'{fname_prefix}_{nn_yaml}_dim{internal_dim}{fname_suffix}/'
for d in ['pickles/', 'nnets/', 'figs/', 'params/']:
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
    _fname, loss_weights, param_update, i = arg
    fname = f'{exp_dir}{_fname}_{i}'
    encoder_type = 'variational' if loss_weights[-1] > 0 else 'regular'
    parameters = {
        'nn_yaml': nn_yaml,
        'higher_dim_obs': higher_dim_obs,
        'internal_dim': internal_dim,
        'fname': fname,
        'steps_per_epoch': 500,
        'epochs': epochs,
        'steps_per_test': 1000,
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
        'batch_size': 64, #256,
        'freeze_interval': 1000,
        'deterministic': False,
        'loss_weights': loss_weights,
        'foraging_give_rewards': foraging_give_rewards,
        'size_maze': size_maze,
        'pred_len': 1,
        'pred_gamma': 0., 
        'yaml_mods': {}
        }
    parameters.update(param_update)
    with open(f'{engram_dir}params/{_fname}.yaml', 'w') as outfile:
        yaml.dump(parameters, outfile, default_flow_style=False)
    rng = np.random.RandomState()
    env = Env(
        rng, reward=parameters['foraging_give_rewards'],
        higher_dim_obs=parameters['higher_dim_obs'], plotfig=False,
        size_maze=parameters['size_maze']
        )
    os.makedirs(f'{engram_dir}nnets/{fname}', exist_ok=True)
    with open(f'{engram_dir}nnets/{fname}/goal.txt', 'w') as goalfile:
        goalfile.write(str(env._pos_goal))
    learning_algo = CRAR(
        env, parameters['freeze_interval'], parameters['batch_size'], rng,
        internal_dim=parameters['internal_dim'], lr=parameters['learning_rate'],
        nn_yaml=parameters['nn_yaml'], yaml_mods=parameters['yaml_mods'],
        double_Q=True, loss_weights=parameters['loss_weights'],
        encoder_type=parameters['encoder_type'],
        pred_len=parameters['pred_len'], pred_gamma=parameters['pred_gamma']
        )
    train_policy = EpsilonGreedyPolicy(
        learning_algo, env.nActions(), rng, policy_eps)
    test_policy = EpsilonGreedyPolicy(learning_algo, env.nActions(), rng, 0.)
    agent = NeuralAgent(
        env, learning_algo, parameters['replay_memory_size'], 1,
        parameters['batch_size'], rng, save_dir=engram_dir,
        train_policy=train_policy, test_policy=test_policy)
    agent.run(10, 500)
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
    'entro',
    'mb',
    'mf',
#    'sr',
    ]
loss_weights_grid = [ # [1E-2, 1E-1, 1E-1, 1, 0] 
#    [0, 1E-2, 0, 1, 0], #neighbor
#    [1E-2, 1E-2, 0, 1, 0], #neighbor
    [0, 1E-1, 1E-1, 1, 0],
    [1E-2, 1E-1, 1E-1, 1, 0],
    [0, 0, 0, 1, 0],
#    [1E-2, 1E-1, 1E-1, 1, 0],
    ]
param_updates = [
    {},
    {},
    {},
#    {'pred_len': 10, 'pred_gamma': 0.93}
    ]
# If you want latents to predict future latents
# {'pred_len': 15, 'pred_gamma': 0.93},
# If you wanted latents to predict observations:
# {'yaml_mods': {'trans-pred': {'predict_z': False}}}

fname_grid = [f'{fname_prefix}_{f}' for f in fname_grid]
iters = np.arange(18)
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
