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
from figure8_env import MyEnv as figure8_env
import deer.experiment.base_controllers as bc

from deer.policies import EpsilonGreedyPolicy, FixedFigure8Policy


def main():
    fname_grid = [
        'mf_only', 'mf_and_vae', 'mf_and_mb', 'mf_and_mb_and_vae']
    loss_weights_grid = [
        [0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 1., 1E-4],
        [5E-3, 1E-2, 1E-2, 0, 0, 1E-2, 1., 0],
        [5E-3, 1E-2, 1E-2, 0, 0, 1E-2, 1., 1E-4]
        ]
    iters = np.arange(10)
    args = []
    for fname, loss_weights in zip(fname_grid, loss_weights_grid):
        for i in iters:
            args.append([fname, loss_weights, i])
    results = {}
    results['separability_matrix'] = []
    results['separability_slope'] = []
    results['separability_tracking'] = []
    results['dimensionality_tracking'] = []
    results['valid_score'] = []
    results['fname'] = []
    results['loss_weights'] = []
    job_results = Parallel(n_jobs=8)(delayed(run_env)(arg) for arg in args)
    for job_result in job_results:
        fname, loss_weights, result = job_result
        for key in result.keys():
            results[key].append(result[key])
        results['fname'].append(fname)
        results['loss_weights'].append(loss_weights)
    with open('pickles/figure8_grid.p', 'wb') as f:
        pickle.dump(results, f)

def run_env(arg):
    _fname, loss_weights, i = arg
    fname = f'{_fname}_{i}'
    parameters = {
        'figure8_give_rewards': True,
        'nn_yaml': 'network_noconv.yaml',
        'higher_dim_obs': False,
        'internal_dim': 10,
        'fname': fname,
        'steps_per_epoch': 5000,
        'epochs': 50,
        'steps_per_test': 1000,
        'period_btw_summary_perfs': 1,
        'nstep': 15,
        'nstep_decay': 0.8,
        'encoder_type': 'variational',
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
        intern_dim=parameters['internal_dim'],
        higher_dim_obs=parameters['higher_dim_obs'],
        show_rewards=parameters['show_rewards'], nstep=parameters['nstep'],
        nstep_decay=parameters['nstep_decay'], plotfig=False
        )
    learning_algo = CRAR(
        env, parameters['freeze_interval'], parameters['batch_size'], rng,
        high_int_dim=False, internal_dim=parameters['internal_dim'],
        lr=parameters['learning_rate'], nn_yaml=parameters['nn_yaml'],
        double_Q=True, loss_weights=parameters['loss_weights'],
        nstep=parameters['nstep'], nstep_decay=parameters['nstep_decay'],
        encoder_type=parameters['encoder_type']
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
        periodicity=1, show_score=True, summarize_every=1, unique_fname=fname))
    agent.run(parameters['epochs'], parameters['steps_per_epoch'])

    result = {
        'separability_matrix':  env._separability_matrix,
        'separability_slope': env._separability_slope[-1], 
        'separability_tracking': [s[-1] for s in env._separability_tracking],
        'dimensionality_tracking': env._dimensionality_tracking[-1],
        'valid_score':  best_controller._validationScores[-1]
        }
    return _fname, loss_weights, result

if __name__ == "__main__":
    main()
