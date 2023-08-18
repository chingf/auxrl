import gym
import enum
import time
import acme
import torch
import base64
import dm_env
import random
import warnings
import itertools
import collections

import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

from acme import environment_loop
from acme import specs
from acme import wrappers
from acme.utils import tree_utils
from acme.utils import loggers

def run_train_episode(
    environment: dm_env.Environment, agent: acme.Actor, clip_norm: float=-1.):
    """
    Each episode is itself a loop which interacts first with the environment to
    get an observation and then give that observation to the agent in order to
    retrieve an action.

    Args:
      environment: dm_env.Environment used to generate trajectories.
      agent: acme.Actor for selecting actions in the run loop.
    """

    episode_steps = 0
    episode_return = 0
    summed_episode_losses = []

    timestep = environment.reset()
    agent.reset()
    agent.observe_first(timestep)

    while not timestep.last(): # Until terminal state reached
        action = agent.select_action(timestep.observation)
        timestep = environment.step(action)
        agent.observe(
            action, next_timestep=timestep, latent=agent.get_curr_latent())
        episode_losses = agent.update(clip_norm=clip_norm)
        if summed_episode_losses == []:
            summed_episode_losses = episode_losses
        else:
            for i in range(len(summed_episode_losses)):
                summed_episode_losses[i] += episode_losses[i]
        episode_steps += 1
        episode_return += timestep.reward

    avg_episode_losses = [l/episode_steps for l in summed_episode_losses]
    return avg_episode_losses, episode_return, episode_steps

def run_eval_episode(
    env: dm_env.Environment, agent: acme.Actor, n_test_episodes: int=1,
    verbose: bool=False, max_episode_steps=None):
    all_episode_steps = []
    all_episode_return = []
    for _t in range(n_test_episodes):
        episode_steps = 0
        episode_return = 0
        timestep = env.reset()
        while not timestep.last():
            action = agent.select_action(
                timestep.observation, force_greedy=True, verbose=verbose)
            if verbose:
                #env.plot_state()
                plt.figure()
                plt.imshow(timestep.observation.squeeze(), vmin=-1, vmax=4.5)
                plt.show()
            timestep = env.step(action)
            episode_steps += 1
            episode_return += timestep.reward
            if (max_episode_steps != None) and (episode_steps >= max_episode_steps):
                break

        all_episode_steps.append(episode_steps)
        all_episode_return.append(episode_return)
    all_episode_steps = np.mean(all_episode_steps)
    all_episode_return = np.mean(all_episode_return)
    print(f'[EVAL] {all_episode_return} score over {all_episode_steps} steps.')
    return all_episode_return, all_episode_steps

def run_modified_transition(
    environment: dm_env.Environment, agent: acme.Actor, max_timestep=20):
    """
    Each episode is itself a loop which interacts first with the environment to
    get an observation and then give that observation to the agent in order to
    retrieve an action.
    
    Args:
      environment: dm_env.Environment used to generate trajectories.
      agent: acme.Actor for selecting actions in the run loop.
    """
    
    episode_steps = 0
    episode_return = 0
    summed_episode_losses = []
    
    timestep = environment.reset()
    timestep_pairs = environment.get_timestep_pairs_of_swapped_transitions()
    n_iters = (agent._batch_size // len(timestep_pairs)) + 1
    for _ in range(n_iters):
        for timestep, next_timestep, action in timestep_pairs:
            agent._replay_buffer.add_artificial_transition(
                timestep, next_timestep, action)
    
    while (not timestep.last()) and (episode_steps < max_timestep):
        episode_losses = agent.update()
        if summed_episode_losses == []:
            summed_episode_losses = episode_losses
        else:
            for i in range(len(summed_episode_losses)):
                summed_episode_losses[i] += episode_losses[i]
        episode_steps += 1
    
    avg_episode_losses = [l/episode_steps for l in summed_episode_losses]
    return avg_episode_losses, episode_return, episode_steps

#def display_video(frames: Sequence[np.ndarray],
#                  filename: str = 'temp.mp4',
#                  frame_rate: int = 12):
#  """Save and display video."""
#  # Write the frames to a video.
#  with imageio.get_writer(filename, fps=frame_rate) as video:
#    for frame in frames:
#      video.append_data(frame)
#
#  # Read video and display the video.
#  video = open(filename, 'rb').read()
#  b64_video = base64.b64encode(video)
#  video_tag = ('<video  width="320" height="240" controls alt="test" '
#               'src="data:video/mp4;base64,{0}">').format(b64_video.decode())
#  return IPython.display.HTML(video_tag)
