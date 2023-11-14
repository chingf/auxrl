import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import copy
import dm_env
import enum

import torch
import torchvision
import torchvision.transforms as transforms

from acme import specs
from acme import wrappers
from acme.utils import tree_utils

from auxrl.environments import GridWorld

class Env(GridWorld.Env):
    def __init__(
        self, layout, local=True, random_p=0.3,
        ):

        super().__init__(layout, shuffle_obs=True)
        self._local = local
        self._random_p = random_p

    def step(self, action):
        x, y = self._state
        if np.random.uniform() < self._random_p:
            swap_action_pool = [i for i in range(4) if i != action]
            action = np.random.choice(swap_action_pool)
        if action == 0:  # left
            new_state = (x-1, y)
        elif action == 1:  # right
            new_state = (x+1, y)
        elif action == 2:  # up
            new_state = (x, y-1)
        elif action == 3:  # down
            new_state = (x, y+1)
        else:
            raise ValueError('Invalid action')
       
        new_x, new_y = new_state
        step_type = dm_env.StepType.MID
        if self._layout[new_x, new_y] == -1:  # wall
            reward = self._penalty_for_walls
            discount = self._discount
            new_state = (x, y)
        elif self._layout[new_x, new_y] == 0:  # empty cell
            reward = 0.
            discount = self._discount
        else:  # a goal
            reward = self._layout[new_x, new_y]
            discount = self._discount #0.
            new_state = self._start_state
            step_type = dm_env.StepType.LAST
    
        self._state = new_state
        self._num_episode_steps += 1
        if (self._max_episode_length is not None and
            self._num_episode_steps >= self._max_episode_length):
            step_type = dm_env.StepType.LAST
        return dm_env.TimeStep(
            step_type=step_type, reward=np.float32(reward),
            discount=discount, observation=self.get_obs())
