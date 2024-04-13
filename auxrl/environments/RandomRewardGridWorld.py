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

from auxrl.environments.GridWorld import Env as GridworldEnv

class Env(GridworldEnv):

    def __init__(
        self, layout, start_state=None, goal_state=None,
        p_reward=0.5, shuffle_obs=False,
        ):

        """
        Same as regular Gridworld but sometimes reward is not given
        """

        GridworldEnv.__init__(
            self, layout, start_state=start_state, goal_state=goal_state,
            shuffle_obs=shuffle_obs)
        self.p_reward = p_reward

    def step(self, action): 
        timestep = super().step(action)
        if self._eval:
            return timestep
        else:
            if timestep.step_type == dm_env.StepType.LAST:
                if (np.random.random() > self.p_reward):
                    timestep = dm_env.TimeStep(
                        step_type=timestep.step_type, reward=0.,
                        discount=timestep.discount,
                        observation=timestep.observation)
            return timestep

