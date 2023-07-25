import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import copy
import dm_env
import enum

from acme import specs
from acme import wrappers
from acme.utils import tree_utils

class TrialType(enum.IntEnum):
    VERTICAL = enum.auto() # Rewarded
    ANGLED = enum.auto() # Not rewarded

class Action(enum.IntEnum):
    FORWARD = enum.auto()
    BACKWARD = enum.auto()
    LICK = enum.auto()

class Env(dm_env.Environment):
    """
    From "Learning Enhances Sensory and Multiple Non-sensory Representations
    in Primary Visual Cortex", Poort & Khan 2022
    """

    def __init__(
        self, n_approach_states=5, n_grating_states=10,
        ):

        self.n_approach_states = n_approach_states
        self.n_grating_states = n_grating_states
        self.n_total_states = n_approach_states + n_grating_states
        self.eval_mode = False
        self.create_stimuli()
        self.layout_dims = (1,) + self.base_vertical_state.shape
        self.reset()

    def create_stimuli(self):
        period = 10
        n_periods = 6
        vert_stripe_width = 3
        stimulus_size = (period, period*n_periods)
        self.approach_states = np.random.choice(
            2, size=(self.n_approach_states,)+stimulus_size)
        self.base_vertical_state = np.zeros(stimulus_size)
        for start in np.arange(0, (period*n_periods)-vert_stripe_width, vert_stripe_width*2):
            self.base_vertical_state[:, start:start+vert_stripe_width] = 1
        self.base_angled_state = np.diag(np.ones(period))
        self.base_angled_state += np.diag(np.ones(period-1), k=1)
        self.base_angled_state += np.diag(np.ones(period-1), k=-1)
        self.base_angled_state = np.tile(self.base_angled_state, (1, n_periods))

    def reset(self):
        self.curr_state = np.random.choice(self.n_approach_states)
        self.trial_type = np.random.choice([TrialType.VERTICAL, TrialType.ANGLED])
        if True: #self.eval_mode:
            self.trial_type = TrialType.VERTICAL
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST, reward=None, discount=None,
            observation=self.get_obs())

    def observation_spec(self):
        return specs.Array(
            shape=self.layout_dims, dtype=np.float32,
            name='observation_grid') # (C, H, W)

    def action_spec(self):
        return specs.DiscreteArray(3, dtype=int, name='action')

    def get_obs(self):
        obs = np.zeros(self.layout_dims, dtype=np.float32)
        if self.curr_state < self.n_approach_states:
            obs[0] = self.approach_states[self.curr_state]
        elif self.trial_type == TrialType.VERTICAL:
            shift = self.curr_state - self.n_approach_states
            obs[0] = np.roll(self.base_vertical_state, -shift)
        else:
            shift = self.curr_state - self.n_approach_states
            obs[0] = np.roll(self.base_angled_state, -shift)
        return obs

    def step(self, action):
        if self.curr_state == self.n_total_states - 1:
            if action == Action.FORWARD:
                new_state = self.curr_state + 1
                reward = 0
                step_type = dm_env.StepType.LAST
            elif action == Action.BACKWARD:
                new_state = max(0, self.curr_state - 1)
                reward = 0
                step_type = dm_env.StepType.MID
            else:
                new_state = self.curr_state + 1
                reward = self.trial_type == TrialType.VERTICAL
                step_type = dm_env.StepType.LAST
                print('rewarded')
        else:
            if action == Action.FORWARD:
                new_state = self.curr_state + 1
            elif action == Action.BACKWARD:
                new_state = max(0, self.curr_state - 1)
            else:
                new_state = self.curr_state
            reward = 0
            step_type = dm_env.StepType.MID
        self.curr_state = new_state
        discount =  1.
        return dm_env.TimeStep(
            step_type=step_type, reward=np.float32(reward),
            discount=discount, observation=self.get_obs())

def setup_environment(environment):
  """Returns the environment and its spec."""

  # Make sure the environment outputs single-precision floats.
  environment = wrappers.SinglePrecisionWrapper(environment)

  # Grab the spec of the environment.
  environment_spec = specs.make_environment_spec(environment)

  return environment, environment_spec
