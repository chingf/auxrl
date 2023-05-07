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

class ObservationType(enum.IntEnum):
    """
    * STATE_INDEX: int32 index of agent occupied tile.
    * AGENT_ONEHOT: NxN float32 grid, 1 where the
      agent is and 0 elsewhere.
    * GRID: NxNx3 float32 grid of feature channels.
      First channel contains walls (1 if wall, 0 otherwise), second the
      agent position (1 if agent, 0 otherwise) and third goal position
      (1 if goal, 0 otherwise)
    * AGENT_GOAL_POS: float32 tuple with
      (agent_y, agent_x, goal_y, goal_x)

    """

    STATE_INDEX = enum.auto()
    AGENT_ONEHOT = enum.auto()
    GRID = enum.auto()
    AGENT_GOAL_POS = enum.auto()

class Env(dm_env.Environment):
    VALIDATION_MODE = 0

    def __init__(
        self, layout, start_state=None, goal_state=None,
        observation_type=ObservationType.GRID, discount=1.,
        penalty_for_walls=0., reward_goal=1., hide_goal=True,
        max_episode_length=300, prev_reward_goal=None):

        """Build a grid environment.

        Simple gridworld defined by a map layout, a start and a goal state.

        Layout should be a NxN grid, containing:
          * 0: empty
          * -1: wall
          * Any other positive value: reward; episode will terminate
        
        Args:
          layout: NxN array of numbers, layout of the environment.
          start_state: Tuple (y, x) of starting location.
          goal_state: Optional tuple (y, x) of goal location. Will be randomly
            sampled once if None.
          observation_type: Enum observation type to use.
          discount: Discounting factor included in all Timesteps.
          penalty_for_walls: Reward added when hitting a wall (<0).
          reward_goal: Reward added when finding the goal (should be positive).
          max_episode_length: If set, episode terminates after this many steps.
        """

        if observation_type not in ObservationType:
            raise ValueError('observation_type is the wrong type.')
        if type(layout) == int: 
            layout = np.zeros((layout+2, layout+2))
            layout[0,:] = layout[:,0] = layout[-1,:] = layout[:,-1] = -1
            self._layout = layout
        else:
            self._layout = np.array(layout)
        self._layout_dims = self._layout.shape
        if start_state is None:
            start_state = self._sample_start()
        self._start_state = start_state
        self._state = self._start_state
        self._number_of_states = np.prod(np.shape(self._layout))
        self._discount = discount
        self._penalty_for_walls = penalty_for_walls
        self._reward_goal = reward_goal
        self._hide_goal = hide_goal
        self._observation_type = observation_type
        self._max_episode_length = max_episode_length
        self._num_episode_steps = 0
        goal_state = self._sample_goal()
        self.goal_state = goal_state

    def _sample_start(self):
        """Randomly sample starting state."""
        n = 0
        max_tries = 1e5
        while n < max_tries:
            start_state = tuple(np.random.randint(d) for d in self._layout_dims)
            if self._layout[start_state] == 0:
                return start_state
            n += 1
        raise ValueError('Failed to sample a start state.')

    def _sample_goal(self):
        """Randomly sample reachable non-starting state."""
        n = 0
        max_tries = 1e5
        while n < max_tries:
            goal_state = tuple(np.random.randint(d) for d in self._layout_dims)
            if goal_state != self._state and self._layout[goal_state] == 0:
                if self.start_state == goal_state:
                    raise ValueError('Collision')
                return goal_state
        n += 1
        raise ValueError('Failed to sample a goal state.')

    @property
    def layout(self):
        return self._layout

    @property
    def number_of_states(self):
        return self._number_of_states

    @property
    def goal_state(self):
        return self._goal_state

    @property
    def start_state(self):
        return self._start_state

    @property
    def state(self):
        return self._state

    def set_state(self, x, y):
        self._state = (x, y)

    @goal_state.setter
    def goal_state(self, new_goal):
        if new_goal == self._state or self._layout[new_goal] < 0:
            raise ValueError('This is not a valid goal!')
        self._layout[self._layout > 0] = 0
        self._layout[new_goal] = self._reward_goal
        print('goal set')
        print(self._layout)
        self._goal_state = new_goal

    def observation_spec(self):
        if self._observation_type is ObservationType.AGENT_ONEHOT:
            return specs.Array(
                shape=self._layout_dims, dtype=np.float32,
                name='observation_agent_onehot')
        elif self._observation_type is ObservationType.GRID:
            return specs.Array(
                shape=(1,) + self._layout_dims, dtype=np.float32,
                name='observation_grid') # (C, H, W)
        elif self._observation_type is ObservationType.AGENT_GOAL_POS:
            return specs.Array(
                shape=(4,), dtype=np.float32,
                name='observation_agent_goal_pos')
        elif self._observation_type is ObservationType.STATE_INDEX:
            return specs.DiscreteArray(
                self._number_of_states, dtype=int,
                name='observation_state_index')

    def action_spec(self):
        return specs.DiscreteArray(4, dtype=int, name='action')

    def get_obs(self):
        if self._observation_type is ObservationType.AGENT_ONEHOT:
            obs = np.zeros(self._layout.shape, dtype=np.float32)
            obs[self._state] = 1 # Place agent
            return obs
        elif self._observation_type is ObservationType.GRID:
            obs = np.zeros((1,) + self._layout.shape, dtype=np.float32)
            obs[0, ...] = (self._layout < 0)*(-1)
            obs[0, self._state[0], self._state[1]] = 1
            if not self._hide_goal:
                obs[0, self._goal_state[0], self._goal_state[1]] = 5
            return obs
        elif self._observation_type is ObservationType.AGENT_GOAL_POS:
            return np.array(self._state + self._goal_state, dtype=np.float32)
        elif self._observation_type is ObservationType.STATE_INDEX:
            x, y = self._state
            return y * self._layout.shape[1] + x

    def reset(self, reset_start=True):
        if reset_start:
            self._start_state = self._sample_start()
            if self._start_state == self.goal_state:
                raise ValueError('Collision')
        self._state = self._start_state
        self._num_episode_steps = 0
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST, reward=None, discount=None,
            observation=self.get_obs())

    def step(self, action):
        x, y = self._state
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
            discount = 1. #0.
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

    def plot_grid(self, add_start=True):
        plt.figure(figsize=(4, 4))
        plt.imshow(self._layout <= -1, interpolation='nearest', cmap='Pastel1_r')
        ax = plt.gca()
        ax.grid(0)
        plt.xticks([]); plt.yticks([])
        if add_start:
            plt.text(
                self._start_state[1], self._start_state[0], r'$\mathbf{S}$',
                fontsize=16, ha='center', va='center')
        goal_text = r'$\mathbf{G}$'
        if self._hide_goal: goal_text = f'({goal_text})'
        plt.text(
            self._goal_state[1], self._goal_state[0], goal_text,
            fontsize=16, ha='center', va='center')
        w, h = self._layout.shape
        for y in range(h - 1):
            plt.plot([-0.5, w - 0.5], [y + 0.5, y + 0.5], '-k', lw=2)
        for x in range(w - 1):
            plt.plot([x + 0.5, x + 0.5], [-0.5, h - 0.5], '-k', lw=2)

    def plot_state(self, return_rgb=False):
        self.plot_grid(add_start=False)
        # Add the agent location as a smiley face
        plt.text(
            self._state[1], self._state[0], '\U0001F603',
            fontsize=18, ha='center', va='center')
        if return_rgb:
            fig = plt.gcf()
            plt.axis('tight')
            plt.subplots_adjust(0, 0, 1, 1, 0, 0)
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            w, h = fig.canvas.get_width_height()
            data = data.reshape((h, w, 3))
            plt.close(fig)
            return data

    def plot_policy(self, policy):
        action_names = [
             r'$\leftarrow$', r'$\rightarrow$', r'$\uparrow$', r'$\downarrow$']
        self.plot_grid()
        plt.title('Policy Visualization')
        w, h = self._layout.shape
        for y in range(h):
            for x in range(w):
                if (x, y) != self._goal_state:
                    action_name = action_names[policy[x, y]]
                    plt.text(x, y, action_name, ha='center', va='center')

    def plot_greedy_policy(self, q):
        greedy_actions = np.argmax(q, axis=2)
        self.plot_policy(greedy_actions)

def build_gridworld_task(
    task, discount=0.9, penalty_for_walls=-5,
    observation_type=ObservationType.STATE_INDEX, max_episode_length=200
    ):

    """ Construct a particular Gridworld layout with start/goal states. """

    tasks_specifications = {
        'simple': {
            'layout': [
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, -1, -1, 0, 0, 0, -1],
                [-1, 0, 0, 0, -1, -1, 0, 0, 0, -1],
                [-1, 0, 0, 0, -1, -1, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            ],
            'start_state': (2, 2),
            'goal_state': (7, 2)
        },
        'obstacle': {
            'layout': [
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, 0, 0, 0, 0, 0, -1, 0, 0, -1],
                [-1, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, -1, -1, 0, 0, 0, -1],
                [-1, 0, 0, 0, -1, -1, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            ],
            'start_state': (2, 2),
            'goal_state': (2, 8)
        },
        'random_goal': {
            'layout': [
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, -1, -1, 0, 0, 0, -1],
                [-1, 0, 0, 0, -1, -1, 0, 0, 0, -1],
                [-1, 0, 0, 0, -1, -1, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            ],
            'start_state': (2, 2),
        },
    }
    return GridWorld(
        discount=discount, penalty_for_walls=penalty_for_walls,
        observation_type=observation_type,
        max_episode_length=max_episode_length, **tasks_specifications[task])

def setup_environment(environment):
  """Returns the environment and its spec."""

  # Make sure the environment outputs single-precision floats.
  environment = wrappers.SinglePrecisionWrapper(environment)

  # Grab the spec of the environment.
  environment_spec = specs.make_environment_spec(environment)

  return environment, environment_spec
