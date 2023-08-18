import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import copy
import dm_env
import enum

from acme import specs

class RewardLoc(enum.IntEnum):
    RIGHT = enum.auto()
    LEFT = enum.auto()
    RESET = enum.auto()

class Env(dm_env.Environment):
    """
    Alternating-T Maze. Note that due to the borders of the maze, the effective
    maze size is actually (width-2, height-2).
    """

    def __init__(
        self, height=6, width=7, hide_goal=True, max_episode_length=2000,
        obs_noise=0., penalty_for_walls=-1., discount=1., reward_val=5.,
        temporal_context_len=0, temporal_context_gamma=0.9
        ):

        if width % 2 == 0: raise ValueError('Maze width must be odd.')
        self._height = height
        self._width = width
        self._layout_dims = (width, height)
        self._number_of_states = np.prod(self._layout_dims)
        self.make_layout()
        self._hide_goal = hide_goal
        self._max_episode_length = max_episode_length
        self._obs_noise = obs_noise
        self._penalty_for_walls = penalty_for_walls
        self._reward_val = reward_val
        self._tcm_len = temporal_context_len
        self._tcm_gamma = temporal_context_gamma
        self._discount = discount
        self._state = (width//2, 1)
        self._reward_loc = RewardLoc.LEFT
        self._goal_state = (1, self._height-2)
        self._last_reward_loc = RewardLoc.RESET
        self._space_label = self.make_space_labels()
        self._agent_loc_map = {}
        self._num_episode_steps = 0
        empty_obs = np.zeros((1,) + self._layout_dims, dtype=np.float32)
        self._prev_obs = [empty_obs for _ in range(self._tcm_len)]

    def in_bounds(self, state):
        x, y = state
        midpoint = self._width//2
        if (x < 1) or (x > self._width-2) or (y < 1) or (y > self._height-2):
            return False
        elif (y==1) or (y==self._height-2):
            return True
        elif x in [1, midpoint, self._width-2]:
            return True
        else:
            return False

    def make_layout(self):
        layout = np.zeros(self._layout_dims)
        for x in np.arange(self._width):
            for y in np.arange(self._height):
                if not self.in_bounds((x, y)):
                    layout[x, y] = -1
        self._layout = layout

    def make_space_labels(self):
        space_labels = np.zeros(self._layout_dims)
        midpoint = self._width//2
        space_labels[midpoint, self._height-2] = 0 # Decision point
        space_labels[:midpoint, :] = 1 # Left
        space_labels[midpoint+1:, :] = 2 # Right
        space_labels[midpoint, :] = 3 # Central
        space_labels[self._layout == -1] = -1 # Out of bounds
        return space_labels

    def observation_spec(self):
        return specs.Array(
            shape=(1,) + self._layout_dims, dtype=np.float32,
            name='observation_grid') # (C, H, W)

    def action_spec(self):
        return specs.DiscreteArray(4, dtype=int, name='action')

    def get_obs(self):
        left_reward = (1, self._height-2)
        right_reward = (self._width-2, self._height-2)
        reset_reward = (self._width//2, 1)
        obs = np.zeros((1,) + self._layout_dims, dtype=np.float32)
        obs[0, ...] = (self._layout < 0)*(-1)
        obs[0, self._state[0], self._state[1]] = 1
        if not self._hide_goal:
            if self._reward_loc == RewardLoc.LEFT:
                obs[0, left_reward[0], left_reward[1]] = 5
            elif self._reward_loc == RewardLoc.RIGHT:
                obs[0, right_reward[0], right_reward[1]] = 5
            else:
                obs[0, reset_reward[0], reset_reward[1]] = 5
        if self._obs_noise > 0.:
            obs = obs + np.random.normal(0, self._obs_noise, size=obs.shape)

        if self._tcm_len > 0:
            obs_without_tcm = obs.copy()
            for t in range(self._tcm_len):
                weight = self._tcm_gamma**(t+1)
                non_borders = obs >= 0
                obs[non_borders] += self._prev_obs[-(t+1)][non_borders] * weight
            self._prev_obs[:self._tcm_len-1] = self._prev_obs[1:]
            self._prev_obs[-1] = obs_without_tcm
        return obs

    def reset(self):
        self._state = (self._width//2, 1)
        self._reward_loc = RewardLoc.LEFT
        self._goal_state = (1, self._height-2)
        self._last_reward_loc = RewardLoc.RESET
        self._num_episode_steps = 0
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST, reward=None, discount=None,
            observation=self.get_obs())

    def step(self, action):
        x, y = self._state
        left_reward = (1, self._height-2)
        right_reward = (self._width-2, self._height-2)
        reset_reward = (self._width//2, 1)
        if action == 0:  # left
            new_state = (x-1, y)
        elif action == 1:  # right
            new_state = (x+1, y)
        elif action == 2:  # down
            new_state = (x, y-1)
        elif action == 3:  # up
            new_state = (x, y+1)
        else:
            raise ValueError('Invalid action')
        new_x, new_y = new_state
        step_type = dm_env.StepType.MID

        # If move hits a wall or is blocked, then it is invalid
        invalid_move = False
        if self._layout[new_x, new_y] == -1:  # wall
            invalid_move = True
        elif self._last_reward_loc == RewardLoc.LEFT:
            if (self._state == left_reward) and (action == 1):
                invalid_move = True
        elif self._last_reward_loc == RewardLoc.RIGHT:
            if (self._state == right_reward) and (action == 0):
                invalid_move = True
        elif self._last_reward_loc == RewardLoc.RESET:
            if (self._state == reset_reward) and (action != 3):
                invalid_move = True

        # Execute move if valid
        if invalid_move:
            reward = self._penalty_for_walls
            discount = self._discount
            new_state = (x, y)
        elif new_state == left_reward and self._reward_loc == RewardLoc.LEFT:
            reward = self._reward_val
            discount = self._discount
            self._reward_loc = RewardLoc.RESET
            self._goal_state = reset_reward
            self._last_reward_loc = RewardLoc.LEFT
            print('Left reward')
        elif new_state == right_reward and self._reward_loc == RewardLoc.RIGHT:
            reward = self._reward_val
            discount = self._discount
            self._reward_loc = RewardLoc.RESET
            self._goal_state = reset_reward
            self._last_reward_loc = RewardLoc.RIGHT
            print('Right reward')
        elif new_state == reset_reward and self._reward_loc == RewardLoc.RESET:
            reward = self._reward_val
            discount = self._discount
            if self._last_reward_loc == RewardLoc.RIGHT:
                self._reward_loc = RewardLoc.LEFT
                self._goal_state = left_reward
            elif self._last_reward_loc == RewardLoc.LEFT:
                self._reward_loc = RewardLoc.RIGHT
                self._goal_state = right_reward
            else:
                reward = 0
            self._last_reward_loc = RewardLoc.RESET
            print('Reset reward')
        else: # Just a standard move
            reward = 0.
            discount = self._discount
        self._state = new_state
        self._num_episode_steps += 1
        if (self._max_episode_length is not None and
            self._num_episode_steps >= self._max_episode_length):
            step_type = dm_env.StepType.LAST
        obs = self.get_obs()
        return dm_env.TimeStep(
            step_type=step_type, reward=np.float32(reward),
            discount=discount, observation=obs)

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

