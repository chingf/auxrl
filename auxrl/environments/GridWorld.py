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
    * GRID: NxNx3 float32 grid of feature channels.
      First channel contains walls (1 if wall, 0 otherwise), second the
      agent position (1 if agent, 0 otherwise) and third goal position
      (1 if goal, 0 otherwise)
    * CIFAR: Randomly selected images from the CIFAR10 testset, in grayscale.

    """

    GRID = enum.auto()
    CIFAR = enum.auto()

class Env(dm_env.Environment):

    def __init__(
        self, layout, start_state=None, goal_state=None,
        observation_type=ObservationType.GRID, discount=1.,
        penalty_for_walls=0., reward_goal=1., hide_goal=True,
        max_episode_length=150, prev_goal_state=None, shuffle_obs=False,
        add_barrier=False, shuffle_states=False):

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
        self._number_of_states = np.prod(self._layout_dims)
        if shuffle_obs:
            self._shuffle_indices = np.arange(self._number_of_states)
            np.random.shuffle(self._shuffle_indices)
        if start_state is None:
            start_state = self._sample_start()
        self._start_state = start_state
        self._state = self._start_state
        self._discount = discount
        self._penalty_for_walls = penalty_for_walls
        self._reward_goal = reward_goal
        self._hide_goal = hide_goal
        self._observation_type = observation_type
        self._max_episode_length = max_episode_length
        self._prev_goal_state = prev_goal_state
        self._shuffle_obs = shuffle_obs
        self._add_barrier = add_barrier
        if self._prev_goal_state != None:
            self._new_goal_state_gap = min(min(self._layout_dims)//3, 3)
        self._num_episode_steps = 0
        goal_state = self._sample_goal()
        self.goal_state = goal_state
        self.transitions_swapped = False
        self._shuffle_states = shuffle_states
        if self._observation_type == ObservationType.CIFAR:
            self._make_cifar_images()
        if shuffle_states:
            self._swap_map = {}
            initial_state = np.argwhere(self._layout >= 0)
            swapped_state = initial_state.copy()
            np.random.shuffle(swapped_state)
            for idx in range(initial_state.shape[0]):
                s1 = str(initial_state[idx].tolist())
                s2 = swapped_state[idx].tolist()
                self._swap_map[s1] = s2

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
                if self._prev_goal_state == None:
                    self._goal_neighbors = self._get_neighbors(goal_state)
                    return goal_state
                else:
                    prev_x, prev_y = self._prev_goal_state
                    new_goal_x_dist = abs(goal_state[0]-prev_x)
                    new_goal_y_dist = abs(goal_state[1]-prev_y)
                    distance_check = new_goal_x_dist > self._new_goal_state_gap\
                        or new_goal_y_dist > self._new_goal_state_gap
                    if distance_check:
                        print(f'Previous goal: {self._prev_goal_state}')
                        print(f'New goal: {goal_state}')
                        self._goal_neighbors = self._get_neighbors(goal_state)
                        return goal_state
        n += 1
        raise ValueError('Failed to sample a goal state.')

    def _get_neighbors(self, state):
        actions = [[-1,0], [0,-1], [1,0], [0,1]]
        neighbors = []
        for action in actions:
            possible_neighbor = [state[0]+action[0], state[1]+action[1]]
            if self._layout[possible_neighbor[0], possible_neighbor[1]] == -1:
                continue
            neighbors.append(possible_neighbor)
        return neighbors

    def _make_cifar_images(self):
        def grayscale(image): # image is [channels, height, width]
            image = np.transpose(image, (1, 2, 0))
            weights = np.array([0.2989, 0.5870, 0.1140])
            grayscale = np.dot(image, weights)
            return grayscale

        transform = transforms.Compose([transforms.ToTensor()])
        testset = torchvision.datasets.CIFAR10(root='./cifar_data', train=False,
            download=True, transform=transform)
        image_indices = np.random.choice(
            len(testset), size=self._number_of_states, replace=False
            ).reshape(self._layout_dims)
        self._cifar_images = {}
        self._cifar_images_indices = {}
        for x in range(self._layout_dims[0]):
            self._cifar_images[x] = {}
            self._cifar_images_indices[x] = {}
            for y in range(self._layout_dims[1]):
                image_index = image_indices[x][y]
                image = testset[image_index][0]
                self._cifar_images[x][y] = grayscale(image)
                self._cifar_images_indices[x][y] = image_index

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
        if self._observation_type is ObservationType.GRID:
            return specs.Array(
                shape=(1,) + self._layout_dims, dtype=np.float32,
                name='observation_grid') # (C, H, W)
        elif self._observation_type is ObservationType.CIFAR:
            return specs.DiscreteArray(
                shape=(1,32,32), dtype=np.float32,
                name='cifar_image')

    def action_spec(self):
        return specs.DiscreteArray(4, dtype=int, name='action')

    def get_obs(self, state=None):
        if state == None:
            state = self._state
        if self._observation_type is ObservationType.GRID:
            goal_state = self._goal_state
            if self._shuffle_states:
                state = self._swap_map[str([state[0], state[1]])]
            obs = np.zeros((1,) + self._layout_dims, dtype=np.float32)
            obs[0, ...] = (self._layout < 0)*(-1)
            obs[0, state[0], state[1]] = 1
            if not self._hide_goal:
                obs[0, self._goal_state[0], self._goal_state[1]] = 5
            if self._shuffle_obs:
                obs = obs.flatten()
                obs = obs[self._shuffle_indices]
                obs = obs.reshape((1,) + self._layout_dims)
            return obs
        elif self._observation_type is ObservationType.CIFAR:
            img = self._cifar_images[state[0], state[1]]
            return img 

    def reset(self, reset_start=True, reset_goal=False):
        if reset_goal:
            self._prev_goal_state = self.goal_state
            self._new_goal_state_gap = min(min(self._layout_dims)//3, 3)
            goal_state = self._sample_goal()
            self.goal_state = goal_state
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
       
        # Transitions may be warped to test the predictions of the DiCarlo paper
        if self.transitions_swapped:
            if (self._state==self._b) and (action==self._bf_action):
                new_state = self._f
            elif (self._state==self._c) and (action==self._ce_action):
                new_state = self._e
            elif (self._state==self._e) and (action==self._ec_action):
                new_state = self._c
            elif (self._state==self._f) and (action==self._fb_action):
                new_state = self._b

        new_x, new_y = new_state
        if self._add_barrier:
            if self._layout[new_x, new_y] <= 0: # Barrier is only around goal
                blocked_by_barrier = False
            else:
                invalid_neighbors = self._goal_neighbors[1:]
                if [x,y] in invalid_neighbors:
                    blocked_by_barrier = True
                else:
                    blocked_by_barrier = False
        step_type = dm_env.StepType.MID
        if self._add_barrier and blocked_by_barrier:
            reward = self._penalty_for_walls
            discount = self._discount
            new_state = (x, y)
        elif self._layout[new_x, new_y] == -1:  # wall
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
            self._state[1], self._state[0], '\N{SMILING FACE WITH OPEN MOUTH}',
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

    def swap_transitions(self, swap_params):
        self._a = swap_params['a']
        self._b = swap_params['b']
        self._c = swap_params['c']
        self._d = swap_params['d']
        self._e = swap_params['e']
        self._f = swap_params['f']
        self._bf_action = self.invert_action(self._b, self._c)
        self._fb_action = self.invert_action(self._f, self._e)
        self._ec_action = self.invert_action(self._e, self._f)
        self._ce_action = self.invert_action(self._c, self._b)

    def get_timestep_pairs_of_swapped_transitions(self):
        pairs = []
        a = dm_env.TimeStep(
            step_type=dm_env.StepType.MID, reward=0., discount=self._discount,
            observation=self.get_obs(state=self._a))
        b = dm_env.TimeStep(
            step_type=dm_env.StepType.MID, reward=0., discount=self._discount,
            observation=self.get_obs(state=self._b))
        c = dm_env.TimeStep(
            step_type=dm_env.StepType.MID, reward=0., discount=self._discount,
            observation=self.get_obs(state=self._c))
        d = dm_env.TimeStep(
            step_type=dm_env.StepType.MID, reward=0., discount=self._discount,
            observation=self.get_obs(state=self._d))
        e = dm_env.TimeStep(
            step_type=dm_env.StepType.MID, reward=0., discount=self._discount,
            observation=self.get_obs(state=self._e))
        f = dm_env.TimeStep(
            step_type=dm_env.StepType.MID, reward=0., discount=self._discount,
            observation=self.get_obs(state=self._f))
        pairs.append([a, b, self.invert_action(self._b, self._a)])
        pairs.append([b, a, self.invert_action(self._a, self._b)])
        pairs.append([b, f, self._bf_action])
        pairs.append([f, b, self._fb_action])
        pairs.append([d, e, self.invert_action(self._e, self._d)])
        pairs.append([e, d, self.invert_action(self._d, self._e)])
        pairs.append([e, c, self._ec_action])
        pairs.append([c, e, self._ce_action])
        return pairs

    def invert_action(self, state1, state2):
        """ Returns the action responsible for transition from state1 to state2"""
        x_diff = state2[0] - state1[0]
        y_diff = state2[1] - state1[1]
        if x_diff == -1 and y_diff == 0:
            return 0 # left
        elif x_diff == 1 and y_diff == 0:
            return 1 # right
        elif x_diff == 0 and y_diff == -1:
            return 2 # up
        elif x_diff == 0 and y_diff == 1:
            return 3 #down
        else:
            raise ValueError('Not a valid transition')

def setup_environment(environment):
  """Returns the environment and its spec."""

  # Make sure the environment outputs single-precision floats.
  environment = wrappers.SinglePrecisionWrapper(environment)

  # Grab the spec of the environment.
  environment_spec = specs.make_environment_spec(environment)

  return environment, environment_spec
