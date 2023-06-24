import enum
import dm_env
import random
import collections
import numpy as np
from itertools import islice
import torch
from acme.utils import tree_utils

# Convenient container for the SARS tuples required by deep RL agents.
# Note: At time t, OBS is encoded into LATENT. The agent then selects an
# ACTION and transitions to timestep t+1. A REWARD may then be received, and
# NEXT_OBS is seen. next_obs may or may not be a TERMINAL state.
Transitions = collections.namedtuple(
    'Transitions',
    ['obs', 'latent', 'action', 'reward', 'discount', 'next_obs', 'terminal'])

class ReplayBuffer(object):
    """A simple Python replay buffer."""

    def __init__(self, capacity: int=None):
        self.buffer = collections.deque(maxlen=capacity)
        self._prev_obs = None

    def add_first(self, initial_timestep: dm_env.TimeStep):
        self._prev_obs = initial_timestep.observation

    def add(
        self, action: int, timestep: dm_env.TimeStep, latent: torch.tensor):

        if latent != None:
            latent = latent.cpu().numpy()

        transition = Transitions(
            obs=self._prev_obs, action=action.cpu().numpy(),
            reward=timestep.reward,
            discount=timestep.discount, next_obs=timestep.observation,
            terminal=timestep.last(),
            latent=latent)
        self.buffer.append(transition)
        self._prev_obs = timestep.observation

    def add_artificial_transition(
        self, timestep: dm_env.TimeStep, next_timestep: dm_env.TimeStep,
        action: int):
        transition = Transitions(
            obs=timestep.observation, action=action,
            reward=next_timestep.reward,
            discount=next_timestep.discount, next_obs=next_timestep.observation,
            terminal=next_timestep.last(),
            latent=None)
        self.buffer.append(transition)

    def sample_deque(self, indices):
        batch_as_list = [None for _ in range(indices.size)]
        for buffer_i, val in enumerate(self.buffer):
            if buffer_i in indices:
                list_indices = np.argwhere(indices==buffer_i)
                for list_i in list_indices:
                    batch_as_list[list_i[0]] = self.buffer[buffer_i]
        return batch_as_list

    def sample(
        self, batch_size: int, seq_len: int=1,
        respect_terminals: bool=False) -> Transitions:
        ''' Sample a random batch of Transitions as a list. '''

        n_items = len(self.buffer)
        if seq_len > 1:
            if respect_terminals:
                terminals = [b.terminal for b in self.buffer]
                n_terminals = len(terminals)
                valid_indices = []
                for test_idx in range(n_terminals-seq_len):
                    if np.sum(terminals[test_idx:test_idx+seq_len-2]) > 0:
                        continue
                    valid_indices.append(test_idx)
            else:
                valid_indices = n_items - seq_len
            start_indices = np.random.choice(valid_indices, size=batch_size)
            batch_as_list = [
                list(islice(self.buffer, i, i+seq_len)) for i in start_indices]
        else:
            start_indices = np.random.choice(n_items, size=batch_size)
            batch_as_list = [self.buffer[i] for i in start_indices]
        # Convert list of `batch_size` Transitions into a single Transitions
        # object where each field has `batch_size` stacked fields.
        stacked_batch = tree_utils.stack_sequence_fields(batch_as_list)
        return stacked_batch

    def flush(self) -> Transitions:
        entire_buffer = tree_utils.stack_sequence_fields(self.buffer)
        self.buffer.clear()
        return entire_buffer

    def is_ready(self, batch_size: int, seq_len: int) -> bool:
        return max(batch_size, seq_len) <= len(self.buffer)

