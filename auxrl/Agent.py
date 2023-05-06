import gym
import enum
import time
import acme
import torch
import dm_env
import random
import warnings
import itertools
import collections

import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn.functional import mse_loss

from acme import specs
from acme import wrappers
from acme.utils import tree_utils
import copy

from auxrl.networks.Network import Network
from auxrl.ReplayBuffer import ReplayBuffer

Transitions = collections.namedtuple(
    'Transitions', ['state', 'action', 'reward', 'discount', 'next_state'])

class Agent(acme.Actor):

    def __init__(self,
        env_spec: specs.EnvironmentSpec, network: Network,
        loss_weights: list=[0,0,0,1], lr: float=1e-4,
        pred_len: int=1, pred_gamma: float=0.,
        replay_capacity: int=1_000_000, epsilon: float=1.,
        batch_size: int=32, target_update_frequency: int=1000,
        device: torch.device=torch.device('cpu'), train_seq_len: int=1,
        entropy_temp: int=5, discount_factor: float=0.9):

        self._env_spec = env_spec
        self._loss_weights = loss_weights
        self._n_actions = env_spec.actions.num_values
        self._pred_len = pred_len
        self._pred_gamma = pred_gamma
        self._mem_len = network._mem_len
        self._replay_seq_len = pred_len
        if network._mem_len > 0:
            self._replay_seq_len += max(network._mem_len + train_seq_len)
        # Initialize networks
        self._network = network
        self._target_network = network.copy()
        self._replay_buffer = ReplayBuffer(replay_capacity)
        # Store training parameters
        self._epsilon = epsilon
        self._batch_size = batch_size
        self._lr = lr
        self._target_update_frequency = target_update_frequency
        self._device = device
        self._entropy_temp = entropy_temp
        self._discount_factor = discount_factor
        self._train_seq_len = train_seq_len
        # Initialize optimizer
        self._optimizer = torch.optim.Adam(
            self._network.get_trainable_params(), lr=lr)
        self._n_updates = 0

        if (network._mem_len > 0) and (self._pred_len > 1):
            raise ValueError('Incompatible mem/pred len')

    def reset(self):
        self._network.encoder.reset()
        self._target_network.encoder.reset()

    def select_action(self, obs, force_greedy=False):
        """ Epsilon-greedy action selection. """

        with torch.no_grad():
            z = self._network.encoder(
                torch.tensor(obs).unsqueeze(0).to(self._device))
            q_values = self._network.Q(z)
        q_values = q_values.squeeze(0).detach()
        if force_greedy or (self._epsilon < torch.rand(1)):
            action = q_values.argmax(axis=-1)
        else:
            action = torch.randint(
                low=0, high=self._n_actions , size=(1,), dtype=torch.int64)
        return action

    def get_curr_latent(self):
        return self._network.encoder.get_curr_latent()

    def update(self):
        if not self._replay_buffer.is_ready(
            self._batch_size, self._replay_seq_len):
            return [0,0,0,0,0]
        device = self._device
        self._optimizer.zero_grad()
        # Sample a minibatch of transitions from experience replay.
        transitions = self._replay_buffer.sample(
            self._batch_size, self._replay_seq_len)

        # If a sequence, you may need to do some formattingf #TODO
        if self._replay_seq_len > 1:
            actions_val = actions_val[:, 0]
            rewards_val = rewards_val[:, 0]
            terminals_val = terminals_val[:, 0]
            T_states_val = torch.as_tensor(
                states_val, device=self.device).float()
            T_next_states_val = torch.as_tensor(
                next_states_val, device=self.device).float()
            states_val = states_val[:,:1].squeeze(1)
            next_states_val = next_states_val[:,:1].squeeze(1)

        # Unpack transition information
        obs = torch.tensor(transitions.obs.astype(np.float32)).to(device) # (N,C,H,W)
        a = transitions.action.astype(int) # (N,1)
        r = torch.tensor(
            transitions.reward.astype(np.float32)).view(-1,1).to(device) # (N,1)
        next_obs = torch.tensor(
            transitions.next_obs.astype(np.float32)).to(device)
        terminal = torch.tensor(transitions.terminal.astype(np.float32)) # (N,)
        onehot_actions = np.zeros((self._batch_size, self._n_actions))
        onehot_actions[np.arange(self._batch_size), a] = 1
        onehot_actions = torch.as_tensor(onehot_actions, device=self._device).float()

        # Burn in latents for POMDP
        if self._mem_len > 0: # TODO
            import pdb; pdb.set_trace()
            saved_z = torch.as_tensor(
                transitions.latent, device=device).float() # (N, 1, Z)
            _zs = torch.tensor(zs[:, :self._mem_len]).to(device).float()
            for t in range(self._mem_len, states_val.shape[1]):
                Es = self.crar.encoder(states_val[:, t], zs=_zs)
                _zs = torch.hstack((_zs[:,1:], Es.unsqueeze(1)))
            Esp = self.crar.encoder(next_states_val[:,-1], zs=_zs)
        else:
            next_z = self._network.encoder(next_obs)
            z = self._network.encoder(obs)
        z_and_action = torch.cat([z, onehot_actions], dim=1)
        Tz = self._network.T(z_and_action)

        # Positive Sample Loss (transition predictions)
        T_target = next_z
        pred_gamma = self._pred_gamma
        for t in np.arange(1, self._pred_len): # TODO
            import pdb; pdb.set_trace()
            _obs = pred_next_obs[:,t:t+1].squeeze(1)
            _z = self._network.encoder(_obs)
            T_target = T_target + (sr_gamma**t) * _z
        loss_pos_sample = torch.norm(Tz - T_target, dim=1)
        terminal_mask = (1-terminal).to(device)
        loss_pos_sample = torch.mean(loss_pos_sample * terminal_mask)

        # Negative Sample Loss (entropy)
        rolled = torch.roll(z, 1, dims=0)
        loss_neg_random = torch.mean(torch.exp(
            -self._entropy_temp * torch.norm(z - rolled, dim=1)
            ))
        loss_neg_neighbor = torch.mean(torch.exp(
            -self._entropy_temp * torch.norm(z - next_z, dim=1)
            ))

        # Q Loss
        if self._mem_len > 0: # TODO
            raise ValueError('Proper POMDP latents not yet done.')
            _zs = torch.tensor(zs[:,:self._mem_len]).to(self.device)
            for t in range(self._mem_len, states_val.shape[1]):
                Esp_target = self.crar_target.encoder(states_val[:, t], zs=_zs)
                _zs = torch.hstack((_zs[:,1:], Esp_target.unsqueeze(1)))
            Esp_target = self.crar_target.encoder(
                next_states_val[:,-1], zs=_zs)
            next_q_target = self.crar_target.Q(Esp_target)
        else:
            target_next_q = self._target_network.Q( # (N, A)
                self._target_network.encoder(next_obs))
            next_q = self._network.Q(next_z)
            argmax_next_q = torch.argmax(next_q, axis=1)
            max_next_q = target_next_q[
                np.arange(self._batch_size), argmax_next_q].reshape((-1,1))
        Q_target = r + terminal_mask*self._discount_factor*max_next_q
        q_vals = self._network.Q(z)
        q_vals = q_vals[np.arange(self._batch_size), a]
        loss_Q = torch.mean(torch.norm(q_vals - Q_target, dim=1))

        # Aggregate all losses and update parameters
        all_losses = self._loss_weights[0] * loss_pos_sample \
            + self._loss_weights[1] * loss_neg_neighbor \
            + self._loss_weights[2] * loss_neg_random \
            + self._loss_weights[3] * loss_Q
        all_losses.backward()
        self._optimizer.step()
        self._n_updates += 1

        # Update target network if needed
        if self._n_updates % self._target_update_frequency == 0:
            self._target_network.set_params(self._network.get_params())

        return [
            loss_pos_sample.item(), loss_neg_neighbor.item(),
            loss_neg_random.item(), loss_Q.item(), all_losses.item()]

    def observe_first(self, timestep: dm_env.TimeStep):
        self._replay_buffer.add_first(timestep)

    def observe(
        self, action: int, next_timestep: dm_env.TimeStep,
        latent: torch.tensor):
        """
        If NEXT_TIMESTEP is at time t+1, then LATENT corresponds to the
        observation from time t.
        """

        self._replay_buffer.add(action, next_timestep, latent)

    def save_network(self, path, episode=None):
        network_params = self._network.get_params()
        file_suffix = '' if episode == None else f'_ep{episode}'
        torch.save(network_params, f'{path}network{file_suffix}.pth')

    def load_network(self, path, episode=None, encoder_only=True):
        file_suffix = '' if episode == None else f'_ep{episode}'
        network_params = torch.load(f'{path}network{file_suffix}.pth')
        self._network.set_params(network_params, encoder_only)


