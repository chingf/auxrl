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
        pred_TD: bool=False, pred_gamma: float=0.,
        pred_len: int=1, pred_scale: bool=False, # Only used if not pred_TD,
        replay_capacity: int=1_000_000, epsilon: float=1.,
        batch_size: int=32, target_update_frequency: int=1000,
        device: torch.device=torch.device('cpu'), train_seq_len: int=1,
        entropy_temp: int=5, discount_factor: float=0.9,
        respect_terminals: bool=False):

        self._env_spec = env_spec
        self._loss_weights = loss_weights
        self._n_actions = env_spec.actions.num_values
        self._pred_TD = pred_TD
        self._pred_gamma = pred_gamma
        self._pred_scale = pred_scale
        self._pred_len = pred_len
        self._mem_len = network._mem_len
        self._replay_seq_len = pred_len
        if network._mem_len > 0:
            if self._pred_len > 1:
                raise ValueError('Incompatible mem/pred len.')
            if train_seq_len < self._mem_len:
                raise ValueError('Training length is too short.')
            self._replay_seq_len += train_seq_len
        if self._pred_TD:
            if self._replay_seq_len != 2:
                warnings.warn('Overriding replay length to 2 for predictive TD loss!')
            self._pred_len = self._replay_seq_len = 2
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
        self._respect_terminals = respect_terminals
        # Initialize optimizer
        self._optimizer = torch.optim.Adam(
            self._network.get_trainable_params(), lr=lr)
        self._n_updates = 0

    def reset(self):
        self._network.encoder.reset()
        self._target_network.encoder.reset()

    def select_action(
        self, obs, force_greedy=False, verbose=False, return_latent=False
        ):
        """ Epsilon-greedy action selection. """

        with torch.no_grad():
            z = self._network.encoder(
                torch.tensor(obs).unsqueeze(0).to(self._device))
            q_values = self._network.Q(z)
        q_values = q_values.squeeze(0).detach()
        if force_greedy or (self._epsilon < torch.rand(1)):
            if verbose: print(q_values)
            action = q_values.argmax(axis=-1)
        else:
            action = torch.randint(
                low=0, high=self._n_actions , size=(1,), dtype=torch.int64)
        if return_latent:
            return action, z
        else:
            return action

    def get_curr_latent(self):
        return self._network.encoder.get_curr_latent()

    def update(self, clip_norm=-1):
        if not self._replay_buffer.is_ready(
            self._batch_size, self._replay_seq_len):
            return [0,0,0,0,0]
        device = self._device
        self._optimizer.zero_grad()
        transitions_seq = self._replay_buffer.sample(
            self._batch_size, self._replay_seq_len, self._respect_terminals)
        if self._replay_seq_len > 1:
            transitions = transitions_seq[-1]
        else:
            transitions = transitions_seq

        # Unpack transition information
        obs = torch.tensor(transitions.obs.astype(np.float32)).to(device) # (N,C,H,W)
        a = transitions.action.astype(int) # (N,1)
        r = torch.tensor(
            transitions.reward.astype(np.float32)).view(-1,1).to(device) # (N,1)
        next_obs = torch.tensor(
            transitions.next_obs.astype(np.float32)).to(device)
        onehot_actions = np.zeros((self._batch_size, self._n_actions))
        onehot_actions[np.arange(self._batch_size), a.squeeze()] = 1
        onehot_actions = torch.as_tensor(onehot_actions, device=self._device).float()

        # Burn in latents for POMDP
        if self._mem_len > 0:
            _zs = np.array(transitions_seq[self._mem_len].latent.astype(np.float32))
            _zs = torch.as_tensor(_zs).squeeze(1) # (mem_len, N, Z)
            _zs = _zs.to(device)

            for t in range(self._mem_len, self._replay_seq_len):
                _obs_t = torch.tensor( # (N,C,H,W)
                    transitions_seq[t].obs.astype(np.float32)).to(device)
                z = self._network.encoder(_obs_t, prev_zs=_zs)
                _zs = torch.hstack((_zs[:,1:], z.unsqueeze(1)))
            next_z = self._network.encoder(next_obs, prev_zs=_zs)
        else:
            next_z = self._network.encoder(next_obs)
            z = self._network.encoder(obs)
        z_and_action = torch.cat([z, onehot_actions], dim=1)
        Tz = self._network.T(z_and_action)

        # Positive Sample Loss (transition predictions)
        if self._pred_TD:
            _obs = torch.tensor(transitions_seq[0].obs.astype(np.float32))
            _next_obs = torch.tensor(transitions_seq[0].next_obs.astype(np.float32))
            _a = transitions_seq[0].action.astype(int)
            _z = self._network.encoder(_obs.to(device))
            _next_z = self._network.encoder(_next_obs.to(device))
            _onehot_actions = np.zeros((self._batch_size, self._n_actions))
            _onehot_actions[np.arange(self._batch_size), _a.squeeze()] = 1
            _onehot_actions = torch.as_tensor(
                _onehot_actions, device=self._device).float()
            _z_and_action = torch.cat([_z, _onehot_actions], dim=1)
            _Tz = self._network.T(_z_and_action)
            with torch.no_grad():
                _next_Tz = self._network.T(z_and_action)
            T_target = _next_z + self._pred_gamma * _next_Tz
            loss_pos_sample = torch.nn.functional.mse_loss(
                _Tz, T_target, reduction='none')
            loss_pos_sample = torch.mean(loss_pos_sample)
        else:
            T_target = next_z
            loss_pos_sample = torch.nn.functional.mse_loss(
                Tz, T_target, reduction='none')
            loss_pos_sample = torch.mean(loss_pos_sample)

        # Negative Sample Loss (entropy)
        rolled = torch.roll(z, 1, dims=0)
        loss_neg_random = torch.mean(torch.exp(
            -self._entropy_temp * torch.norm(z - rolled, dim=1)
            ))
        loss_neg_neighbor = torch.mean(torch.exp(
            -self._entropy_temp * torch.norm(z - next_z, dim=1)
            ))

        # Q Loss and target network update
        if self._n_updates%self._target_update_frequency == 0:
            self._target_network.set_params(self._network.get_params())
        if self._mem_len > 0:
            _zs = np.array(transitions_seq[self._mem_len].latent.astype(np.float32))
            _zs = torch.as_tensor(_zs).squeeze(1) # (mem_len, N, Z)
            _zs = _zs.to(device)
            for t in range(self._mem_len, self._replay_seq_len):
                _obs_t = torch.tensor( # (N,C,H,W)
                    transitions_seq[t].obs.astype(np.float32)).to(device)
                _zs = torch.hstack((
                    _zs[:,1:],
                    self._target_network.encoder(_obs_t, prev_zs=_zs).unsqueeze(1)
                    ))
            target_next_z = self._target_network.encoder(next_obs, prev_zs=_zs)
        else:
            target_next_z = self._target_network.encoder(next_obs) # (N, z)
        target_next_q = self._target_network.Q(target_next_z)  # (N, a)
        next_q = self._network.Q(next_z)
        argmax_next_q = torch.argmax(next_q, axis=1)
        max_next_q = target_next_q[
            np.arange(self._batch_size), argmax_next_q].reshape((-1,1))
        Q_target = r + self._discount_factor*max_next_q
        q_vals = self._network.Q(z)
        q_vals = q_vals[np.arange(self._batch_size), a.squeeze()]
        loss_Q = torch.mean(
            torch.nn.functional.mse_loss(q_vals, Q_target.squeeze(), reduction='none'))

        # Aggregate all losses and update parameters
        all_losses = self._loss_weights[0] * loss_pos_sample \
            + self._loss_weights[1] * loss_neg_neighbor \
            + self._loss_weights[2] * loss_neg_random \
            + self._loss_weights[3] * loss_Q
        all_losses.backward()
        if clip_norm != -1:
            nn.utils.clip_grad_norm_(
                self._network.get_encoder_params(), clip_norm)
        self._optimizer.step()
        self._n_updates += 1

        # Update target network if needed
        if self._n_updates % self._target_update_frequency == 0:
            print(f'Q loss at step {self._n_updates}: {loss_Q.item()}')

        return [
            self._loss_weights[0]*loss_pos_sample.item(),
            self._loss_weights[1]*loss_neg_neighbor.item(),
            self._loss_weights[2]*loss_neg_random.item(),
            self._loss_weights[3]*loss_Q.item(), all_losses.item()]

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

    def load_network(
        self, path, episode=None, encoder_only=True, shuffle=False):
        file_suffix = '' if episode == None else f'_ep{episode}'
        try:
            network_params = torch.load(f'{path}network{file_suffix}.pth')
        except:
            network_params = torch.load(
                f'{path}network{file_suffix}.pth', map_location=torch.device('cpu')
                )
        self._network.set_params(network_params, encoder_only, shuffle=shuffle)

