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

class Agent(acme.Actor):

    def __init__(self,
        env_spec: specs.EnvironmentSpec, network: Network,
        loss_weights: list=[0,0,0,1], lr: float=1e-4,
        device: torch.device=torch.device('cpu'), discount_factor: float=0.9
        ):

        self._env_spec = env_spec
        self._loss_weights = loss_weights
        self._n_actions = env_spec.actions.num_values
        self._network = network
        self._lr = lr
        self._device = device
        self._discount_factor = discount_factor
        # Initialize optimizer
        self._optimizer = torch.optim.Adam(
            self._network.get_trainable_params(), lr=lr)
        self._n_updates = 0

    def reset(self):
        self._network.encoder.reset()

    def actor_critic(self, obs):
        z = self._network.encoder(
            torch.tensor(obs).unsqueeze(0).to(self._device))
        policy_dist, value = self._network.A2C(z)
        return policy_dist, value

    def select_action(
        self, obs, force_greedy=False, verbose=False, return_latent=False
        ):
        """ action selection. """

        with torch.no_grad():
            z = self._network.encoder(
                torch.tensor(obs).unsqueeze(0).to(self._device))
            policy_dist, _ = self._network.A2C(z)
        if force_greedy:
            if verbose: print(policy_dist)
            action = policy_dist.argmax(axis=-1) # TODO: doublecheck
        else:
            action = policy_dist.multinomial(num_samples=1).item()
        if return_latent:
            return action, z
        else:
            return action

    def get_curr_latent(self):
        return self._network.encoder.get_curr_latent()

    def update(self, trajectory_info):
        # Unpack dict of trajectory information
        log_probs = trajectory_info['log_probs']
        values = trajectory_info['values']
        rewards = trajectory_info['rewards']
        masks = trajectory_info['masks']
        entropies = trajectory_info['entropies']
        obs = trajectory_info['obs']
        actions = trajectory_info['actions']
        next_obs = trajectory_info['next_obs']
        n_steps = len(values)

        # Set up optimizer
        device = self._device
        self._optimizer.zero_grad()

        # A2C loss
        Qvals = np.zeros_like(values)
        Qval = 0
        for step in reversed(range(len(rewards))):
            Qval = rewards[step] + self._discount_factor * Qval * masks[step]
            Qvals[step] = Qval
        values = torch.tensor(values, device=self._device) # (n_steps,)
        Qvals = torch.tensor(Qvals, device=self._device) # (n_steps,)
        log_probs = torch.stack(log_probs) # (n_steps,)
        entropies = torch.stack(entropies).mean() # scalar
        advantage = Qvals - values # (n_steps,)
        actor_loss = (-log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        print('---')
        print(actor_loss)
        print(critic_loss)
        print(entropies)
        print('---')
        a2c_loss = actor_loss + critic_loss - 0.00001 * entropies

        # Positive sample loss
        onehot_actions = np.zeros((n_steps, self._n_actions))
        onehot_actions[np.arange(n_steps), actions] = 1
        onehot_actions = torch.as_tensor(onehot_actions, device=self._device).float()
        next_obs = torch.tensor(np.array(next_obs), device=self._device)
        obs = torch.tensor(np.array(obs), device=self._device)
        next_z = self._network.encoder(next_obs)
        z = self._network.encoder(obs)
        z_and_action = torch.cat([z, onehot_actions], dim=1)
        Tz = self._network.T(z_and_action)
        T_target = next_z
        loss_pos_sample = torch.nn.functional.mse_loss(
            Tz, T_target, reduction='none')
        loss_pos_sample = torch.mean(loss_pos_sample)

        # Negative Sample Loss
        rolled = torch.roll(z, 1, dims=0)
        loss_neg_random = torch.mean(torch.exp(-5*torch.norm(z - rolled, dim=1)))
        loss_neg_neighbor = torch.mean(torch.exp(-5*torch.norm(z - next_z, dim=1)))

        # Aggregate all losses and update parameters
        all_losses = self._loss_weights[0] * loss_pos_sample \
            + self._loss_weights[1] * loss_neg_neighbor \
            + self._loss_weights[2] * loss_neg_random \
            + self._loss_weights[3] * a2c_loss
        all_losses.backward()
        self._optimizer.step()
        self._n_updates += 1

        return [
            self._loss_weights[0]*loss_pos_sample.item(),
            self._loss_weights[1]*loss_neg_neighbor.item(),
            self._loss_weights[2]*loss_neg_random.item(),
            self._loss_weights[3]*a2c_loss.item(), all_losses.item()]

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

    def observe_first(self, timestep: dm_env.TimeStep):
        return

    def observe(
        self, action: int, next_timestep: dm_env.TimeStep,
        latent: torch.tensor):
        return
