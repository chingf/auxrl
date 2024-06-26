import os
import numpy as np
import inspect
import yaml
import warnings
from pathlib import Path
import torch
import torch.nn as nn
from copy import deepcopy
from auxrl.networks.Modules import Encoder, IQN, T, RecurrentConcatenationEncoder

NETWORK_DIR = os.path.dirname(os.path.abspath(__file__))

def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

class Network(object):
    """
    Container over the various computational modules
    """

    def __init__(
        self, env_spec, latent_dim, network_yaml, yaml_mods={},
        device=torch.device('cpu'), freeze_encoder=False,
        random_quantiles=True, n_quantiles=8):

        self._env_spec = env_spec
        self._n_actions = env_spec.actions.num_values
        self._latent_dim = latent_dim
        self._network_yaml = network_yaml
        self._yaml_mods = yaml_mods
        self._device = device
        self._freeze_encoder = freeze_encoder
        self._random_quantiles = random_quantiles
        self._n_quantiles = n_quantiles

        # Load and update yaml file
        with open(f'{NETWORK_DIR}/yamls/{self._network_yaml}.yaml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            update(config, self._yaml_mods)

        self.encoder = Encoder(
            env_spec, latent_dim, config['encoder'], mem_len=0).to(device)
        if self._freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.Q = IQN(
            env_spec, latent_dim, config['iqn'],
            random_quantiles=random_quantiles, n_quantiles=n_quantiles
            ).to(device)
        self.T = T(env_spec, latent_dim, config['t']).to(device)

    def get_params(self):
        params = {
            'encoder': self.encoder.state_dict(),
            'Q': self.Q.state_dict(), 'T': self.T.state_dict()
            }
        return params

    def set_params(self, params, encoder_only=False, shuffle=False):
        if shuffle:
            modules = ['encoder', 'Q', 'T']
            for m in modules:
                for k in params[m].keys():
                    idx = torch.randperm(params[m][k].nelement())
                    params[m][k] = params[m][k].view(-1)[
                        idx].view(params[m][k].size())

        self.encoder.load_state_dict(params['encoder'])
        if not encoder_only:
            self.Q.load_state_dict(params['Q'])
            self.T.load_state_dict(params['T'])

    def get_trainable_params(self):
        trainable_params = []
        trainable_params.extend(self.Q.parameters())
        trainable_params.extend(self.T.parameters())
        if not self._freeze_encoder:
            trainable_params.extend(self.encoder.parameters())
        return trainable_params

    def get_encoder_params(self):
        return self.encoder.parameters()

    def copy(self):
        duplicate = Network(
            self._env_spec, self._latent_dim, self._network_yaml, self._yaml_mods,
            self._device, self._freeze_encoder)
        return duplicate

