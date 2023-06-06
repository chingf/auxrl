import numpy as np
import torch
import torch.nn as nn
from itertools import accumulate
import inspect
import yaml
from pathlib import Path
import collections.abc

NN_MAP = {
    k: v for k, v in inspect.getmembers(nn) # if nonthrowing_issubclass(v, nn.Module)
}
HERE = Path(__file__).parent

def compute_feature_size(input_shape, convs):
    conv_output = convs(torch.zeros(1, *input_shape))
    return conv_output.view(1, -1).size(1)

def make_convs(input_shape, conv_config):
    convs = []
    for i, layer in enumerate(conv_config):
        if layer[0] == "Conv2d":
            if layer[1] == "auto":
                convs.append(NN_MAP[layer[0]](input_shape[0], layer[2], **layer[3]))
            else:
                convs.append(NN_MAP[layer[0]](layer[1], layer[2], **layer[3]))
        elif layer[0] == "MaxPool2d":
            convs.append(NN_MAP[layer[0]](**layer[1]))
        else:
            convs.append(NN_MAP[layer]())

    return nn.Sequential(*convs)

def make_fc(input_dim, out_dim, fc_config):
    fc = []
    for i, layer in enumerate(fc_config):
        if layer[0] == "Linear":
            if layer[1] == "auto" and layer[2] == "auto":
                fc.append(NN_MAP[layer[0]](input_dim, out_dim))
            elif layer[1] == "auto":
                fc.append(NN_MAP[layer[0]](input_dim, layer[2]))
            elif layer[2] == "auto":
                fc.append(NN_MAP[layer[0]](layer[1], out_dim))
            else:
                fc.append(NN_MAP[layer[0]](layer[1], layer[2]))
        elif layer[0] == "LSTM":
            return NN_MAP[layer[0]](layer[1], layer[2], batch_first=True)
        else:
            fc.append(NN_MAP[layer]())
    return nn.Sequential(*fc)

class Encoder(nn.Module):
    def __init__(self, env_spec, latent_dim, config, mem_len):
        super().__init__()
        self._env_spec = env_spec
        self._input_shape = env_spec.observations.shape # (C, H, W)
        self._latent_dim = latent_dim
        self._mem_len = mem_len
        if config['convs'] != None:
            self._convs = make_convs(self._input_shape, config['convs'])
            self._feature_size = compute_feature_size(
                self._input_shape, self._convs)
        else:
            self._convs = None
            self._feature_size = np.prod(self._input_shape)
        if self._mem_len > 0:
            self._feature_size += self._mem_len*self._latent_dim
        self._prev_latent = None
        self._fc = make_fc(self._feature_size, self._latent_dim, config['fc'])
        
    def forward(self, x, prev_zs=None):
        """
        prev_zs is shape (mem_len, latent_dim)
        """

        N, C, H, W = x.shape
        if self._convs is not None:
            x = self._convs(x)
        x = x.view(N, -1)
        x = x.float()
        prev_zs_provided = prev_zs != None
        if self._mem_len > 0:
            if not prev_zs_provided:
                if self._prev_latent == None:
                    self._prev_latent = torch.zeros(
                        N, self._mem_len, self._latent_dim)
                    self._prev_latent = self._prev_latent.to(x.get_device())
                prev_zs = self._prev_latent
            prev_zs = prev_zs
            x = torch.hstack((x, prev_zs.reshape(N, -1)))
        x = self._fc(x)
        if (self._mem_len > 0) and (not prev_zs_provided):
            self._prev_latent = torch.hstack((
                self._prev_latent[:,1:], x.unsqueeze(1)))
        return x

    def get_curr_latent(self):
        return self._prev_latent

    def reset(self):
        self._prev_latent = None

class T(nn.Module):
    def __init__(
        self, env_spec, latent_dim,  config,
        encode_new_state=False, predict_z=True):

        super().__init__()
        self._env_spec = env_spec
        self._n_actions = env_spec.actions.num_values
        self._latent_dim = latent_dim
        self._encode_new_state = encode_new_state
        self._predict_z = predict_z
        if predict_z:
            self._output_shape = latent_dim
        else:
            obs_shape = env_spec.observations.shape[::-1] # (C, H, W)
            self._output_shape = np.prod(obs_shape)
        self._fc = make_fc(
            latent_dim + self._n_actions, self._output_shape, config['fc'])
    
    def forward(self, x):
        tr = self._fc(x.float())
        if self._encode_new_state or (not self._predict_z):
            return tr
        else:
            return x[:, :self._latent_dim] + tr

class Q(nn.Module):
    def __init__(self, env_spec, latent_dim, config):
        super().__init__()
        self._env_spec = env_spec
        self._n_actions = env_spec.actions.num_values
        self._latent_dim = latent_dim
        self._fc = make_fc(latent_dim, self._n_actions, config['fc'])

    def forward(self, x):
        x = self._fc(x)
        return x

