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
        elif layer[0] == 'Identity':
            fc.append(nn.Identity())
        else:
            fc.append(NN_MAP[layer]())
    return nn.Sequential(*fc)

class Encoder(nn.Module):
    def __init__(
        self, env_spec, latent_dim, config, mem_len,
        eligibility_gamma=1., mem_location=0
        ):

        super().__init__()
        self._env_spec = env_spec
        self._input_shape = env_spec.observations.shape # (C, H, W)
        self._latent_dim = latent_dim
        self._mem_len = mem_len
        self._eligibility_gamma = eligibility_gamma
        self._mem_location = mem_location
        if config['convs'] != None:
            self._convs = make_convs(self._input_shape, config['convs'])
            self._feature_size = compute_feature_size(
                self._input_shape, self._convs)
        else:
            self._convs = None
            self._feature_size = np.prod(self._input_shape)
        self._prev_latent = None
        self._fc = make_fc(self._feature_size, self._latent_dim, config['fc'])
        if self._mem_len>0:
            layer_str = str(self._fc[self._mem_location])
            if ('Linear' not in layer_str) and ('Identity' not in layer_str):
                raise ValueError('Memory location not a linear/I layer.')
        
    def forward(self, x, prev_latents=None, save_conv_activity=False):
        """
        prev_latents is shape (mem_len, latent_dim)
        """

        N, C, H, W = x.shape
        prev_latents_provided = prev_latents != None

        # Run through convolutional layer
        if self._convs is not None:
            x = self._convs(x)
            if save_conv_activity:
                self._prev_conv_activity = x.detach()
        x = x.view(N, -1)
        x = x.float()

        # Run through MLP, where there may be residual latents added.
        # Without residual latents, it would be x = self._fc(x)
        for idx in range(len(self._fc)):
            mlp_layer = self._fc[idx]
            if (self._mem_len > 0) and (self._mem_location==idx):
                x_device = x.get_device()
                if not prev_latents_provided:
                    if self._prev_latent == None:
                        self._prev_latent = torch.zeros(N, self._mem_len, x.shape[-1]) 
                        if x_device != -1:
                            self._prev_latent = self._prev_latent.to(x_device)
                    prev_latents = self._prev_latent

                for t in range(self._mem_len):
                    scaling = self._eligibility_gamma**(t+1)
                    prev_latents[:, t, :] = prev_latents[:, t, :] * scaling
                prev_latents = torch.sum(prev_latents, dim=1) #.detach() #TODO
                new_latent = x.clone()
                self._new_latent = new_latent
                x = x + prev_latents # Residual adding
            x = mlp_layer(x)

        # Store the current latent state
        if (self._mem_len > 0) and (not prev_latents_provided):
            #if self._mem_len == 1:
            #    self._prev_latent = torch.hstack((
            #        self._prev_latent, new_latent.unsqueeze(1)))
            #else:
            self._prev_latent = torch.hstack((
                self._prev_latent[:,1:], new_latent.unsqueeze(1)))
        return x

    def get_curr_latent(self):
        return self._prev_latent

    def reset(self):
        self._prev_latent = None

class RecurrentConcatenationEncoder(nn.Module):
    def __init__(
        self, env_spec, latent_dim, config, mem_len, eligibility_gamma=None
        ):

        super().__init__()
        self._env_spec = env_spec
        self._input_shape = env_spec.observations.shape # (C, H, W)
        self._latent_dim = latent_dim
        self._mem_len = mem_len
        self._eligibility_gamma = eligibility_gamma
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
        
    def forward(self, x, prev_latents=None, save_conv_activity=False):
        """
        prev_latents is shape (mem_len, latent_dim)
        """

        N, C, H, W = x.shape
        if self._convs is not None:
            x = self._convs(x)
            if save_conv_activity:
                self._prev_conv_activity = x.detach()
        x = x.view(N, -1)
        x = x.float()
        prev_latents_provided = prev_latents != None
        if self._mem_len > 0:
            if not prev_latents_provided:
                if self._prev_latent == None:
                    self._prev_latent = torch.zeros(
                        N, self._mem_len, self._latent_dim)
                    x_device = x.get_device()
                    if x_device != -1:
                        self._prev_latent = self._prev_latent.to(x_device)
                prev_latents = self._prev_latent

            if self._eligibility_gamma != None:
                for t in range(self._mem_len):
                    scaling = self._eligibility_gamma**(t+1)
                    prev_latents[:, t, :] = prev_latents[:, t, :] * scaling

            x = torch.hstack((x, prev_latents.reshape(N, -1)))
        x = self._fc(x)
        if (self._mem_len > 0) and (not prev_latents_provided):
            self._prev_latent = torch.hstack((
                self._prev_latent[:,1:], x.unsqueeze(1)))
        self._new_latent = x
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

### DISTRIBUTIONAL RL SECTION: IMPLICIT QUANTILE NETWORKS ###

class IQN(nn.Module):
    def __init__(
            self, env_spec, latent_dim, config,
            quantile_embed_dim=64, random_quantiles=True, n_quantiles=8
            ):

        super().__init__()
        self._env_spec = env_spec
        self._n_actions = env_spec.actions.num_values
        self._latent_dim = latent_dim
        self._quantile_embed_dim = quantile_embed_dim
        self._random_quantiles = random_quantiles
        self._n_quantiles = n_quantiles
        self._pis = torch.FloatTensor([np.pi*i for i in range(self._quantile_embed_dim)])
        self._pis = self._pis.view(1, 1, self._quantile_embed_dim)

        # The two networks used in the IQN Module
        self._quantile_embed_net = nn.Sequential(
            nn.Linear(self._quantile_embed_dim, self._latent_dim), nn.ReLU())
        self._iqn_net = make_fc(latent_dim, self._n_actions, config['fc'])

    def forward(self, x, n_quantiles=None):
        if n_quantiles is None:
            n_quantiles = self._n_quantiles
        device = x.get_device()
        if device == -1: device = 'cpu'
        batch_size = x.shape[0]
        if self._random_quantiles:
            quantiles = torch.rand(batch_size, n_quantiles).to(device).unsqueeze(-1)
        else:
            quantiles = torch.linspace(0.05, 0.95, n_quantiles).repeat(batch_size, 1)
            quantiles = quantiles.to(device).unsqueeze(-1)
        quantile_embeddings = self._quantile_embed_net(
            torch.cos(quantiles*self._pis.to(device))) # (batch, n_quantiles, latent)
        x = x.unsqueeze(1).repeat(1, n_quantiles, 1)
        quantile_values = self._iqn_net(x * quantile_embeddings)
        return quantile_values, quantiles

### ON-POLICY ACTOR CRITIC MODEL ###

class A2C(nn.Module):
    def __init__(self, env_spec, latent_dim, config):
        super().__init__()
        self._env_spec = env_spec
        self._n_actions = env_spec.actions.num_values
        self._latent_dim = latent_dim
        self.actor = Actor(latent_dim, self._n_actions, config['fc'])
        self.critic = Critic(latent_dim, config['fc'])

    def forward(self, x):
        policy_dist = self.actor(x)
        value = self.critic(x)
        return policy_dist, value

class Actor(nn.Module):
    def __init__(self, input_size, action_size, config):
        super(Actor, self).__init__()
        self.fc = make_fc(input_size, action_size, config)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc(x))
        x = torch.nn.functional.softmax(x, dim=-1)
        return x

class Critic(nn.Module):
    def __init__(self, input_size, config):
        super(Critic, self).__init__()
        self.fc = make_fc(input_size, 1, config)

    def forward(self, x):
        x = self.fc(x)
        return x


