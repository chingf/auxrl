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

def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

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

class NN():
    """
    Deep Q-learning network using Keras
    
    Parameters
    -----------
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent
    input_dimensions :
    n_actions :
    random_state : numpy random number generator
    """
    def __init__(
        self, batch_size, input_dimensions, n_actions,
        random_state, device, yaml='basic', yaml_mods={}, mem_len=1, **kwargs
        ):

        self._input_dimensions=input_dimensions
        self._batch_size=batch_size
        self._random_state=random_state
        self._n_actions=n_actions
        self.device = device
        self.ddqn_only = kwargs.get('ddqn_only', False)
        self._yaml = yaml
        self._yaml_mods = yaml_mods
        self._mem_len = mem_len
        self._encoder_type = kwargs.get('encoder_type', None)
        self.internal_dim = kwargs["internal_dim"]
        self.encoder = self.encoder_model().to(self.device)
        self.Q = self.Q_model().to(self.device)
        self.transition = self.transition_model().to(self.device)
        self.models = [self.encoder, self.transition, self.Q]
        self.params = []
        for model in self.models:
            self.params.extend([p for p in model.parameters()])

    def encoder_model(self): # MODULE
        """ Instantiate a Keras model for the encoder of the CRAR learning algorithm.
        
        The model takes the following as input 
        s : list of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        
        Parameters
        -----------
        
    
        Returns
        -------
        Keras model with output x (= encoding of s)
    
        """

        class Encoder(nn.Module):
            def __init__(
                self, input_shape, fc, convs=None, abstract_dim=2, mem_len=0
                ):
                super().__init__()
                self.input_shape = input_shape
                self.convs = convs
                self.fc = fc
                self.abstract_dim = abstract_dim
                self.mem_len = mem_len

            def forward(self, x, zs=None):
                n_batches, h, w = x.shape
                if self.convs is not None:
                    x = x.unsqueeze(1) # Add singular channel
                    x = self.convs(x)
                x = x.view(n_batches, -1)
                x = x.float()
                #if self.mem_len > 0: # TODO
                #    zs = zs.view(n_batches, -1)
                #    zs = zs *0
                #    x = torch.hstack((x, zs))
                x = self.fc(x)
                return x

        class EncoderVariational(nn.Module):
            def __init__(
                self, input_shape, fc_mu, fc_var, abstract_dim=2
                ):
                super().__init__()
                self.input_shape = input_shape
                self.fc_mu = fc_mu
                self.fc_var = fc_var
                self.abstract_dim = abstract_dim
                self.kls = None

            def forward(self, x, save_kls=True, mu_only=False):
                x = x.squeeze().float()
                mu = self.fc_mu(x)
                log_var = self.fc_var(x)
                std = torch.exp(log_var/2)
                if mu_only:
                    z = mu
                else:
                    q = torch.distributions.Normal(mu, std)
                    z = q.rsample()
                if save_kls:
                    self.kls = self.calculate_kls(z, mu, std)
                return z

            def return_kls(self):
                return self.kls

            def calculate_kls(self, z, mu, std):
                p_std = torch.ones_like(std)
                p = torch.distributions.Normal(torch.zeros_like(mu), p_std)
                q = torch.distributions.Normal(mu, std)
                log_qzx = q.log_prob(z)
                log_pz = p.log_prob(z)
                kl = (log_qzx - log_pz)
                kl = kl.sum(-1)
                return kl

        input_shape = self._input_dimensions[0]
        abstract_dim = self.internal_dim

        # Load yaml file
        with open(HERE / f'yamls/{self._yaml}.yaml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            update(config, self._yaml_mods)

        # Add convolutional layers if needed
        encoder_config = config["encoder"]
        if encoder_config["convs"] is not None:
            convs = make_convs(input_shape, encoder_config["convs"])
            feature_size = compute_feature_size(input_shape, convs)
        else:
            convs = None
            feature_size = np.prod(input_shape)

        #if self._mem_len > 0: # TODO
        #    feature_size += self._mem_len*abstract_dim

        # Variational, or regular encoder
        if self._encoder_type == 'variational':
            fc_mu = make_fc(feature_size, abstract_dim, encoder_config["fc"])
            fc_var = make_fc(feature_size, abstract_dim, encoder_config["fc"])
            encoder = EncoderVariational(
                input_shape, fc_mu, fc_var, abstract_dim=abstract_dim)
        else:
            fc = make_fc(feature_size, abstract_dim, encoder_config["fc"])
            encoder = Encoder(
                input_shape, fc, convs, abstract_dim=abstract_dim,
                mem_len=self._mem_len
                )
        return encoder

    def transition_model(self): # MODULE
        """  Instantiate a Keras model for the transition between two encoded pseudo-states.
    
        The model takes as inputs:
        x : internal state
        a : int
            the action considered
        
        Parameters
        -----------
    
        Returns
        -------
        model that outputs the transition of (x,a)
    
        """

        class TransitionPredictor(nn.Module):
            def __init__(
                self, abstract_state_dim, num_actions, fc,
                encode_new_state, predict_z
                ):
                super().__init__()
                self.abstract_state_dim = abstract_state_dim
                self.num_actions = num_actions
                self.fc = fc
                self.encode_new_state = encode_new_state
                self.predict_z = predict_z
        
            def forward(self, x):
                tr = self.fc(x.float())
                if self.encode_new_state or (not self.predict_z):
                    return tr
                else:
                    return x[:, :self.abstract_state_dim] + tr

        with open(HERE / f'yamls/{self._yaml}.yaml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            update(config, self._yaml_mods)

        obs_shape = self._input_dimensions[0]
        abstract_dim = self.internal_dim
        num_actions = self._n_actions
        tp_config = config["trans-pred"]
        encode_new_state = tp_config["encode_new_state"]
        predict_z = tp_config["predict_z"]
        output_dim = abstract_dim if predict_z else np.prod(obs_shape)
        fc = make_fc(abstract_dim + num_actions, output_dim, tp_config["fc"])
        transition_predictor = TransitionPredictor(
            abstract_dim, num_actions, fc, encode_new_state, predict_z)
        return transition_predictor

    def Q_model(self):
        """ Instantiate a  a Keras model for the Q-network from x.

        The model takes the following inputs:
        x : internal state

        Parameters
        -----------
            
        Returns
        -------
        model that outputs the Q-values for each action
        """

        class QNetwork(nn.Module):
            def __init__(self, convs, fc):
                super().__init__()
                self.convs = convs
                self.fc = fc

            def forward(self, x):
                if self.convs is not None:
                    x = self.convs(x)
                    x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

            def act(self, state):
                q_values = self(state)
                action = torch.argmax(q_value).item()
                return action

        with open(HERE / f'yamls/{self._yaml}.yaml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            update(config, self._yaml_mods)

        qnet_config = config['qnet']
        if self.ddqn_only:
            print('q model')
            convs = make_convs(self._input_dimensions[0], qnet_config['convs'])
            feature_size = compute_feature_size(self._input_dimensions[0], convs)
        else:
            convs = None
            feature_size = self.internal_dim
        fc = make_fc(self.internal_dim, self._n_actions, qnet_config['fc'])
        qnet = QNetwork(convs, fc)
        return qnet

