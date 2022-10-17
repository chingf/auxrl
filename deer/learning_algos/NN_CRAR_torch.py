import numpy as np
import torch
import torch.nn as nn
from itertools import accumulate
import inspect
import yaml
from pathlib import Path

NN_MAP = {
    k: v for k, v in inspect.getmembers(nn) # if nonthrowing_issubclass(v, nn.Module)
}
HERE = Path(__file__).parent

def compute_feature_size(input_shape, convs):
    return convs(torch.zeros(1, *input_shape)).view(1, -1).size(1)

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
    high_int_dim : Boolean
        Whether the abstract state should be high dimensional in the form of frames/vectors or whether it should 
        be low-dimensional
    """
    def __init__(self, batch_size, input_dimensions, n_actions, random_state, **kwargs):
        self._input_dimensions=input_dimensions
        self._batch_size=batch_size
        self._random_state=random_state
        self._n_actions=n_actions
        self._high_int_dim=kwargs["high_int_dim"]
        if self._high_int_dim:
            raise ValueError("Not implemented")

        if(self._high_int_dim==True):
            self.n_channels_internal_dim=kwargs["internal_dim"] #dim[-3]
        else:
            self.internal_dim=kwargs["internal_dim"]    #2 for laby
                                                        #3 for catcher

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
                self, input_shape, fc, convs=None, act=nn.Tanh, abstract_state_dim=2
            ):
                super().__init__()
                self.input_shape = input_shape
                self.convs = convs
                self.fc = fc
        
            def forward(self, x):
                if self.convs is not None:
                    x = self.convs(x)
                    x = x.view(x.size(0), -1)
                x = self.fc(x.float())
                return x

        input_shape = self._input_dimensions[0]
        abstract_dim = self.internal_dim
        if (len(self._input_dimensions) > 1) or (len(self._input_dimensions[0]) < 3):
            raise ValueError('Not implemented')

        with open(HERE / "network.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        encoder_config = config["encoder"]
        convs, feature_size = None, input_shape[0]
        if encoder_config["convs"] is not None:
            convs = make_convs(input_shape, encoder_config["convs"])
            feature_size = compute_feature_size(input_shape, convs)
        fc = make_fc(feature_size, abstract_dim, encoder_config["fc"])
        encoder = Encoder(input_shape, fc, convs)
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
            def __init__(self, abstract_state_dim, num_actions, fc):
                super().__init__()
                self.abstract_state_dim = abstract_state_dim
                self.num_actions = num_actions
                self.fc = fc
        
            def forward(self, x):
                tr = self.fc(x.float())
                return x[:, :self.abstract_state_dim] + tr

        with open(HERE / "network.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        num_actions = self._n_actions
        abstract_dim = self.internal_dim
        tp_config = config["trans-pred"]
        fc = make_fc(abstract_dim + num_actions, abstract_dim, tp_config["fc"])
        transition_predictor = TransitionPredictor(abstract_dim, num_actions, fc)
        return transition_predictor

    def float_model(self):
        """ Instantiate a Keras model for fitting a float from x.
                
        The model takes the following inputs:
        x : internal state
        a : int
            the action considered at x
        
        Parameters
        -----------
            
        Returns
        -------
        model that outputs a float
    
        """

        class ScalarPredictor(nn.Module):
            def __init__(self, abstract_state_dim, num_actions, fc):
                super().__init__()
                self.fc = fc
        
            def forward(self, x):
                x = self.fc(x.float())
                return x
        with open(HERE / "network.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        num_actions = self._n_actions
        abstract_dim = self.internal_dim
        rp_config = config['float-pred']
        fc = make_fc(abstract_dim + num_actions, abstract_dim, rp_config["fc"])
        scalar_predictor = ScalarPredictor(abstract_dim, num_actions, fc)
        return scalar_predictor

    def force_features(self, encoder_model, transition_model, plan_depth=0):
        """ Instantiate a Keras model that provides the vector of the transition at E(s1). It is calculated as the different between E(s1) and E(T(s1)). 
        Used to force the directions of the transitions.
        
        The model takes the four following inputs:
        s1 : list of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        a : list of ints with length (plan_depth+1)
            the action(s) considered at s1
        
        Parameters
        -----------
        encoder_model: instantiation of a Keras model for the encoder (E)
        transition_model: instantiation of a Keras model for the transition (T)
        plan_depth: if>1, it provides the possibility to consider a sequence of transitions between s1 and s2 
        (input a is then a list of actions)
            
        Returns
        -------
        model with output E(s1)-T(E(s1))
    
        """

        if plan_depth > 0: raise ValueError('Not implemented.')

        def f(Es, TEs):
            diff_features = Es - TEs
        return f

    def full_float(
            self, encoder_model, float_model,
            plan_depth=0, transition_model=None
            ):
        """ Instantiate a Keras model for fitting a float from s.
                
        The model takes the four following inputs:
        s : list of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        a : list of ints with length (plan_depth+1)
            the action(s) considered at s
                
        Parameters
        -----------
        encoder_model: instantiation of a Keras model for the encoder (E)
        float_model: instantiation of a Keras model for fitting a float from x
        plan_depth: if>1, it provides the possibility to consider a sequence of transitions following s 
        (input a is then a list of actions)
        transition_model: instantiation of a Keras model for the transition (T)
            
        Returns
        -------
        model with output the reward r
        """

        if plan_depth > 0: raise ValueError('Not implemented.')

        def f(Es_and_actions):
            out = float_model(Es_and_actions)
            return out
        return f

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

        return None

    def full_Q_model(self, encoder_model, Q_model, plan_depth=0, transition_model=None, R_model=None, discount_model=None):
        """ Instantiate a  a Keras model for the Q-network from s.

        The model takes the following inputs:
        s : list of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        a : list of ints with length plan_depth; if plan_depth=0, there isn't any input for a.
            the action(s) considered at s
    
        Parameters
        -----------
        encoder_model: instantiation of a Keras model for the encoder (E)
        Q_model: instantiation of a Keras model for the Q-network from x.
        plan_depth: if>1, it provides the possibility to consider a sequence of transitions following s 
        (input a is then a list of actions)
        transition_model: instantiation of a Keras model for the transition (T)
        R_model: instantiation of a Keras model for the reward
        discount_model: instantiation of a Keras model for the discount
            
        Returns
        -------
        model with output the Q-values
        """

        if plan_depth > 0: raise ValueError('Not implemented.')

        return None
