import numpy as np
import inspect
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from copy import deepcopy
from ..base_classes import LearningAlgo
from .NN_CRAR_torch import NN # Default Neural network used

class CRAR(LearningAlgo):
    """
    Combined Reinforcement learning via Abstract Representations (CRAR) using Keras
    
    Parameters
    -----------
    environment : object from class Environment
        The environment in which the agent evolves.
    freeze_interval : int
        Period during which the target network is freezed and after which the target network is updated. Default : 1000
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent. Default : 32
    random_state : numpy random number generator
        Set the random seed.
    double_Q : bool, optional
        Activate or not the double_Q learning.
        More informations in : Hado van Hasselt et al. (2015) - Deep Reinforcement Learning with Double Q-learning.
    neural_network : object, optional
        Default is deer.learning_algos.NN_keras
    """

    def __init__(
        self, environment, freeze_interval=1000, batch_size=32,
        random_state=np.random.RandomState(), double_Q=False,
        neural_network=NN, lr=1E-4, nn_yaml='basic', yaml_mods=None,
        loss_weights=[1, 1, 1, 1, 1], # T, entropy, entropy, Q, VAE
        internal_dim=5, entropy_temp=5, mem_len=1, encoder_type=None,
        pred_len=1, pred_gamma=0.
        ):
        """ Initialize the environment. """

        LearningAlgo.__init__(self,environment, batch_size)

        self._freeze_interval = freeze_interval
        self._double_Q = double_Q
        self._random_state = random_state
        self._loss_weights = loss_weights
        self._internal_dim = internal_dim
        self._entropy_temp = entropy_temp
        self._mem_len = mem_len
        self._encoder_type = encoder_type
        self._pred_len = pred_len
        self._pred_gamma = pred_gamma
        self.update_counter = 0
        self.loss_T = [0]
        self.loss_Q = [0]
        self.loss_VAE = [0]
        self.loss_entropy_neighbor = [0]
        self.loss_entropy_random = [0]
        self.loss_total = [0]
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        if pred_len > 1 and mem_len > 1:
            raise ValueError("Not implemented for pred_len and mem_len > 1")

        # Base network
        self.crar = neural_network(
            self._batch_size, self._input_dimensions, self._n_actions,
            self._random_state, internal_dim=self._internal_dim,
            device=self.device, yaml=nn_yaml, yaml_mods=yaml_mods,
            mem_len=self._mem_len, encoder_type=self._encoder_type
            )
        self.optimizer = torch.optim.Adam(self.crar.params, lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.9)

        # Target network
        self.crar_target = neural_network(
            self._batch_size, self._input_dimensions, self._n_actions,
            self._random_state, internal_dim=self._internal_dim,
            device=self.device, yaml=nn_yaml, yaml_mods=yaml_mods,
            mem_len=self._mem_len, encoder_type=self._encoder_type
            )
        self.optimizer_target = torch.optim.Adam(
            self.crar_target.Q.parameters(), lr=lr
            )
        self.scheduler_target = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer_target, gamma=0.9
            )
        self.resetQHat()

    def getAllParams(self):
        """ Provides all parameters used by the learning algorithm

        Returns
        -------
        Values of the parameters: list of numpy arrays
        """

        crar_state_dicts = []
        crar_target_state_dicts = []
        for model in self.crar.models:
            crar_state_dicts.append(model.state_dict())
        for model in self.crar_target.models:
            crar_target_state_dicts.append(model.state_dict())
        params = {
            'crar': crar_state_dicts, 'crar_target': crar_target_state_dicts,
            }
        return params

    def setAllParams(self, params, encoder_only):
        """ Set all parameters used by the learning algorithm

        Arguments
        ---------
        params: dict as created in getAllParams
        """

        crar_state_dicts = params['crar']
        crar_target_state_dicts = params['crar_target']
        if len(crar_state_dicts) == (len(self.crar.models)+2):
            crar_state_dicts = [crar_state_dicts[0], crar_state_dicts[3], crar_state_dicts[4]]
            crar_target_state_dicts = [
                crar_target_state_dicts[0],
                crar_target_state_dicts[3], crar_target_state_dicts[4]]
        for model, p_to_load in zip(self.crar.models, crar_state_dicts):
            if encoder_only:
                if type(model) == type(self.crar.Q):
                    continue
            model.load_state_dict(p_to_load)
        for model, p_to_load in zip(self.crar_target.models, crar_target_state_dicts):
            if encoder_only:
                if type(model) == type(self.crar.Q):
                    continue
            model.load_state_dict(p_to_load)

    def train(self, states_val, actions_val, rewards_val, next_states_val, terminals_val):
        """
        Train CRAR from one batch of data.

        Parameters
        -----------
        states_val : numpy array of objects [(N, T, 4, 4)]
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        actions_val : numpy array of integers with size [self._batch_size] [(5,)]*N
            actions[i] is the action taken after having observed states[:][i].
        rewards_val : numpy array of floats with size [self._batch_size] (64,)
            rewards[i] is the reward obtained for taking actions[i-1].
        next_states_val : numpy array of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        terminals_val : numpy array of booleans with size [self._batch_size]
            terminals[i] is True if the transition leads to a terminal state and False otherwise

        Returns
        -------
        Average loss of the batch training for the Q-values (RMSE)
        Individual (square) losses for the Q-values for each tuple
        """

        if self._pred_len > 1:
            actions_val = actions_val[:, 0]
            rewards_val = rewards_val[:, 0]
            terminals_val = terminals_val[:, 0]
            T_states_val = states_val; T_next_states_val = next_states_val
            states_val = [states_val[0][:,:1]]
            next_states_val = [next_states_val[0][:,:1]]
        self.optimizer.zero_grad()

        onehot_actions = np.zeros((self._batch_size, self._n_actions))
        onehot_actions[np.arange(self._batch_size), actions_val] = 1
        onehot_actions = torch.as_tensor(onehot_actions, device=self.device).float()

        if (len(states_val) > 1) or (len(next_states_val) > 1):
            raise ValueError('Dimension mismatch')
        states_val = states_val[0]
        next_states_val = next_states_val[0]
        states_val = self.make_state_with_history(states_val)
        next_states_val = self.make_state_with_history(next_states_val)

        states_val = torch.as_tensor(states_val, device=self.device).float()
        next_states_val = torch.as_tensor(next_states_val, device=self.device).float()

        Esp = self.crar.encoder(next_states_val)
        if self._encoder_type == 'variational':
            Es = self.crar.encoder(states_val, save_kls=True)
        else:
            Es = self.crar.encoder(states_val)
        Es_and_actions = torch.cat([Es, onehot_actions], dim=1)
        TEs = self.crar.transition(Es_and_actions)

        # Transition loss
        predict_z = self.crar.transition.predict_z
        T_target = Esp if predict_z else \
            next_states_val.reshape((self._batch_size, -1))
        sr_gamma = self._pred_gamma
        for t in np.arange(1, self._pred_len):
            s = self.make_state_with_history(T_next_states_val[0][:,t:t+1])
            s = torch.as_tensor(s, device=self.device).float()
            next_step_pred = self.crar.encoder(s) if predict_z else \
                s.reshape((self._batch_size, -1))
            T_target = T_target + (sr_gamma**t) * next_step_pred
        loss_T = torch.nn.functional.mse_loss(TEs, T_target, reduction='none')
        terminals_mask = torch.tensor(1-terminals_val).float().to(self.device)
        loss_T = loss_T * terminals_mask[:, None]
        loss_T = torch.mean(loss_T)
        self.loss_T[-1] += loss_T.item()

        # Increase entropy of randomly sampled states
        # This works only when states_val is made up of only one observation
        rolled = torch.roll(Es, 1, dims=0)
        loss_entropy_random = torch.exp(
            -self._entropy_temp * torch.norm(Es - rolled, dim=1)
            )
        loss_entropy_random = torch.mean(loss_entropy_random)
        self.loss_entropy_random[-1] += loss_entropy_random.item()

        # Increase entropy of neighboring states
        loss_entropy_neighbor = torch.exp(
            -self._entropy_temp * torch.norm(Es - Esp, dim=1)
            )
        loss_entropy_neighbor = torch.mean(loss_entropy_neighbor)
        self.loss_entropy_neighbor[-1] += loss_entropy_neighbor.item()

        # Q network stuff
        if self.update_counter % self._freeze_interval == 0:
            self.resetQHat()
        next_q_target = self.crar_target.Q(
            self.crar_target.encoder(next_states_val)
            )
        if self._double_Q: # Action selection by Q and evaluation by Q'
            next_q = self.crar.Q(Esp)
            argmax_next_q = torch.argmax(next_q, axis=1)
            max_next_q = next_q_target[
                np.arange(self._batch_size), argmax_next_q
                ].reshape((-1, 1))
        else: # Action selection and evaluation by Q'
            max_next_q = np.max(next_q_target, axis=1, keepdims=True)
        rewards_val = torch.tensor(rewards_val).view(-1,1).to(self.device).float()
        target = rewards_val.squeeze() + terminals_mask*self._df*max_next_q.squeeze()
        q_vals = self.crar.Q(Es)
        q_vals = q_vals[np.arange(self._batch_size), actions_val]
        loss_Q = torch.nn.functional.mse_loss(q_vals, target, reduction='none')
        loss_Q_unreduced = loss_Q
        loss_Q = torch.mean(loss_Q)
        self.loss_Q[-1] += loss_Q.item()

        # Aggregate all losses and update parameters
        all_losses = self._loss_weights[0] * loss_T \
            + self._loss_weights[1] * loss_entropy_neighbor \
            + self._loss_weights[2] * loss_entropy_random \
            + self._loss_weights[3] * loss_Q
        if self._encoder_type == 'variational':
            loss_VAE = torch.mean(self.crar.encoder.return_kls())
            self.loss_VAE[-1] += loss_VAE.item()
            all_losses = all_losses + self._loss_weights[4] * loss_VAE
        self.loss_total[-1] += all_losses.item()
        all_losses.backward()
        self.optimizer.step()
        self.update_counter += 1

        # Occasional logging
        self.loss_Q.append(0); self.loss_T.append(0)
        self.loss_entropy_neighbor.append(0)
        self.loss_entropy_random.append(0)
        self.loss_VAE.append(0)
        self.loss_total.append(0)

        return loss_Q.item(), loss_Q_unreduced

    def make_state_with_history(self, states_buffer):
        if self._mem_len <= 1:
            return np.squeeze(states_buffer, axis=1)
        states_buffer = torch.tensor(states_buffer) # (N, T, H, W)
        return states_buffer

    def step_scheduler(self):
        self.scheduler.step()
        self.scheduler_target.step()

    def qValues(self, x, d=5, from_target=False):
        """ Get the q values for one pseudo-state (without planning)

        Arguments
        ---------
        state_val : array of objects (or list of objects)
            Each object is a numpy array that relates to one of the observations
            with size [1 * history size * size of punctual observation (which is 2D,1D or scalar)]).

        Returns
        -------
        The q values for the provided pseudo state
        """ 

        crar_net = self.crar_target if from_target else self.crar
        with torch.no_grad():
            if d == 0:
                return crar_net.Q(x)
            else:
                q_plan_values = []
                x = x.view(1, -1)
                for a in range(self._n_actions):
                    onehot_actions = np.zeros((1, self._n_actions))
                    onehot_actions[:, a] = 1
                    onehot_actions = torch.tensor(onehot_actions, device=self.device).float()
                    state_and_actions = torch.cat([x, onehot_actions], 1) 
                    rewards = crar_net.R(state_and_actions)
                    discounts = torch.tensor([self._df]*rewards.shape[0], device=self.device)
                    next_x = crar_net.transition(state_and_actions)
                    next_q_vals = self.qValues(next_x, d-1, from_target)
                    q_plan_values.append(
                        rewards + discounts * torch.max(next_q_vals, dim=1)[0]
                        )
                return torch.cat(q_plan_values, dim=1)

    def chooseBestAction(self, state, mode, *args, **kwargs):
        """ Get the best action for a pseudo-state

        Arguments
        ---------
        state : list of numpy arrays
             One pseudo-state. The number of arrays and their dimensions matches self.environment.inputDimensions().
        mode : int
            Identifier of the mode (-1 is reserved for the training mode).

        Returns
        -------
        The best action : int
        """
    
        if mode == None:
            mode = 0
        depths = [0,1,3,6,10] # Mode defines the planning depth di
        depth = depths[mode]
        with torch.no_grad():
            _state = self.make_state_with_history(state)
            state = torch.as_tensor(_state, device=self.device).float()
            xs = self.crar.encoder(state)
        q_vals = self.qValues(xs, d=depth)
        return torch.argmax(q_vals), torch.max(q_vals)

    def resetQHat(self):
        """ Set the target Q-network weights equal to the main Q-network weights
        """

        for target_model, model in zip(self.crar_target.models, self.crar.models):
            target_model.load_state_dict(model.state_dict())

    def get_losses(self):
        losses = [
            self.loss_T[:-1],
            self.loss_entropy_neighbor[:-1], self.loss_entropy_random[:-1],
            self.loss_Q[:-1], self.loss_VAE[:-1], self.loss_total[:-1]
            ]
        loss_names = [
            'Transition', 'Entropy (neighbor)', 'Entropy (random)',
            'Q Value', 'VAE', 'TOTAL'
            ]
        return losses, loss_names

