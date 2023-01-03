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
            self, environment, 
            freeze_interval=1000, batch_size=32,
            random_state=np.random.RandomState(),
            double_Q=False, neural_network=NN, lr=1E-4, nn_yaml='network.yaml',
            loss_weights=[1, 0.2, 1, 1, 1, 1, 1, 0],
            **kwargs
            ):
        """ Initialize the environment. """

        LearningAlgo.__init__(self,environment, batch_size)

        self._freeze_interval = freeze_interval
        self._double_Q = double_Q
        self._random_state = random_state
        self._loss_weights = loss_weights
        self.update_counter = 0    
        self._high_int_dim = kwargs.get('high_int_dim', False)
        self._internal_dim = kwargs.get('internal_dim', 2)
        self._entropy_temp = kwargs.get('entropy_temp', 5.)
        self._nstep = kwargs.get('nstep', 1)
        #self._nstep_decay = kwargs.get('nstep_decay', 1)
        self._nstep_decay = nn.Parameter(torch.tensor(1.))
        self._expand_tcm = kwargs.get('expand_tcm', False)
        self._encoder_type = kwargs.get('encoder_type', None)
        self.loss_T = [0]
        self.loss_R = [0]
        self.loss_Q = [0]
        self.loss_VAE = [0]
        self.loss_entropy_neighbor = [0]
        self.loss_entropy_random = [0]
        self.loss_volume = [0]
        self.loss_gamma = [0]
        self.loss_total = [0]
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        if self._expand_tcm:
            for i in range(len(self._input_dimensions)):
                dim_tuple = list(self._input_dimensions[i])
                dim_tuple[-1] *= self._nstep
                self._input_dimensions[i] = tuple(dim_tuple)

        self.crar = neural_network(
            self._batch_size, self._input_dimensions, self._n_actions,
            self._random_state, high_int_dim=self._high_int_dim,
            internal_dim=self._internal_dim, device=self.device,
            yaml=nn_yaml, nstep=self._nstep, encoder_type=self._encoder_type
            )
        self.optimizer = torch.optim.Adam(self.crar.params, lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

        self.crar_target = neural_network(
            self._batch_size, self._input_dimensions, self._n_actions,
            self._random_state, high_int_dim=self._high_int_dim,
            internal_dim=self._internal_dim, device=self.device,
            yaml=nn_yaml, nstep=self._nstep, encoder_type=self._encoder_type
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
            #'tau': self._nstep_decay
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
        for model, p_to_load in zip(self.crar.models, crar_state_dicts):
            if encoder_only:
                if type(model) == type(self.crar.Q):
                    continue
            model.load_state_dict(p_to_load)
        for model, p_to_load in zip(self.crar_target.models, crar_target_state_dicts):
            if encoder_only:
                if type(model) == type(self.crar.Q):
                    continue
            model.load_state_dict(p_to_load) # TODO: set for tau

    def train(self, states_val, actions_val, rewards_val, next_states_val, terminals_val):
        """
        Train CRAR from one batch of data.

        Parameters
        -----------
        states_val : numpy array of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        actions_val : numpy array of integers with size [self._batch_size]
            actions[i] is the action taken after having observed states[:][i].
        rewards_val : numpy array of floats with size [self._batch_size]
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

        self.optimizer.zero_grad()

        onehot_actions = np.zeros((self._batch_size, self._n_actions))
        onehot_actions[np.arange(self._batch_size), actions_val] = 1
        onehot_actions = torch.as_tensor(onehot_actions, device=self.device).float()

        if (len(states_val) > 1) or (len(next_states_val) > 1):
            raise ValueError('Dimension mismatch')
        states_val = states_val[0]
        next_states_val = next_states_val[0]

        if self._nstep > 1:
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
        R = self.crar.R(Es_and_actions)

        if(self.update_counter%500==0):
            print("Printing a few elements useful for debugging:")
            print("actions_val[0], rewards_val[0], terminals_val[0]")
            print(actions_val[0], rewards_val[0], terminals_val[0])
            print("Es[0], TEs[0], Esp_[0]")
            if(Es.ndim==4):
                print(
                    np.transpose(Es, (0, 3, 1, 2))[0],
                    np.transpose(TEs, (0, 3, 1, 2))[0],
                    np.transpose(Esp, (0, 3, 1, 2))[0]
                    ) # data_format='channels_last' --> 'channels_first'
            else:
                print(Es[0].data, TEs[0].data, Esp[0].data)
            print("R[0]")
            print(R[0])

        # Transition loss
        loss_T = torch.nn.functional.mse_loss(TEs, Esp, reduction='none')
        terminals_mask = torch.tensor(1-terminals_val).float().to(self.device)
        loss_T = loss_T * terminals_mask[:, None]
        loss_T = torch.mean(loss_T)
        self.loss_T[-1] += loss_T.item()

        # Rewards loss
        rewards_val = torch.tensor(rewards_val).view(-1,1).to(self.device).float()
        loss_R = torch.nn.functional.mse_loss(
            self.crar.R(Es_and_actions), rewards_val
            )
        self.loss_R[-1] += loss_R.item()

        # Fit gammas
        gamma_val = terminals_mask * self._df
        gamma_val = gamma_val.view(-1,1).float().to(self.device)
        loss_gamma = torch.nn.functional.mse_loss(
            self.crar.gamma(Es_and_actions), gamma_val
            )
        self.loss_gamma[-1] += loss_gamma.item()

        # Enforce limited volume in abstract state space
        loss_volume = torch.pow(
            torch.norm(Es, p=float('inf'), dim=1),
            2) - 1
        loss_volume = torch.clip(loss_volume, min=0)
        loss_volume = torch.mean(loss_volume)
        self.loss_volume[-1] += loss_volume.item()

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
        if self._double_Q: # Action selection by Q' and evaluation by Q
            # old code-- action selection by Q and evaluation by Q'?
#            next_q_target = self.crar_target.Q(
#                self.crar_target.encoder(next_states_val)
#                )
#            next_q = self.crar.Q(Esp)
#            argmax_next_q = torch.argmax(next_q, axis=1)
#            max_next_q = next_q_target[
#                np.arange(self._batch_size), argmax_next_q
#                ].reshape((-1, 1))

            # Fixed code??
            target_Q_curr = self.crar_target.Q(Es)
            model_Q_next = self.crar.Q(Esp)
            selected_action = torch.argmax(target_Q_curr, axis=1)
            max_next_q = model_Q_next[
                np.arange(self._batch_size), selected_action
                ].reshape((-1, 1))
        else: # Action selection and evaluation by Q' #TODO this is broken
            max_next_q = np.max(next_q_target, axis=1, keepdims=True)
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
            + self._loss_weights[3] * loss_volume \
            + self._loss_weights[4] * loss_gamma \
            + self._loss_weights[5] * loss_R \
            + self._loss_weights[6] * loss_Q
        if self._encoder_type == 'variational':
            loss_VAE = torch.mean(self.crar.encoder.return_kls())
            self.loss_VAE[-1] += loss_VAE.item()
            all_losses = all_losses + self._loss_weights[7] * loss_VAE
        self.loss_total[-1] += all_losses.item()
        all_losses.backward()
        self.optimizer.step()
        self.update_counter += 1

        # Occasional logging
        if(self.update_counter%500==0):
            self.loss_R[-1] /= 500; self.loss_gamma[-1] /= 500
            self.loss_Q[-1] /= 500; self.loss_T[-1] /= 500
            self.loss_entropy_neighbor[-1] /= 500
            self.loss_entropy_random[-1] /= 500
            self.loss_volume[-1] /= 500; self.loss_VAE[-1] /= 500
            self.loss_total[-1] /= 500
            print('LOSSES')
            print(f'T = {self.loss_T[-1]}; R = {self.loss_R[-1]}; \
                Gamma = {self.loss_gamma[-1]}; Q = {self.loss_Q[-1]};')
            print(f'Entropy Neighbor = {self.loss_entropy_neighbor[-1]}; \
                Entropy Random = {self.loss_entropy_random[-1]}; \
                Volume = {self.loss_volume[-1]}; VAE = {self.loss_VAE[-1]}')
            self.loss_R.append(0); self.loss_gamma.append(0)
            self.loss_Q.append(0); self.loss_T.append(0)
            self.loss_entropy_neighbor.append(0)
            self.loss_entropy_random.append(0)
            self.loss_volume.append(0); self.loss_VAE.append(0)
            self.loss_total.append(0)

        return loss_Q.item(), loss_Q_unreduced

    def recurrent_train(
        self, observations, actions_val, rewards_val,
        terminals_val, hidden_states):
        """
        Train CRAR from one batch of data. Like the train method, but each
        value in a batch is now a history of observations.
        """

        raise ValueError('LSTM encoder is not fully implemented.')

        self.optimizer.zero_grad()
        states_val = [obs[:,:-1,:] for obs in observations]
        next_states_val = [obs[:,1:,:] for obs in observations]
        actions_val = actions_val[:, -1]
        rewards_val = rewards_val[:,-1]
        terminals_val = terminals_val[:,-1]
        onehot_actions = np.zeros((self._batch_size, self._n_actions))
        onehot_actions[np.arange(self._batch_size), actions_val] = 1
        onehot_actions = torch.as_tensor(onehot_actions, device=self.device).float()

        if (len(states_val) > 1) or (len(next_states_val) > 1):
            raise ValueError('Dimension mismatch')

        states_val = states_val[0]
        next_states_val = next_states_val[0]
        states_val = torch.as_tensor(states_val, device=self.device).float()
        next_states_val = torch.as_tensor(next_states_val, device=self.device).float()

        Esp = self.crar.encoder(
            next_states_val, [h[1] for h in hidden_states], update_hidden=True
            )
        Es = self.crar.encoder(
            states_val, [h[0] for h in hidden_states], update_hidden=True
            )
        Es_and_actions = torch.cat([Es, onehot_actions], dim=1)
        TEs = self.crar.transition(Es_and_actions)
        R = self.crar.R(Es_and_actions)

        if(self.update_counter%500==0):
            print("Printing a few elements useful for debugging:")
            print("actions_val[0], rewards_val[0], terminals_val[0]")
            print(actions_val[0], rewards_val[0], terminals_val[0])
            print("Es[0], TEs[0], Esp_[0]")
            if(Es.ndim==4):
                print(
                    np.transpose(Es, (0, 3, 1, 2))[0],
                    np.transpose(TEs, (0, 3, 1, 2))[0],
                    np.transpose(Esp, (0, 3, 1, 2))[0]
                    ) # data_format='channels_last' --> 'channels_first'
            else:
                print(Es[0].data, TEs[0].data, Esp[0].data)
            print("R[0]")
            print(R[0])

        # Transition loss
        loss_T = torch.nn.functional.mse_loss(TEs, Esp, reduction='none')
        terminals_mask = torch.tensor(1-terminals_val).float().to(self.device)
        loss_T = loss_T * terminals_mask[:, None]
        loss_T = torch.mean(loss_T)
        self.loss_T += loss_T.item()

        # Rewards loss
        rewards_val = torch.tensor(rewards_val).view(-1,1).to(self.device).float()
        loss_R = torch.nn.functional.mse_loss(
            self.crar.R(Es_and_actions), rewards_val
            )
        self.loss_R += loss_R.item()

        # Fit gammas
        gamma_val = terminals_mask * self._df
        gamma_val = gamma_val.view(-1,1).float().to(self.device)
        loss_gamma = torch.nn.functional.mse_loss(
            self.crar.gamma(Es_and_actions), gamma_val
            )
        self.loss_gamma += loss_gamma.item()

        # Enforce limited volume in abstract state space
        loss_volume = torch.pow(
            torch.norm(Es, p=float('inf'), dim=1),
            2) - 1
        loss_volume = torch.clip(loss_volume, min=0)
        loss_volume = torch.mean(loss_volume)
        self.loss_volume += loss_volume.item()

        # Increase entropy of randomly sampled states
        # This works only when states_val is made up of only one observation
        rolled = torch.roll(Es, 1, dims=0)
        loss_entropy_random = torch.exp(
            -self._entropy_temp * torch.norm(Es - rolled, dim=1)
            )
        loss_entropy_random = torch.mean(loss_entropy_random)
        self.loss_entropy_random += loss_entropy_random.item()

        # Increase entropy of neighboring states
        loss_entropy_neighbor = torch.exp(
            -self._entropy_temp * torch.norm(Es - Esp, dim=1)
            )
        loss_entropy_neighbor = torch.mean(loss_entropy_neighbor)
        self.loss_entropy_neighbor += loss_entropy_neighbor.item()

        # Q network stuff
        if self.update_counter % self._freeze_interval == 0:
            self.resetQHat()
        next_q_target = self.crar_target.Q(self.crar_target.encoder(
            next_states_val, [h[1] for h in hidden_states], update_hidden=False
            ))

        if self._double_Q: # Action selection by Q
            next_q = self.crar.Q(Esp)
            argmax_next_q = torch.argmax(next_q, axis=1)
            max_next_q = next_q_target[
                np.arange(self._batch_size), argmax_next_q
                ].reshape((-1, 1))
        else: # Action selection by Q'
            max_next_q = np.max(next_q_target, axis=1, keepdims=True)
        target = rewards_val.squeeze() + terminals_mask*self._df*max_next_q.squeeze()
        q_vals = self.crar.Q(Es)
        q_vals = q_vals[np.arange(self._batch_size), actions_val]
        loss_Q = torch.nn.functional.mse_loss(q_vals, target, reduction='none')
        loss_Q_unreduced = loss_Q
        loss_Q = torch.mean(loss_Q)
        self.loss_Q += loss_Q.item()

        # Aggregate all losses and update parameters
        all_losses = self._loss_weights[0] * loss_T \
            + self._loss_weights[1] * loss_entropy_neighbor \
            + self._loss_weights[2] * loss_entropy_random \
            + self._loss_weights[3] * loss_volume \
            + self._loss_weights[4] * loss_gamma \
            + self._loss_weights[5] * loss_R \
            + self._loss_weights[6] * loss_Q
        if self._encoder_type == 'variational':
            loss_VAE = torch.mean(self.crar.encoder.return_kls())
            self.loss_VAE[-1] += loss_VAE.item()
            all_losses = all_losses + self._loss_weights[7] * loss_VAE
        self.loss_total[-1] += all_losses.item()
        all_losses.backward()
        self.optimizer.step()
        self.update_counter += 1

        # Occasional logging
        if(self.update_counter%500==0):
            self.loss_R[-1] /= 500; self.loss_gamma[-1] /= 500
            self.loss_Q[-1] /= 500; self.loss_T[-1] /= 500
            self.loss_entropy_neighbor[-1] /= 500
            self.loss_entropy_random[-1] /= 500
            self.loss_volume[-1] /= 500; self.loss_VAE[-1] /= 500
            self.loss_total[-1] /= 500
            print('LOSSES')
            print(f'T = {self.loss_T[-1]}; R = {self.loss_R[-1]}; \
                Gamma = {self.loss_gamma[-1]}; Q = {self.loss_Q[-1]};')
            print(f'Entropy Neighbor = {self.loss_entropy_neighbor[-1]}; \
                Entropy Random = {self.loss_entropy_random[-1]}; \
                Volume = {self.loss_volume[-1]}; VAE = {self.loss_VAE[-1]}')
            self.loss_R.append(0); self.loss_gamma.append(0)
            self.loss_Q.append(0); self.loss_T.append(0)
            self.loss_entropy_neighbor.append(0)
            self.loss_entropy_random.append(0)
            self.loss_volume.append(0); self.loss_VAE.append(0)
            self.loss_total.append(0)
        return loss_Q.item(), loss_Q_unreduced

    def make_state_with_history(self, states_buffer):
        states_buffer = torch.tensor(states_buffer) # (N, T, H, W)
        n_batches = states_buffer.shape[0]
        tau = self._nstep_decay
        new_states = []
        for batch in range(n_batches):
            #walls = torch.argwhere(states_buffer[batch,-1] == -1).squeeze()
            batch_obs = []
            for t in range(self._nstep):
                discount = torch.pow(tau, self._nstep-t)
                batch_obs.append(states_buffer[batch, t]*discount)
                #states_val[batch, t][walls] = -1.
            if self._expand_tcm:
                new_states.append(torch.hstack(batch_obs))
            else:
                new_states.append(torch.sum(batch_obs, dim=0))
        new_states = torch.stack(new_states) # (N, H, W)
        return new_states

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
    
        if(mode==None):
            mode=0
        depths = [0,1,3,6,10] # Mode defines the planning depth di
        depth = depths[mode]
        with torch.no_grad():
            if self._encoder_type == 'recurrent':
                state = torch.as_tensor(state, device=self.device).float()
                xs = self.crar.encoder(state, self.crar.encoder.hidden)
            else:
                _state = self.make_state_with_history(state)
                state = torch.as_tensor(_state, device=self.device).float()
                xs = self.crar.encoder(state)
        q_vals = self.qValues(xs, d=depth)
        return torch.argmax(q_vals), torch.max(q_vals)

    def getPossibleTransitions(self, state):
        """
        From a single state, return the possible transition states.
        """
        with torch.no_grad():
            state = torch.as_tensor(state, device=self.device).float()
            Es = self.crar.encoder(state)
            Es = [torch.clone(Es) for _ in range(self._n_actions)]
            Es = torch.stack(Es)
            onehot_actions = np.zeros((self._n_actions, self._n_actions))
            onehot_actions[np.arange(self._n_actions), np.arange(self._n_actions)] = 1
            onehot_actions = torch.as_tensor(onehot_actions, device=self.device).float()
            Es_and_actions = torch.cat([Es, onehot_actions], dim=1)
            TEs = self.crar.transition(Es_and_actions)
        return Es, TEs

    def resetQHat(self):
        """ Set the target Q-network weights equal to the main Q-network weights
        """

        for target_model, model in zip(self.crar_target.models, self.crar.models):
            target_model.load_state_dict(model.state_dict())

    def transfer(self, original, transfer, epochs=1):
        raise ValueError("Not implemented.")

    def reset_rnn(self, vec=None):
        self.crar.encoder.reset_hidden(vec)

    def get_rnn_hidden(self):
        hidden = self.crar.encoder.get_hidden()
        return hidden

    def get_losses(self):
        losses = [
            self.loss_T[:-1],
            self.loss_entropy_neighbor[:-1], self.loss_entropy_random[:-1],
            self.loss_volume[:-1], 
            self.loss_gamma[:-1], self.loss_R[:-1], self.loss_Q[:-1],
            self.loss_VAE[:-1], self.loss_total[:-1]
            ]
        loss_names = [
            'Transition', 'Entropy (neighbor)', 'Entropy (random)', 'Volume',
            'Gamma', 'Reward', 'Q Value', 'VAE', 'TOTAL'
            ]
        return losses, loss_names

