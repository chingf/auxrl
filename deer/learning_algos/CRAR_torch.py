import numpy as np
import inspect
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from ..base_classes import LearningAlgo
from .NN_CRAR_torch import NN # Default Neural network used

class CRAR(LearningAlgo):
    """
    Combined Reinforcement learning via Abstract Representations (CRAR) using Keras
    
    Parameters
    -----------
    environment : object from class Environment
        The environment in which the agent evolves.
    rho : float
        Parameter for rmsprop. Default : 0.9
    rms_epsilon : float
        Parameter for rmsprop. Default : 0.0001
    momentum : float
        Momentum for SGD. Default : 0
    clip_norm : float
        The gradient tensor will be clipped to a maximum L2 norm given by this value.
    freeze_interval : int
        Period during which the target network is freezed and after which the target network is updated. Default : 1000
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent. Default : 32
    update_rule: str
        {sgd,rmsprop}. Default : rmsprop
    random_state : numpy random number generator
        Set the random seed.
    double_Q : bool, optional
        Activate or not the double_Q learning.
        More informations in : Hado van Hasselt et al. (2015) - Deep Reinforcement Learning with Double Q-learning.
    neural_network : object, optional
        Default is deer.learning_algos.NN_keras
    """

    def __init__(
            self, environment, rho=0.9, rms_epsilon=0.0001, momentum=0,
            clip_norm=0, freeze_interval=1000, batch_size=32,
            update_rule="rmsprop", random_state=np.random.RandomState(),
            double_Q=False, neural_network=NN, lr=1E-4, **kwargs
            ):
        """ Initialize the environment. """

        LearningAlgo.__init__(self,environment, batch_size)

        self._rho = rho
        self._rms_epsilon = rms_epsilon
        self._momentum = momentum
        self._clip_norm = clip_norm
        self._update_rule = update_rule
        self._freeze_interval = freeze_interval
        self._double_Q = double_Q
        self._random_state = random_state
        self.update_counter = 0    
        self._high_int_dim = kwargs.get('high_int_dim',False)
        self._internal_dim = kwargs.get('internal_dim',2)
        self._div_entrop_loss = kwargs.get('div_entrop_loss',5.)
        self.loss_interpret=0
        self.loss_T=0
        self.loss_R=0
        self.loss_Q=0
        self.loss_disentangle_t=0
        self.loss_disambiguate1=0
        self.loss_disambiguate2=0
        self.loss_gamma=0
        self.tracked_losses = []
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
            )

        self.learn_and_plan = neural_network(
            self._batch_size, self._input_dimensions, self._n_actions,
            self._random_state, high_int_dim=self._high_int_dim,
            internal_dim=self._internal_dim
            )
        self.encoder = self.learn_and_plan.encoder_model().to(self.device)
        self.R = self.learn_and_plan.float_model().to(self.device)
        self.Q = self.learn_and_plan.Q_model()
        self.gamma = self.learn_and_plan.float_model().to(self.device)
        self.transition = self.learn_and_plan.transition_model().to(self.device)
        self.full_Q = self.learn_and_plan.full_Q_model(self.encoder,self.Q,0,self._df)

        # used to fit rewards
        self.full_R = self.learn_and_plan.full_float(self.encoder,self.R)
        
        # used to fit gamma
        self.full_gamma = self.learn_and_plan.full_float(self.encoder,self.gamma)
        
        # used to fit transitions
        self.diff_Tx_x_ = self.learn_and_plan.diff_Tx_x_(self.encoder,self.transition)
        
        # constraint on consecutive t
        self.diff_s_s_ = self.learn_and_plan.diff_encoder(self.encoder)

        # used to force features variations
        if(self._high_int_dim==False):
            self.force_features=self.learn_and_plan.force_features(self.encoder,self.transition)
                
        self.params = []
        for model in [self.encoder, self.R, self.gamma, self.transition]:
            self.params.extend([p for p in model.parameters()])
        self.optimizer = torch.optim.Adam(self.params, lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.85)

        # Make targets

    def getAllParams(self):
        """ Provides all parameters used by the learning algorithm

        Returns
        -------
        Values of the parameters: list of numpy arrays
        """

        return None # TODO

    def setAllParams(self, list_of_values):
        """ Set all parameters used by the learning algorithm

        Arguments
        ---------
        list_of_values : list of numpy arrays
             list of the parameters to be set (same order than given by getAllParams()).
        """

        pass # TODO

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
        onehot_actions_rand = np.zeros((self._batch_size, self._n_actions))
        onehot_actions_rand[np.arange(self._batch_size), np.random.randint(0,2,(32))] = 1
        onehot_actions = torch.as_tensor(onehot_actions, device=self.device).float()
        onehot_actions_rand = torch.as_tensor(onehot_actions_rand, device=self.device).float()

        if (len(states_val) > 1) or (len(next_states_val) > 1):
            raise ValueError('Dimension mismatch')
        states_val = states_val[0]
        next_states_val = next_states_val[0]
        states_val = torch.as_tensor(states_val, device=self.device).float()
        next_states_val = torch.as_tensor(next_states_val, device=self.device).float()

        Esp = self.encoder(next_states_val)
        Es = self.encoder(states_val)
        Es_and_actions = torch.cat([Es, onehot_actions], dim=1)
        TEs = self.transition(Es_and_actions)
        R = self.R(Es_and_actions)

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

        # Fit transition
        loss_T = torch.nn.functional.mse_loss(
            self.diff_Tx_x_(Esp, TEs, torch.tensor(terminals_val).to(self.device)),
            torch.zeros_like(Es)
            )
        self.loss_T += loss_T.item()

        # Fit rewards
        loss_R = torch.nn.functional.mse_loss(
            self.full_R(Es_and_actions),
            torch.tensor(rewards_val).view(-1,1).to(self.device)
            )
        self.loss_R += loss_R.item()

        # Fit gammas
        loss_gamma = torch.nn.functional.mse_loss(
            self.full_gamma(Es_and_actions),
            torch.tensor((1-terminals_val[:])*self._df).view(-1,1).float().to(self.device)
            )
        self.loss_gamma += loss_gamma.item()

        # Loss to ensure entropy but limited volume in abstract state space, avg=0 and sigma=1
        # reduce the squared value of the abstract features
        loss_disambiguate1 = torch.nn.functional.mse_loss(Es, torch.zeros_like(Es))
        self.loss_disambiguate1 += loss_disambiguate1.item()

        # Increase the entropy in the abstract features of two states
        # This works only when states_val is made up of only one observation --> FIXME
        rolled = torch.roll(Es, 1, dims=0)
        loss_disambiguate2 = torch.nn.functional.mse_loss(
                self.diff_s_s_(Es, rolled),
                torch.reshape(torch.zeros_like(Es),(self._batch_size,-1))
                )
        self.loss_disambiguate2 += loss_disambiguate2.item()

        # Some other disentangling loss
        loss_disentangle_t = torch.nn.functional.mse_loss(
                self.diff_s_s_(Es, Esp),
                torch.reshape(torch.zeros_like(Es),(self._batch_size,-1))
                )
        self.loss_disentangle_t += loss_disentangle_t.item()

#        # Interpretable AI
#        target_modif_features=np.zeros((self._n_actions,self._internal_dim))
#        ## Catcher
#        #target_modif_features[0,0]=1    # dir
#        #target_modif_features[1,0]=-1   # opposite dir
#        #target_modif_features[0:2,1]=1    # temps
#        ## Laby
#        target_modif_features[0,0]=1
#        target_modif_features[1,0]=0
#        #target_modif_features[2,1]=0
#        #target_modif_features[3,1]=0
#        target_modif_features = np.repeat(target_modif_features,self._batch_size,axis=0)
#        states_val_tiled = []
#        for obs in states_val:
#            states_val_tiled.append(np.tile(obs,(self._n_actions,1,1,1)))
#        onehot_actions_tiled = np.diag(np.ones(self._n_actions))
#        onehot_actions_tiled = np.repeat(onehot_actions_tiled,self._batch_size,axis=0)
#        loss_interpret = torch.nn.functional.mse_loss(
#            self.force_features(states_val_tiled+[onehot_actions_tiled]),
#            target_modif_features
#            ) 
#        self.loss_interpret += loss_interpret.item()

#        all_losses = loss_T + loss_R + loss_gamma + loss_disambiguate1 + \
#            loss_disambiguate2 + loss_disentangle_t #+ loss_interpret
        all_losses = loss_T + -1*1E-1*loss_disentangle_t + -1*1E-2*loss_disambiguate2
        self.tracked_losses.append(all_losses.item())
        all_losses.backward()
        self.optimizer.step()
        self.update_counter += 1

        if(self.update_counter%500==0):
            print ("self.loss_T/500., self.lossR/500., self.loss_gamma/500., self.loss_Q/500., self.loss_disentangle_t/500., self.loss_disambiguate1/500., self.loss_disambiguate2/500.")
            print (self.loss_T/500., self.loss_R/500.,self.loss_gamma/500., self.loss_Q/500., self.loss_disentangle_t/500., self.loss_disambiguate1/500., self.loss_disambiguate2/500.)

            if(self._high_int_dim==False):
                print ("self.loss_interpret/500.")
                print (self.loss_interpret/500.)

            self.lossR=0
            self.loss_gamma=0
            self.loss_Q=0
            self.loss_T=0
            self.loss_interpret=0

            self.loss_disentangle_t=0
            self.loss_disambiguate1=0
            self.loss_disambiguate2=0

        # Q network stuff
        return 0., 0.

    def step_scheduler(self):
        print("adjusting learning rate!!!")
        self.scheduler.step()

    def qValues(self, state_val):
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

        return None

    def qValues_planning(self, state_val, R, gamma, T, Q, d=5):
        """ Get the average Q-values up to planning depth d for one pseudo-state.
        
        Arguments
        ---------
        state_val : array of objects (or list of objects)
            Each object is a numpy array that relates to one of the observations
            with size [1 * history size * size of punctual observation (which is 2D,1D or scalar)]).
        R : float_model
            Model that fits the reward
        gamma : float_model
            Model that fits the discount factor
        T : transition_model
            Model that fits the transition between abstract representation
        Q : Q_model
            Model that fits the optimal Q-value
        d : int
            planning depth

        Returns
        -------
        The average q values with planning depth up to d for the provided pseudo-state
        """

        return None

    def qValues_planning_abstr(self, state_abstr_val, R, gamma, T, Q, d, branching_factor=None):
        """ Get the q values for pseudo-state(s) with a planning depth d. 
        This function is called recursively by decreasing the depth d at every step.

        Arguments
        ---------
        state_abstr_val : internal state(s).
        R : float_model
            Model that fits the reward
        gamma : float_model
            Model that fits the discount factor
        T : transition_model
            Model that fits the transition between abstract representation
        Q : Q_model
            Model that fits the optimal Q-value
        d : int
            planning depth

        Returns
        -------
        The Q-values with planning depth d for the provided encoded state(s)
        """

        return None

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
        copy_state=copy.deepcopy(state) #Required because of the "hack" below

        if(mode==None):
            mode=0
        di=[0,1,3,6]
        # We use the mode to define the planning depth
        q_vals = self.qValues_planning([np.expand_dims(s,axis=0) for s in copy_state],self.R,self.gamma, self.transition, self.Q, d=di[mode])

        return np.argmax(q_vals),np.max(q_vals)

    def _compile(self):
        """ Compile all the optimizers for the different losses
        """
        pass

    def _resetQHat(self):
        """ Set the target Q-network weights equal to the main Q-network weights
        """
        pass

    def setLearningRate(self, lr):
        """ Setting the learning rate

        Parameters
        -----------
        lr : float
            The learning rate that has to be set
        """
        pass

    def transfer(self, original, transfer, epochs=1):
        pass
