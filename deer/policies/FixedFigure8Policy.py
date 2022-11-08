import numpy as np
from ..base_classes import Policy


class FixedFigure8Policy(Policy):

    """The policy acts greedily with probability :math:`1-\epsilon` and acts randomly otherwise.
    It is now used as a default policy for the neural agent.

    Parameters
    -----------
    epsilon : float
        Proportion of random steps
    """
    def __init__(
            self, learning_algo, n_actions, random_state, epsilon=0,
            height=6, width=7
            ):
        Policy.__init__(self, learning_algo, n_actions, random_state)
        self._epsilon = epsilon
        self.left_turn = True
        self.height = height
        self.width = width

    def action(self, state, mode=None, *args, **kwargs):
        if self.random_state.rand() < self._epsilon:
            action, V = self.randomAction()
        else:
            state = state[0].squeeze()
            if len(state.shape) == 1:
                state = state.reshape((self.width, self.height))
            agent_x, agent_y = np.argwhere(state==10)[0]
            midpoint = self.width//2
            h_extent = self.height - 1
            w_extent = self.width - 1
    
            if (agent_y == 0) and (agent_x < midpoint):
                action = 1
            elif (agent_y == 0) and (agent_x > midpoint):
                action = 0
            elif (agent_y < h_extent) and (agent_x == midpoint):
                action = 2
            elif (agent_y > 0) and (agent_x == 0):
                action = 3
            elif (agent_y > 0) and (agent_x == w_extent):
                action = 3
            elif (agent_y == h_extent) and (agent_x < midpoint):
                action = 0
            elif (agent_y == h_extent) and (agent_x > midpoint):
                action = 1
            else:
                if self.left_turn:
                    action = 0
                    self.left_turn = False
                else:
                    action = 1
                    self.left_turn = True
            V = 0.
        return action, V

    def setEpsilon(self, e):
        """ Set the epsilon used for :math:`\epsilon`-greedy exploration
        """
        self._epsilon = e

    def epsilon(self):
        """ Get the epsilon for :math:`\epsilon`-greedy exploration
        """
        return self._epsilon
