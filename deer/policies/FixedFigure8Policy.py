import numpy as np
from ..base_classes import Policy


class FixedFigure8Policy(Policy):
    HEIGHT = 6
    WIDTH = 7

    """The policy acts greedily with probability :math:`1-\epsilon` and acts randomly otherwise.
    It is now used as a default policy for the neural agent.

    Parameters
    -----------
    epsilon : float
        Proportion of random steps
    """
    def __init__(self, learning_algo, n_actions, random_state, epsilon=0):
        Policy.__init__(self, learning_algo, n_actions, random_state)
        self._epsilon = epsilon
        self.left_turn = True

    def action(self, state, mode=None, *args, **kwargs):
        if self.random_state.rand() < self._epsilon:
            action, V = self.randomAction()
        else:
            state = state[0].squeeze()
            if len(state.shape) == 1:
                state = state.reshape(
                    (FixedFigure8Policy.WIDTH, FixedFigure8Policy.HEIGHT)
                    )
            agent_x, agent_y = np.argwhere(state==10)[0]
    
            if (agent_y == 0) and (agent_x < 3):
                action = 1
            elif (agent_y == 0) and (agent_x > 3):
                action = 0
            elif (agent_y < 5) and (agent_x == 3):
                action = 2
            elif (agent_y > 0) and (agent_x == 0):
                action = 3
            elif (agent_y > 0) and (agent_x == 6):
                action = 3
            elif (agent_y == 5) and (agent_x < 3):
                action = 0
            elif (agent_y == 5) and (agent_x > 3):
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
