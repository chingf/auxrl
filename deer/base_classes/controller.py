"""This file defines the base Controller class and some presets controllers that you can use for controlling 
the training and the various parameters of your agents.

Controllers can be attached to an agent using the agent's ``attach(Controller)`` method. The order in which controllers 
are attached matters. Indeed, if controllers C1, C2 and C3 were attached in this order and C1 and C3 both listen to the
onEpisodeEnd signal, the onEpisodeEnd() method of C1 will be called *before* the onEpisodeEnd() method of C3, whenever 
an episode ends.

.. Authors: Vincent Francois-Lavet, David Taralla
"""

class Controller(object):
    """A base controller that does nothing when receiving the various signals emitted by an agent. This class should 
    be the base class of any controller you would want to define.
    """

    def __init__(self):
        """Activate this controller.

        All controllers inheriting this class should call this method in their own __init()__ using 
        super(self.__class__, self).__init__().
        """

        self._active = True
        self._modes = [-1]

    def setActive(self, active):
        """Activate or deactivate this controller.
        
        A controller should not react to any signal it receives as long as it is deactivated. For instance, if a 
        controller maintains a counter on how many episodes it has seen, this counter should not be updated when 
        this controller is disabled.
        """

        self._active = active

    def onStart(self, agent):
        """Called when the agent is going to start working (before anything else).
        
        This corresponds to the moment where the agent's run() method is called.

        Parameters
        ----------
             agent : NeuralAgent
                The agent firing the event
        """

        pass

    def onEpisodeEnd(self, agent, terminal_reached, reward):
        """Called whenever the agent ends an episode, just after this episode ended and before any onEpochEnd() signal
        could be sent.

        Parameters
        ----------
        agent : NeuralAgent
            The agent firing the event
        terminal_reached : bool
            Whether the episode ended because a terminal transition occured. This could be False 
            if the episode was stopped because its step budget was exhausted.
        reward : float
            The reward obtained on the last transition performed in this episode.
        
        """

        pass

    def onEpochEnd(self, agent):
        """Called whenever the agent ends an epoch, just after the last episode of this epoch was ended and after any 
        onEpisodeEnd() signal was processed.

        Parameters
        ----------
        agent : NeuralAgent
            The agent firing the event
        """

        pass

    def onActionChosen(self, agent, action):
        """Called whenever the agent has chosen an action.

        This occurs after the agent state was updated with the new observation it made, but before it applied this 
        action on the environment and before the total reward is updated.
        """

        pass

    def onActionTaken(self, agent):
        """Called whenever the agent has taken an action on its environment.

        This occurs after the agent applied this action on the environment and before terminality is evaluated. This 
        is called only once, even in the case where the agent skip frames by taking the same action multiple times.
        In other words, this occurs just before the next observation of the environment.
        """

        pass

    def onEnd(self, agent):
        """Called when the agent has finished processing all its epochs, just before returning from its run() method.
        """

        pass
