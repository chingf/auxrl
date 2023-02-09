"""This file defines the base Controller class and some presets controllers that you can use for controlling 
the training and the various parameters of your agents.

Controllers can be attached to an agent using the agent's ``attach(Controller)`` method. The order in which controllers 
are attached matters. Indeed, if controllers C1, C2 and C3 were attached in this order and C1 and C3 both listen to the
onEpisodeEnd signal, the onEpisodeEnd() method of C1 will be called *before* the onEpisodeEnd() method of C3, whenever 
an episode ends.

.. Authors: Vincent Francois-Lavet, David Taralla
"""
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from ..base_classes import Controller

class LearningRateController(Controller):
    """A controller that modifies the learning rate periodically upon epochs end (only in training mode, i.e., agent.mode() == -1).
    
    Parameters
    ----------
    initial_learning_rate : float
        The learning rate upon agent start
    learning_rate_decay : float
        The factor by which the previous learning rate is multiplied every [periodicity] epochs.
    periodicity : int
        How many epochs are necessary before an update of the learning rate occurs
    """

    def __init__(self, initial_learning_rate=0.005, learning_rate_decay=1., periodicity=1):
        """Initializer.

        """
        super(self.__class__, self).__init__()
        self._epoch_count = 0
        self._init_lr = initial_learning_rate
        self._lr = initial_learning_rate
        self._lr_decay = learning_rate_decay
        self._periodicity = periodicity

    def onStart(self, agent):
        if (self._active == False) or (agent.mode() not in self._modes):
            return

        self._epoch_count = 0
        agent._learning_algo.setLearningRate(self._init_lr)
        self._lr = self._init_lr * self._lr_decay

    def onEpochEnd(self, agent):
        if (self._active == False) or (agent.mode() not in self._modes):
            return
        self._epoch_count += 1
        #if self._epoch_count == 1:
        #    return
        if self._periodicity <= 1 or self._epoch_count % self._periodicity == 0:
            agent._learning_algo.setLearningRate(self._lr)
            self._lr *= self._lr_decay
        agent._learning_algo.step_scheduler()

class EpsilonController(Controller):
    """ A controller that modifies the probability "epsilon" of taking a random action periodically (only in training mode, i.e., agent.mode() == -1).
    
    Parameters
    ----------
    initial_e : float
        Start epsilon
    e_decays : int
        How many updates are necessary for epsilon to reach eMin
    e_min : float
        End epsilon
    evaluate_on : str
        After what type of event epsilon should be updated periodically. Possible values: 'action', 'episode', 'epoch'.
    periodicity : int
        How many [evaluateOn] are necessary before an update of epsilon occurs
    reset_every : str
        After what type of event epsilon should be reset to its initial value. Possible values: 
        'none', 'episode', 'epoch'.
    """

    def __init__(self, initial_e=1., e_decays=10000, e_min=0.1, evaluate_on='action', periodicity=1, reset_every='none'):
        """Initializer.
        """

        super(self.__class__, self).__init__()
        self._count = 0
        self._init_e = initial_e
        self._e = initial_e
        self._e_min = e_min
        self._e_decay = (initial_e - e_min) / e_decays
        self._periodicity = periodicity

        self._on_action = 'action' == evaluate_on
        self._on_episode = 'episode' == evaluate_on
        self._on_epoch = 'epoch' == evaluate_on
        if not self._on_action and not self._on_episode and not self._on_epoch:
            self._on_action = True

        self._reset_on_episode = 'episode' == reset_every
        self._reset_on_epoch = 'epoch' == reset_every

    def onStart(self, agent):
        if (self._active == False) or (agent.mode() not in self._modes):
            return

        self._reset(agent)

    def onEpisodeEnd(self, agent, terminal_reached, reward):
        if (self._active == False) or (agent.mode() not in self._modes):
            return

        if self._reset_on_episode:
           self. _reset(agent)
        elif self._on_episode:
            self._update(agent)

    def onEpochEnd(self, agent):
        if (self._active == False) or (agent.mode() not in self._modes):
            return

        if self._reset_on_epoch:
            self._reset(agent)
        elif self._on_epoch:
            self._update(agent)

    def onActionChosen(self, agent, action):
        if (self._active == False) or (agent.mode() not in self._modes):
            return

        if self._on_action:
            self._update(agent)

    def _reset(self, agent):
        self._count = 0
        agent._train_policy.setEpsilon(self._init_e)
        self._e = self._init_e

    def _update(self, agent):
        self._count += 1
        if self._periodicity <= 1 or self._count % self._periodicity == 0:
            agent._train_policy.setEpsilon(self._e)
            self._e = max(self._e - self._e_decay, self._e_min)

class DiscountFactorController(Controller):
    """A controller that modifies the q-network discount periodically (only in training mode, i.e., agent.mode() == -1).
    More informations in : Francois-Lavet Vincent et al. (2015) - How to Discount Deep Reinforcement Learning: Towards New Dynamic Strategies (http://arxiv.org/abs/1512.02011).

    Parameters
    ----------
    initial_discount_factor : float
        Start discount
    discount_factor_growth : float
        The factor by which the previous discount is multiplied every [periodicity]
        epochs.
    discount_factor_max : float
        Maximum reachable discount
    periodicity : int
        How many training epochs are necessary before an update of the discount occurs
    """
    
    def __init__(self, initial_discount_factor=0.9, discount_factor_growth=1., discount_factor_max=0.99, periodicity=1):
        """Initializer.
        """

        super(self.__class__, self).__init__()
        self._epoch_count = 0
        self._init_df = initial_discount_factor
        self._df = initial_discount_factor
        self._df_growth = discount_factor_growth
        self._df_max = discount_factor_max
        self._periodicity = periodicity

    def onStart(self, agent):
        if (self._active == False) or (agent.mode() not in self._modes):
            return

        self._epoch_count = 0
        agent._learning_algo.setDiscountFactor(self._init_df)
        if (self._init_df < self._df_max):
            self._df = 1 - (1 - self._init_df) * self._df_growth
        else:
            self._df = self._init_df

    def onEpochEnd(self, agent):
        if (self._active == False) or (agent.mode() not in self._modes):
            return

        self._epoch_count += 1
        if self._periodicity <= 1 or self._epoch_count % self._periodicity == 0:
            if (self._df < self._df_max):
                agent._learning_algo.setDiscountFactor(self._df)
                self._df = 1 - (1 - self._df) * self._df_growth

class InterleavedTestEpochController(Controller):
    """A controller that interleaves a valid/test epoch between training epochs of the agent (only in training mode, i.e., agent.mode() == -1).
    
    Parameters
    ----------
    id : int
        The identifier (>= 0) of the mode each test epoch triggered by this controller will belong to. 
        Can be used to discriminate between datasets in your Environment subclass (this is the argument that 
        will be given to your environment's reset() method when starting the test epoch).
    epoch_length : float
        The total number of transitions that will occur during a test epoch. This means that
        this epoch could feature several episodes if a terminal transition is reached before this budget is 
        exhausted.
    periodicity : int 
        How many train epochs are necessary before a valid/test epoch is ran.
    show_score : bool
        Whether to print an informative message on stdout at the end of each test epoch, about 
        the total reward obtained in the course of the test epoch.
    summarize_every : int
        How many of this controller's test epochs are necessary before the attached agent's 
        summarizeTestPerformance() method is called. Give a value <= 0 for "never". If > 0, the first call will
        occur just after the first test epoch.
    """

    def __init__(
        self, id=0, epoch_length=500, periodicity=1, show_score=True,
        summarize_every=10, unique_fname=None
        ):
        super(self.__class__, self).__init__()
        self._epoch_count = 0
        self._id = id
        self._epoch_length = epoch_length
        self._show_score = show_score
        self._periodicity = periodicity
        self._summary_counter = 0
        self._summary_periodicity = summarize_every
        self._unique_fname = unique_fname
        self.scores = []
        self.episodes = []

    def onStart(self, agent):
        if (self._active == False) or (agent.mode() not in self._modes):
            return

        self._epoch_count = 0

    def onEpochEnd(self, agent):
        if (self._active == False) or (agent.mode() not in self._modes):
            return

        mod = self._epoch_count % self._periodicity
        self._epoch_count += 1
        if mod == 0:
            agent.startMode(self._id, self._epoch_length)
            agent._run_non_train(n_epochs=1, epoch_length=self._epoch_length)
            self._summary_counter += 1

            if self._show_score:
                score, _, nbr_episodes = agent.totalRewardOverLastTest()
                self.scores.append(score)
                self.episodes.append(nbr_episodes)
            if (self._summary_periodicity > 0) and (self._summary_counter % self._summary_periodicity == 0):
                print('SUMMARIZING TEST PERF')
                agent.summarizeTestPerformance(self._unique_fname)
            agent.resumeTrainingMode()

class TrainerController(Controller):
    """A controller that makes the agent train on its current database periodically (only in training mode, i.e., agent.mode() == -1).

    Parameters
    ----------
    evaluate_on : str
        After what type of event the agent should be trained periodically. Possible values: 
        'action', 'episode', 'epoch'. The first training will occur after the first occurrence of [evaluateOn].
    periodicity : int
        How many [evaluateOn] are necessary before a training occurs
        _show_avg_Bellman_residual [bool] - Whether to show an informative message after each episode end (and after a 
        training if [evaluateOn] is 'episode') about the average bellman residual of this episode
    show_episode_avg_V_value : bool
        Whether to show an informative message after each episode end (and after a 
        training if [evaluateOn] is 'episode') about the average V value of this episode
    """
    def __init__(self, evaluate_on='action', periodicity=1, show_episode_avg_V_value=True, show_avg_Bellman_residual=True):
        """Initializer.
        """

        super(self.__class__, self).__init__()
        self._count = 0
        self._periodicity = periodicity
        self._show_avg_Bellman_residual = show_avg_Bellman_residual
        self._show_episode_avg_V_value = show_episode_avg_V_value

        self._on_action = 'action' == evaluate_on
        self._on_episode = 'episode' == evaluate_on
        self._on_epoch = 'epoch' == evaluate_on
        if not self._on_action and not self._on_episode and not self._on_epoch:
            self._on_action = True

    def onStart(self, agent):
        if (self._active == False) or (agent.mode() not in self._modes):
            return
        
        self._count = 0

    def onEpisodeEnd(self, agent, terminal_reached, reward):
        if (self._active == False) or (agent.mode() not in self._modes):
            return
        
        if self._on_episode:
            self._update(agent)

        if self._show_avg_Bellman_residual: print("Average (on the epoch) training loss: {}".format(agent.avgBellmanResidual()))
        if self._show_episode_avg_V_value: print("Episode average V value: {}".format(agent.avgEpisodeVValue())) # (on non-random action time-steps)

    def onEpochEnd(self, agent):
        if (self._active == False) or (agent.mode() not in self._modes):
            return

        if self._on_epoch:
            self._update(agent)

    def onActionTaken(self, agent):
        if (self._active == False) or (agent.mode() not in self._modes):
            return

        if self._on_action:
            self._update(agent)

    def _update(self, agent):
        if self._periodicity <= 1 or self._count % self._periodicity == 0:
            agent.train()
        self._count += 1

class VerboseController(Controller):
    """A controller that print various agent information periodically:
    
    * Count of passed [evaluateOn]
    * Agent current learning rate
    * Agent current discount factor
    * Agent current epsilon

    Parameters
    ----------
    evaluate_on : str
        After what type of event the printing should occur periodically. Possible values: 
        'action', 'episode', 'epoch'. The first printing will occur after the first occurrence of [evaluateOn].
    periodicity : int
        How many [evaluateOn] are necessary before a printing occurs
    modes : list of int
        List of agent modes for which this controller is used
    """

    def __init__(self, evaluateOn=False, evaluate_on='epoch', periodicity=1, modes=[-1]):
        """Initializer.
        """
        if evaluateOn is not False:
            raise Exception('For uniformity the attributes to be provided to the controllers respect PEP8 from deer0.3dev1 onwards. For instance, instead of "evaluateOn", you should now have "evaluate_on". Please have a look at https://github.com/VinF/deer/issues/28.')

        super(self.__class__, self).__init__()
        self._modes = modes
        self._count = 0
        self._periodicity = periodicity
        self._string = evaluate_on

        self._on_action = 'action' == evaluate_on
        self._on_episode = 'episode' == evaluate_on
        self._on_epoch = 'epoch' == evaluate_on
        if not self._on_action and not self._on_episode and not self._on_epoch:
            self._on_epoch = True

    def onStart(self, agent):
        if (self._active == False) or (agent.mode() not in self._modes):
            return
        
        self._count = 0

    def onEpisodeEnd(self, agent, terminal_reached, reward):
        if (self._active == False) or (agent.mode() not in self._modes):
            return
        
        if self._on_episode:
            self._print(agent)

    def onEpochEnd(self, agent):
        if (self._active == False) or (agent.mode() not in self._modes):
            return

        if self._on_epoch:
            self._print(agent)

    def onActionTaken(self, agent):
        if (self._active == False) or (agent.mode() not in self._modes):
            return

        if self._on_action:
            self._print(agent)

    def _print(self, agent):
        if self._periodicity <= 1 or self._count % self._periodicity == 0:
            print("{} {}:".format(self._string, self._count + 1))
            print("Learning rate: {}".format(agent._learning_algo.learningRate()))
            print("Discount factor: {}".format(agent._learning_algo.discountFactor()))
            print("Epsilon: {}".format(agent._train_policy.epsilon()))
        self._count += 1

class FindBestController(Controller):
    """A controller that finds the neural net performing at best in validation mode (i.e. for mode = [validationID]) 
    and computes the associated generalization score in test mode (i.e. for mode = [testID], and this only if [testID] 
    is different from None). This controller should never be disabled by InterleavedTestControllers as it is meant to 
    work in conjunction with them.
    
    At each epoch end where this controller is active, it will look at the current mode the agent is in. 
    
    If the mode matches [validationID], it will take the total reward of the agent on this epoch and compare it to its 
    current best score. If it is better, it will ask the agent to dump its current nnet on disk and update its current 
    best score. In all cases, it saves the validation score obtained in a vector.

    If the mode matches [testID], it saves the test (= generalization) score in another vector. Note that if [testID] 
    is None, no test mode score are ever recorded.

    At the end of the experiment (onEnd), if active, this controller will print information about the epoch at which 
    the best neural net was found together with its generalization score, this last information shown only if [testID] 
    is different from None. Finally it will dump a dictionary containing the data of the plots ({n: number of 
    epochs elapsed, ts: test scores, vs: validation scores}). Note that if [testID] is None, the value dumped for the
    'ts' key is [].
    
    Parameters
    ----------
    validationID : int 
        See synopsis
    testID : int 
        See synopsis
    unique_fname : str
        A unique filename (basename for score and network dumps).
    """

    def __init__(
        self, validationID=0, testID=None, unique_fname="nnet", plotfig=False,
        savefrequency=np.inf # 0: best valid score, n: save every n, np.inf: at end
        ):
        super(self.__class__, self).__init__()

        self._validationScores = []
        self._validationSteps = []
        self._validationEps = []
        self._testScores = []
        self._epochNumbers = []
        self._trainingEpochCount = 0
        self._testID = testID
        self._validationID = validationID
        self._filename = f'{unique_fname}'
        self._plotfig = plotfig
        self._savefrequency = savefrequency
        self._fig_dir = f'figs/{unique_fname}/'
        self._bestValidationScoreSoFar = -9999999

    def onEpochEnd(self, agent):
        if (self._active == False):
            return

        mode = agent.mode()
        if mode == self._validationID:
            score, steps, nbr_eps = agent.totalRewardOverLastTest()
            self._validationScores.append(score)
            self._validationSteps.append(steps)
            self._validationEps.append(nbr_eps)
            self._epochNumbers.append(self._trainingEpochCount)
            if score > self._bestValidationScoreSoFar:
                self._bestValidationScoreSoFar = score
                if self._savefrequency == 0:
                    agent.dumpNetwork(self._filename, self._trainingEpochCount)
            if (self._savefrequency > 0) and (self._trainingEpochCount%self._savefrequency == 0):
                agent.dumpNetwork(self._filename, self._trainingEpochCount)
        elif mode == self._testID:
            score, _, _ = agent.totalRewardOverLastTest()
            self._testScores.append(score)
        else:
            self._trainingEpochCount += 1

        #live plotting of reward over time
        if self._plotfig:
            os.makedirs(f"{agent._save_dir}{self._fig_dir}", exist_ok=True)
            if mode == self._validationID:
                plt.plot(
                    range(1, len(self._validationScores)+1),
                    self._validationScores, label="VS", color='b')
                plt.legend()
                plt.xlabel("Number of epochs")
                plt.ylabel("Score")
                plt.savefig(f"{agent._save_dir}{self._fig_dir}validation_scores.pdf")

                plt.close()
            elif mode == self._testID:
                plt.plot(
                    range(1, len(self._testScores)+1),
                    self._testScores, label="TS", color='b')
                plt.legend()
                plt.xlabel("Number of epochs")
                plt.ylabel("Score")
                plt.savefig(f"{agent._save_dir}{self._fig_dir}test_scores.pdf")
                plt.close()
        
    def onEnd(self, agent):
        if (self._active == False):
            return
        bestIndex = np.argmax(self._validationScores)
        if agent.mode() == -1:
            print("Best neural net obtained after {} epochs, with validation score {}".format(bestIndex+1, self._validationScores[bestIndex]))
            if self._testID != None:
                print("Test score of this neural net: {}".format(self._testScores[bestIndex]))
            if np.isinf(self._savefrequency):
                agent.dumpNetwork(self._filename, self._trainingEpochCount)
        os.makedirs(f"{agent._save_dir}{self._fig_dir}", exist_ok=True)
        plt.plot(
            range(1, len(self._validationScores)+1),
            self._validationScores, label="VS", color='b')
        plt.legend()
        plt.xlabel("Number of epochs")
        plt.ylabel("Score")
        plt.savefig(f"{agent._save_dir}{self._fig_dir}validation_scores.pdf")

        plt.close()

if __name__ == "__main__":
    pass