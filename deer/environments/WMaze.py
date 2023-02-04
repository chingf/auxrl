""" Simple maze environment

"""
import numpy as np

from deer.base_classes import Environment
from sklearn.decomposition import PCA
import torch
import matplotlib
import os
#matplotlib.use('agg')
# matplotlib.use('qt5agg')
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.patches import Circle, Rectangle
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, DrawingArea, HPacker            
import copy 

class MyEnv(Environment):
    VALIDATION_MODE = 0

    def __init__(self, rng, **kwargs):
        """
        Alternating W-maze as in Kastner and Dayan 2022. Arm length n and
        arm spacing m determines maze size:

        X  X  X  X  X  X   There are always six arms. Here, n=3 and m=2.
        X  X  X  X  X  X   In total, width = 5m + 6 and length = n+1
        X  X  X  X  X  X
        XXXXXXXXXXXXXXXX

        """ # TODO: splitting tracking

        self._mode = -1
        self._mode_score = 0.0
        self._mode_episode_count = 0
        self._arm_len = kwargs.get("arm_len", 3)
        self._arm_spacing = kwargs.get("arm_spacing", 1) # Should be odd
        self._maze_width = 5*self._arm_spacing + 6
        self._maze_length = self._arm_len + 1
        self._higher_dim_obs = kwargs.get("higher_dim_obs", True)
        self._reward = kwargs.get("reward", True)
        self._contingency = kwargs.get("contingency", [2,3]) #[2,3,4])
        self._dimensionality_tracking = []
        self._dimensionality_variance_ratio = None
        self._reward_location = 0 # Just a placeholder
        self.create_map()
        if not self._higher_dim_obs:
            self._expansion = np.random.normal(size=(32, 2))
            self._obs_map = {}
            for x in range(self._maze_width):
                self._obs_map[x] = {}
                for y in range(self._maze_length):
                    position = np.array([x, y])
                    obs = self._expansion @ position
                    obs[obs < 0] = 0
                    self._obs_map[x][y] = obs

    def create_map(self):
        self._map = np.zeros((self._maze_width+2, self._maze_length+2))
        self._map[-1,:] = 1; self._map[0,:] = 1 # Borders
        self._map[:,0] = 1; self._map[:,-1] = 1 # Borders
        for wall in range(2, self._maze_width, self._arm_spacing+1):
            self._map[wall,2:] = 1
        self._pos_agent = [int(self._maze_width//2), 1]
        self._goal_arm = self._contingency[0]
        self._goal_idx = 0
        self._arm_visits = [np.nan] * len(self._contingency)
        self._x_coord_to_arm = {}
        self._arm_to_x_coord = {}
        for arm, arm_x_coord in enumerate(
            range(1, self._maze_width+1, self._arm_spacing+1)):
            self._x_coord_to_arm[arm_x_coord] = arm
            self._arm_to_x_coord[arm] = arm_x_coord

    def set_new_contingency(self, sequence):
        self._contingency = sequence
                
    def reset(self, mode):
        self.create_map()
        if mode == MyEnv.VALIDATION_MODE:
            if self._mode != MyEnv.VALIDATION_MODE:
                self._mode = MyEnv.VALIDATION_MODE
                self._mode_score = 0.0
                self._mode_episode_count = 0
            else:
                self._mode_episode_count += 1
        elif self._mode != -1:
            self._mode = -1
        return [1 * [self._maze_width * [self._maze_length * [0]]]]

    def act(self, action):
        """Applies the agent action [action] on the environment.

        Parameters
        -----------
        action : int
            The action selected by the agent to operate on the environment.
            Should be an identifier in [0, nActions())
        """

        self._cur_action = action
        reward = 0
        if action == 0:
            if self._map[self._pos_agent[0]-1,self._pos_agent[1]] != 1:
                self._pos_agent[0] = self._pos_agent[0] - 1
        elif action == 1:
            if self._map[self._pos_agent[0]+1,self._pos_agent[1]] != 1:
                self._pos_agent[0] = self._pos_agent[0] + 1
        elif action == 2:
            if self._map[self._pos_agent[0],self._pos_agent[1]-1] != 1:
                self._pos_agent[1] = self._pos_agent[1] - 1
        elif action == 3:
            if self._map[self._pos_agent[0],self._pos_agent[1]+1] != 1:
                self._pos_agent[1] = self._pos_agent[1] + 1

        # Update arm visit queue if at the end of an newly visited arm
        if self._pos_agent[1] == self._maze_length:
            curr_arm = self._x_coord_to_arm[self._pos_agent[0]]
            if self._arm_visits[-1] != curr_arm:
                self._arm_visits[:-1] = self._arm_visits[1:]
                self._arm_visits[-1] = curr_arm
                if curr_arm == self._goal_arm:
                    reward = 1
                    self._goal_idx += 1
                else:
                    self._goal_idx = 0 

        if self._reward:
            if np.array_equal(self._arm_visits, self._contingency):
                #reward = 1
                self._arm_visits = [np.nan]*len(self._arm_visits)
                self._goal_idx = 0
            else:
                pass
                #reward = 0
        self._mode_score += reward
        self._goal_arm = self._contingency[self._goal_idx]
        return reward

    def summarizePerformance(
        self, test_data_set, learning_algo, fname, fig_dir_root='./'
        ):
        """
        Plot of the low-dimensional representation of the environment
        built by the model
        """

        if fname is None:
            fig_dir = './'
        else:
            fig_dir = f'{fig_dir_root}figs/{fname}/'
            if not os.path.isdir(fig_dir):
                os.makedirs(fig_dir)

        # Only seen states
        labels = [] # which quadrant
        intern_dim = learning_algo._internal_dim
        observations = test_data_set.observations()[0]
        observations_tcm = []
        mem_len = learning_algo._mem_len
        for t in np.arange(mem_len, observations.shape[0]):
            tcm_obs = np.swapaxes(observations[t-mem_len:t], 0, 1)
            tcm_obs = learning_algo.make_state_with_history(tcm_obs)
            observations_tcm.append(tcm_obs.detach().numpy().squeeze())
            labels.append(0)
        hlen = 500
        observations_tcm = np.array(observations_tcm, dtype='float')[-hlen:]
        unique_observations_tcm, unique_idxs = np.unique(
            observations_tcm, axis=0, return_index=True)
        labels = np.array(labels)[unique_idxs]
        device = learning_algo.device
        n = unique_observations_tcm.shape[0]
        all_possib_inp = torch.tensor(unique_observations_tcm).float().to(device)
        with torch.no_grad():
            abs_states = learning_algo.crar.encoder(
                torch.tensor(all_possib_inp).float().to(device)
                )
        abs_states_np = abs_states.cpu().numpy()
        if abs_states.ndim == 4: # data_format='channels_last' --> 'channels_first'
            abs_states = np.transpose(abs_states_np, (0, 3, 1, 2))
        labels = np.array(labels)
       
        if not self.inTerminalState():
            self._mode_episode_count += 1
        print("== Mean score per episode is {} over {} episodes ==".format(self._mode_score / (self._mode_episode_count+0.0001), self._mode_episode_count))
        m = cm.ScalarMappable(cmap=cm.jet)
       
        if intern_dim == 2:
            x = np.array(abs_states_np)[:,0]
            y = np.array(abs_states_np)[:,1]
            z = np.zeros(y.shape)
        elif intern_dim == 3:
            x = np.array(abs_states_np)[:,0]
            y = np.array(abs_states_np)[:,1]
            z = np.array(abs_states_np)[:,2]
        else:
            if abs_states_np.shape[0] > 2:
                pca = PCA()
                reduced_states = pca.fit_transform(abs_states_np)
                x = np.array(reduced_states)[:,0]
                y = np.array(reduced_states)[:,1]
                z = np.array(reduced_states)[:,2]
            else:
                x = np.array(abs_states_np)[:,0]
                y = np.array(abs_states_np)[:,1]
                z = np.array(abs_states_np)[:,2]
        fig = plt.figure()
        if intern_dim == 2:
            ax = fig.add_subplot(111)
            ax.set_xlabel(r'$X_1$')
            ax.set_ylabel(r'$X_2$')
        else:
            ax = fig.add_subplot(111,projection='3d')
            ax.set_xlabel(r'$X_1$')
            ax.set_ylabel(r'$X_2$')
            ax.set_zlabel(r'$X_3$')
                    
        # Plot the estimated transitions
        n = abs_states.shape[0]
        action_colors = ["0.9", "0.65", "0.4", "0.15"]
        for i in range(n-1):
            n_actions = 4
            for action in range(n_actions):
                action_encoding = np.zeros(n_actions)
                action_encoding[action] = 1
                with torch.no_grad():
                    pred = learning_algo.crar.transition(torch.cat([
                        abs_states[i:i+1].to(device),
                        torch.as_tensor([action_encoding]).to(device)
                        ], dim=1).float()).cpu().numpy()
                if (intern_dim > 3) and (abs_states_np.shape[0] > 2):
                    pred = pca.transform(pred)
                x_transitions = np.concatenate([x[i:i+1], pred[0,:1]])
                y_transitions = np.concatenate([y[i:i+1], pred[0,1:2]])
                if intern_dim == 2:
                    z_transitions = np.zeros(y_transitions.shape)
                else:
                    z_transitions = np.concatenate([z[i:i+1],pred[0,2:3]])
                ax.plot(
                    x_transitions, y_transitions, z_transitions,
                    color=action_colors[action], alpha=0.75)
        
        # Plot the dots at each time step depending on the action taken
        colors=['blue', 'orange', 'green', 'red']
        for i in range(4): # For each quadrant
            label_idxs = labels == i
            if intern_dim == 2:
                line3 = ax.scatter(x[label_idxs], y[label_idxs],
                    c=colors[i], marker='x', edgecolors='k', alpha=0.5, s=100)
            else:
                line3 = ax.scatter(x[label_idxs], y[label_idxs], z[label_idxs],
                    c=colors[i], marker='x', depthshade=True, edgecolors='k',
                    alpha=0.5, s=50)
        
        # Plot the legend for transition estimates
        box1b = TextArea(" Estimated transitions (action 0, 1, 2 and 3): ", textprops=dict(color="k"))
        box2b = DrawingArea(90, 20, 0, 0)
        el1b = Rectangle((5, 10), 15,2, fc=action_colors[0], alpha=0.75)
        el2b = Rectangle((25, 10), 15,2, fc=action_colors[1], alpha=0.75) 
        el3b = Rectangle((45, 10), 15,2, fc=action_colors[2], alpha=0.75)
        el4b = Rectangle((65, 10), 15,2, fc=action_colors[3], alpha=0.75) 
        box2b.add_artist(el1b)
        box2b.add_artist(el2b)
        box2b.add_artist(el3b)
        box2b.add_artist(el4b)
        boxb = HPacker(children=[box1b, box2b], align="center", pad=0, sep=5)
        anchored_box = AnchoredOffsetbox(
            loc=3, child=boxb, pad=0., frameon=True,
            bbox_to_anchor=(0., 0.98), bbox_transform=ax.transAxes,
            borderpad=0.)        
        ax.add_artist(anchored_box)
        plt.savefig(f'{fig_dir}latents.pdf')

        # Plot continuous measure of dimensionality
        if (intern_dim > 3) and (abs_states_np.shape[0] > 2):
            variance_curve = np.cumsum(pca.explained_variance_ratio_)
            auc = np.trapz(variance_curve, dx=1/variance_curve.size)
            self._dimensionality_tracking.append(auc)
            self._dimensionality_variance_ratio = pca.explained_variance_ratio_
            plt.figure()
            plt.plot(self._dimensionality_tracking)
            plt.ylabel('AUC of PCA Explained Variance Ratio')
            plt.xlabel('Epochs')
            plt.savefig(f'{fig_dir}dimensionality.pdf')
        else:
            self._dimensionality_tracking.append(-1)

        # Plot losses over epochs
        losses, loss_names = learning_algo.get_losses()
        loss_weights = learning_algo._loss_weights
        _, axs = plt.subplots(3, 2, figsize=(7, 10))
        for i in range(5):
            loss = losses[i]; loss_name = loss_names[i]
            ax = axs[i%3][i//3]
            ax.plot(loss)
            ax.set_ylabel(loss_name)
        plt.tight_layout()
        plt.savefig(f'{fig_dir}losses.pdf', dpi=300)
        _, axs = plt.subplots(3, 2, figsize=(7, 10))
        for i in range(5):
            loss = losses[i]; loss_name = loss_names[i]
            ax = axs[i%3][i//3]
            ax.plot(np.array(loss)*loss_weights[i])
            ax.set_ylabel(loss_name)
        plt.tight_layout()
        plt.savefig(f'{fig_dir}scaled_losses.pdf', dpi=300)
        plt.figure()
        plt.plot(losses[-1])
        plt.title('Total Loss')
        plt.savefig(f'{fig_dir}total_losses.pdf', dpi=300)
        matplotlib.pyplot.close("all") # avoid memory leaks

    def inputDimensions(self):
        if self._higher_dim_obs:
            return [(1, self._maze_width, self._maze_length)]
        else:
            return [(1, 32)]

    def observationType(self, subject):
        return np.float

    def nActions(self):
        return 4

    def observe(self):
        if self._higher_dim_obs:
            obs = self.get_higher_dim_obs(self._pos_agent)
        else:
            obs = self.get_low_dim_obs(self._pos_agent)
        return [obs]

    def get_low_dim_obs(self, pos_agent):
        return self._obs_map[self._pos_agent[0]][self._pos_agent[1]].copy()
    
    def get_higher_dim_obs(self, pos_agent):
        """
        Go from box-visualization to a humanoid agent representation
        """


        obs = copy.deepcopy(self._map)
        obs[pos_agent[0], pos_agent[1]] = 0.5
        pos_goal = [self._arm_to_x_coord[self._goal_arm], self._maze_length]
        obs[pos_goal[0], pos_goal[1]] = 10.
        return obs[1:-1, 1:-1]

        obs = copy.deepcopy(self._map)
        obs = obs/1.
        obs = np.repeat(np.repeat(obs, 6, axis=0),6, axis=1)
        agent_obs = np.zeros((6,6))
        agent_obs[0,2] = 0.7
        agent_obs[1,0:5] = 0.8
        agent_obs[2,1:4] = 0.8
        agent_obs[3,1:4] = 0.8
        agent_obs[4,1] = 0.8
        agent_obs[4,3] = 0.8
        agent_obs[5,0:2] = 0.8
        agent_obs[5,3:5] = 0.8
        x, y = pos_agent
        obs[x*6:(x+1)*6:, y*6:(y+1)*6] = agent_obs
        return obs

    def inTerminalState(self):
        return False

if __name__ == "__main__":
    pass
