import os
import numpy as np
import torch
import pickle
from deer.base_classes import Environment
from sklearn.decomposition import PCA
from scipy.stats import linregress
import matplotlib
# matplotlib.use('qt5agg')
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
plt.switch_backend('agg') # For remote servers
import copy 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.patches import Circle, Rectangle
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, DrawingArea, HPacker            

class MyEnv(Environment):
    VALIDATION_MODE = 0
    RIGHT_REWARD = 1; LEFT_REWARD = 0; RESET_REWARD = 2
    HEIGHT = 4
    WIDTH = 5 #Must be odd
    LEFT_STEM = 0; CENTRAL_STEM = WIDTH//2; RIGHT_STEM = WIDTH-1

    def __init__(self, give_rewards=False, **kwargs):
        self._give_rewards = give_rewards
        self._mode = -1
        self._mode_score = 0.0
        self._mode_episode_count = 0

        self._height = MyEnv.HEIGHT
        self._width = MyEnv.WIDTH # Must be odd!
        self._high_dim_obs = kwargs.get('high_dim_obs', False)
        self._show_rewards = kwargs.get('show_rewards', True)
        self._obs_noise = kwargs.get('obs_noise', 0.)
        self.x = MyEnv.CENTRAL_STEM
        self.y = 0
        self._reward_location = MyEnv.LEFT_REWARD
        self._last_reward_location = MyEnv.RIGHT_REWARD
        self._space_label = self.make_space_labels()
        self._dimensionality_tracking = []
        self._separability_tracking = [[] for _ in range(3)]
        self._separability_slope = []
        self._separability_matrix = None
        self._agent_loc_map = {}

    def make_space_labels(self):
        space_labels = np.zeros((MyEnv.WIDTH, MyEnv.HEIGHT), dtype=int)
        midpoint = MyEnv.WIDTH//2
        for x in range(MyEnv.WIDTH):
            for y in range(MyEnv.HEIGHT):
                if not self.in_bounds(x, y):
                    space_labels[x,y] = -1
                elif x == midpoint and y == MyEnv.HEIGHT-1: # Decision point
                    space_labels[x,y] = 0
                elif x < midpoint: # Left
                    space_labels[x,y] = 1
                elif x > midpoint: # Right
                    space_labels[x,y] = 2
                elif x == midpoint: # Central stem
                    space_labels[x,y] = 3
                else:
                    raise ValueError('Unconsidered case')
        return space_labels

    def reset(self, mode):
        if mode == MyEnv.VALIDATION_MODE:
            if self._mode != MyEnv.VALIDATION_MODE:
                self._mode = MyEnv.VALIDATION_MODE
                self._mode_score = 0.0
                self._mode_episode_count = 0
            else:
                self._mode_episode_count += 1
        elif self._mode != -1:
            self._mode = -1
        self.x, self.y = (MyEnv.WIDTH//2, 0)
        self._reward_location = MyEnv.LEFT_REWARD
        self._last_rewarded = MyEnv.RIGHT_REWARD
        
        return [1 * [self._height * [self._width * [0]]]]

    def in_bounds(self, x, y):
        if (x < 0) or (x >= self._width) or (y < 0) or (y >= self._height):
            return False
        elif (y==0) or (y==self._height-1):
            return True
        elif x not in [0, (self._width-1)/2, self._width-1]:
            return False
        elif x in [0, (self._width-1)/2, self._width-1]:
            return True
        else:
            print(x)
            print(y)
            raise ValueError("forgot a case")

    def act(self, action):
        """Applies the agent action [action] on the environment.

        Parameters
        -----------
        action : int
            nActions = 4, where action = {0, 1, 2, 3} corresponds to 
            {left, right, top, down}
        """

        if action == 0:
            new_x = self.x - 1; new_y = self.y
        elif action == 1:
            new_x = self.x + 1; new_y = self.y
        elif action == 2:
            new_x = self.x; new_y = self.y + 1
        elif action == 3:
            new_x = self.x; new_y = self.y - 1
        else:
            raise ValueError('Not a valid action.')

        left_reward = (0, MyEnv.HEIGHT-1)
        right_reward = (MyEnv.WIDTH-1, MyEnv.HEIGHT-1)
        reset_reward = (MyEnv.WIDTH//2, 0)

        if self.in_bounds(new_x, new_y):
            if self._reward_location == MyEnv.RESET_REWARD:
                if self._last_reward_location == MyEnv.LEFT_REWARD:
                    if ((self.x, self.y) == left_reward) and (action==1):
                        pass
                    else:
                        self.x = new_x; self.y = new_y
                else:
                    if ((self.x, self.y) == right_reward) and (action==0):
                        pass
                    else:
                        self.x = new_x; self.y = new_y
            else:
                if ((self.x, self.y) == reset_reward) and (action!=2):
                    pass
                else:
                    self.x = new_x; self.y = new_y

        # Move reward location as needed
        if ((self.x, self.y) == left_reward) and self._reward_location == MyEnv.LEFT_REWARD:
            self.reward = 1
            self._reward_location = MyEnv.RESET_REWARD
            self._last_reward_location = MyEnv.LEFT_REWARD
        elif ((self.x, self.y) == right_reward) and self._reward_location == MyEnv.RIGHT_REWARD:
            self.reward = 1
            self._reward_location = MyEnv.RESET_REWARD
            self._last_reward_location = MyEnv.RIGHT_REWARD
        elif ((self.x, self.y) == reset_reward) and self._reward_location == MyEnv.RESET_REWARD:
            self.reward = 1
            if self._last_reward_location == MyEnv.RIGHT_REWARD:
                self._reward_location = MyEnv.LEFT_REWARD
            else:
                self._reward_location = MyEnv.RIGHT_REWARD
        else:
            self.reward = 0

        # Set reward to 0 if it's not a signal to be used
        if not self._give_rewards:
            self.reward = 0
              
        self._mode_score += self.reward
        return self.reward

    def summarizePerformance(
        self, test_data_set, learning_algo, fname, save_dir='./'
        ):
        """ Plot of the low-dimensional representation of the environment built by the model
        """

        if fname is None:
            fig_dir = './'
            latents_dir = './'
        else:
            fig_dir = f'{save_dir}figs/{fname}/'
            latents_dir = f'{save_dir}latents/{fname}/'
            os.makedirs(fig_dir, exist_ok=True)
            os.makedirs(latents_dir, exist_ok=True)

        # Only seen states
        observations = test_data_set.observations()[0]
        reward_locs = test_data_set.reward_locs()
        latents = test_data_set.latents()[0]
        color_labels = []
        y_locations = []
        x_locations = []
        mem_len = learning_algo._mem_len
        image_dims = (MyEnv.WIDTH+2, MyEnv.HEIGHT+2)
        for t in np.arange(observations.shape[0]):
            agent_location = list(np.unravel_index(
                np.argmax(observations[t-1]), image_dims))
            #agent_location = np.argwhere(observations[t-1]==10)[0,1:].tolist()
            agent_location[0] -= 1; agent_location[1] -= 1;
            if self._high_dim_obs:
                agent_location = self._agent_loc_map[str(agent_location)]
            x_locations.append(agent_location[0])
            y_locations.append(agent_location[1])
            color_label = self._space_label[agent_location[0], agent_location[1]]
            color_labels.append(color_label)
        hlen = 250
        observations = observations[-hlen:]
        reward_locs = reward_locs[-hlen:]
        latents = latents[-hlen:]
        color_labels = np.array(color_labels, dtype=int)[-hlen:]
        x_locations = np.array(x_locations)[-hlen:]
        y_locations = np.array(y_locations)[-hlen:]
        latents, unique_idxs = np.unique(
            latents, axis=0, return_index=True)
        observations = observations[unique_idxs]
        reward_locs = reward_locs[unique_idxs].astype(int)
        color_labels = color_labels[unique_idxs]
        x_locations = x_locations[unique_idxs]
        y_locations = y_locations[unique_idxs]
        device = learning_algo.device
        n = latents.shape[0]
    
        actions = test_data_set.actions()[0:n]
        if not self.inTerminalState():
            self._mode_episode_count += 1
        print("== Mean score per episode is {} over {} episodes ==".format(
            self._mode_score/(self._mode_episode_count+0.0001), self._mode_episode_count
            ))
        
        # Get 3D projection of latents
        m = cm.ScalarMappable(cmap=cm.jet)
        intern_dim = learning_algo._internal_dim
        if intern_dim == 2:
            x = np.array(latents)[:,0]
            y = np.array(latents)[:,1]
            z = np.zeros(y.shape)
        elif intern_dim == 3:
            x = np.array(latents)[:,0]
            y = np.array(latents)[:,1]
            z = np.array(latents)[:,2]
        else:
            if latents.shape[0] > 2:
                pca = PCA()
                reduced_states = pca.fit_transform(latents)
                x = np.array(reduced_states)[:,0]
                y = np.array(reduced_states)[:,1]
                z = np.array(reduced_states)[:,2]
            else:
                x = np.array(latents)[:,0]
                y = np.array(latents)[:,1]
                z = np.array(latents)[:,2]

        # Save latents
        latents_data = {
            'latents': latents, 'reward_locs': reward_locs,
            'color_labels': color_labels, 'xs': x_locations, 'ys': y_locations}
        with open(f'{latents_dir}latents.p', 'wb') as f:
            pickle.dump(latents_data, f)

        # Plot 3D latents
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.set_xlabel(r'$X_1$')
        ax.set_ylabel(r'$X_2$')
        ax.set_zlabel(r'$X_3$')
        colors = ['orange',
            cm.get_cmap('Blues'), cm.get_cmap('Reds'), cm.get_cmap('Purples')]
        color_steps = np.linspace(0.25, 1., MyEnv.HEIGHT, endpoint=True)
        markers = ['s', '^', 'o']
        for i in range(n):
            if x_locations[i] == MyEnv.CENTRAL_STEM:
                marker = markers[reward_locs[i]]
            else:
                marker = markers[-1]
            if color_labels[i] == 0:
                color = colors[0]
            else:
                color_step = color_steps[y_locations[i]]
                color = colors[color_labels[i]](color_step)
            ax.scatter(
                x[i], y[i], z[i], color=color, marker=marker,
                edgecolors='k', alpha=0.75, s=50, depthshade=True
                )
        axes_lims=[ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]
        plt.show()
        plt.savefig(f'{fig_dir}latents.pdf')

        # Plot continuous measure of dimensionality
        if (intern_dim > 3) and (latents.shape[0] > 2):
            variance_curve = np.cumsum(pca.explained_variance_ratio_)
            auc = np.trapz(variance_curve, dx=1/variance_curve.size)
            self._dimensionality_tracking.append(auc)
            plt.figure()
            plt.plot(self._dimensionality_tracking)
            plt.ylabel('AUC of PCA Explained Variance Ratio')
            plt.xlabel('Epochs')
            plt.savefig(f'{fig_dir}dimensionality.pdf')

        # Plot pairwise z-score distances
        dist_matrix = np.ones((MyEnv.HEIGHT*4, MyEnv.HEIGHT*4))*np.nan
        stems = [MyEnv.LEFT_STEM, MyEnv.CENTRAL_STEM, MyEnv.CENTRAL_STEM, MyEnv.RIGHT_STEM]
        for i in range(dist_matrix.shape[0]):
            for j in range(i+1):
                stem_i = i // MyEnv.HEIGHT; yloc_i = i % MyEnv.HEIGHT
                stem_j = j // MyEnv.HEIGHT; yloc_j = j % MyEnv.HEIGHT
                idxs_i = np.logical_and(
                    x_locations == stems[stem_i], y_locations == yloc_i)
                idxs_j = np.logical_and(
                    x_locations == stems[stem_j], y_locations == yloc_j)
                if stems[stem_i] == MyEnv.CENTRAL_STEM:
                    rloc = MyEnv.LEFT_REWARD if stem_i == 1 else MyEnv.RIGHT_REWARD
                    idxs_i = np.logical_and(idxs_i, reward_locs==rloc)
                if stems[stem_j] == MyEnv.CENTRAL_STEM:
                    rloc = MyEnv.LEFT_REWARD if stem_j == 1 else MyEnv.RIGHT_REWARD
                    idxs_j = np.logical_and(idxs_j, reward_locs==rloc)
                if (np.sum(idxs_i) == 0) or (np.sum(idxs_j) == 0):
                    continue
                dist = []
                states_i = latents[idxs_i]
                states_j = latents[idxs_j]
                if (i==j) and states_i.shape[0] == 1:
                    dist_matrix[i,j] = 0
                    continue
                for state_i_idx, state_i in enumerate(states_i):
                    state_j_idx = state_i_idx if i==j else states_j.shape[0]
                    for state_j in states_j[:state_j_idx]:
                        dist.append(np.linalg.norm(state_i-state_j))
                dist_matrix[i, j] = dist_matrix[j, i] = np.nanmean(dist)
        dist_matrix = dist_matrix
        self._separability_matrix = dist_matrix
        plt.figure(); plt.imshow(dist_matrix); plt.colorbar()
        for boundary in [0, MyEnv.HEIGHT, MyEnv.HEIGHT*2, MyEnv.HEIGHT*3]:
            plt.axhline(boundary-0.5, linewidth=2, color='white')
            plt.axvline(boundary-0.5, linewidth=2, color='white')
        plt.xticks(
            np.linspace(0, dist_matrix.shape[0]-0.5, num=9, endpoint=True)[1::2],
            ['Left', 'Central-L', 'Central-R', 'Right'], rotation=30)
        plt.yticks(
            np.linspace(0, dist_matrix.shape[0]-0.5, num=9, endpoint=True)[1::2],
            ['Left', 'Central-L', 'Central-R', 'Right'])
        plt.title('Pairwise distances of column states')
        plt.savefig(f'{fig_dir}pairwise_dist.pdf')

        # Plot separability metric over epochs
        self._separability_tracking[0].append(
            dist_matrix[MyEnv.HEIGHT, MyEnv.HEIGHT*2])
        self._separability_tracking[1].append(
            dist_matrix[MyEnv.HEIGHT+MyEnv.HEIGHT//2, MyEnv.HEIGHT*2+MyEnv.HEIGHT//2])
        self._separability_tracking[2].append(
            dist_matrix[MyEnv.HEIGHT*2 - 1, MyEnv.HEIGHT*3 - 1])
        self._separability_slope.append(linregress(
            np.arange(MyEnv.HEIGHT), np.diagonal(
            dist_matrix[MyEnv.HEIGHT:MyEnv.HEIGHT*2, MyEnv.HEIGHT*2:MyEnv.HEIGHT*3]
            )).slope)
        fig, axs = plt.subplots(3, 1)
        axs[2].plot(self._separability_tracking[0])
        axs[2].set_title('Reset point')
        axs[2].set_xlabel('Epochs')
        axs[1].plot(self._separability_tracking[1])
        axs[1].set_title('Middle of central stem')
        axs[0].plot(self._separability_tracking[2])
        axs[0].set_title('Decision point')
        for ax in axs:
            ax.set_ylabel('L/R Dist')
            ylim_max = np.nanmax(self._separability_tracking)*1.1
            if not np.isnan(ylim_max): ax.set_ylim(0, ylim_max)
        plt.tight_layout()
        plt.savefig(f'{fig_dir}dist_summary.pdf')
        plt.figure()
        plt.plot(self._separability_slope)
        plt.xlabel('Epochs')
        plt.ylabel('Slope of Central Stem Splitting')
        plt.tight_layout()
        plt.savefig(f'{fig_dir}dist_slope.pdf')

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
        if self._high_dim_obs:
            return [(1, self._width*6, self._height*6)]
        else:
            return [(1, self._width+2, self._height+2)]

    def observationType(self, subject):
        return np.float32

    def nActions(self):
        return 4

    def observe(self):
        obs = self.get_observation(self.x, self.y, self._reward_location)
        if self._obs_noise > 0.:
            obs = obs + np.random.normal(0, self._obs_noise, size=obs.shape)
        return [obs]

    def get_observation(self, x, y, reward_location):
        obs = np.zeros((self._width, self._height))
        for _y in np.arange(self._height):
            for _x in np.arange(self._width):
                if not self.in_bounds(_x, _y):
                    obs[_x, _y] = -1
        left_reward = (0, MyEnv.HEIGHT-1)
        right_reward = (MyEnv.WIDTH-1, MyEnv.HEIGHT-1)
        reset_reward = (MyEnv.WIDTH//2, 0)

        if self._show_rewards:
            if reward_location == MyEnv.LEFT_REWARD:
                obs[left_reward[0], left_reward[1]] = 1
            elif reward_location == MyEnv.RIGHT_REWARD:
                obs[right_reward[0], right_reward[1]] = 1
            else:
                obs[reset_reward[0], reset_reward[1]] = 1
        obs[x, y] = 10
        if self._high_dim_obs:
            obs = self.get_higher_dim_obs((x,y), obs)
        pad_obs = np.ones((self._width+2, self._height+2))*2
        pad_obs[1:-1, 1:-1] = obs
        return pad_obs

    def get_higher_dim_obs(self, agent_loc, obs):
        """
        Obtain the high-dimensional observation
        (agent is a humanoid instead of a box).
        """

        obs = copy.deepcopy(obs)
        obs = np.repeat(np.repeat(obs, 6, axis=0),6, axis=1)
        x, y = agent_loc
        agent_obs=np.zeros((6,6))
        agent_obs[0,2] = 10
        agent_obs[1,0:5] = 8
        agent_obs[2,1:4] = 8
        agent_obs[3,1:4] = 8
        agent_obs[4,1] = 8
        agent_obs[4,3] = 8
        agent_obs[5,0:2] = 8
        agent_obs[5,3:5] = 8
        obs[x*6:(x+1)*6:, y*6:(y+1)*6] = agent_obs
        self._agent_loc_map[str([x*6, y*6+2])] = agent_loc
        return obs

    def inTerminalState(self):
        return False

