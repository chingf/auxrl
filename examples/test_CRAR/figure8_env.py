import os
import numpy as np
import torch
from deer.base_classes import Environment
from sklearn.decomposition import PCA
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
    RIGHT = 1
    LEFT = 0
    RESET = 2
    HEIGHT = 5 #4 #3 #6
    WIDTH = 7 #5 #7 # Must be odd!

    def __init__(self, give_rewards=False, intern_dim=2, **kwargs):
        self._give_rewards = give_rewards
        self._mode = -1
        self._mode_score = 0.0
        self._mode_episode_count = 0

        self._height = MyEnv.HEIGHT
        self._width = MyEnv.WIDTH # Must be odd!
        self._higher_dim_obs = kwargs['higher_dim_obs']
        self._show_rewards = kwargs.get('show_rewards', True)
        self._nstep = kwargs.get('nstep', 1)
        self._nstep_decay = kwargs.get('nstep_decay', 1.)
        self.x = 3
        self.y = 0
        self._reward_location = MyEnv.LEFT
        self._last_reward_location = MyEnv.RIGHT
        self._intern_dim = intern_dim
        self._space_label = self.make_space_labels()

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
        if not self._higher_dim_obs:
            space_labels = space_labels.reshape((1, -1))
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

        possible_resets = [
            (MyEnv.WIDTH//2, 0),
            (MyEnv.WIDTH//2, MyEnv.HEIGHT-1),
            (0, MyEnv.HEIGHT//2),
            (MyEnv.WIDTH-1, MyEnv.HEIGHT//2),
            ]
        self.x, self.y = (MyEnv.WIDTH//2, 0) #possible_resets[np.random.choice(4)]
        self._reward_location = MyEnv.LEFT
        self._last_rewarded = MyEnv.RIGHT
        
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
            if self._reward_location == MyEnv.RESET:
                if self._last_reward_location == MyEnv.LEFT:
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
        if ((self.x, self.y) == left_reward) and self._reward_location == MyEnv.LEFT:
            self.reward = 1
            self._reward_location = MyEnv.RESET
            self._last_reward_location = MyEnv.LEFT
        elif ((self.x, self.y) == right_reward) and self._reward_location == MyEnv.RIGHT:
            self.reward = 1
            self._reward_location = MyEnv.RESET
            self._last_reward_location = MyEnv.RIGHT
        elif ((self.x, self.y) == reset_reward) and self._reward_location == MyEnv.RESET:
            self.reward = 1
            if self._last_reward_location == MyEnv.RIGHT:
                self._reward_location = MyEnv.LEFT
            else:
                self._reward_location = MyEnv.RIGHT
        else:
            self.reward = 0

        # Set reward to 0 if it's not a signal to be used
        if not self._give_rewards:
            self.reward = 0
              
        self._mode_score += self.reward
        return self.reward

    def make_state_with_history(self, states_val):
        tau = self._nstep_decay
        new_states_val = []
        for batch in range(states_val.shape[0]):
            walls = np.argwhere(states_val[batch,-1] == -1) # hacky
            new_obs = []
            for t in range(self._nstep):
                discount = tau**(self._nstep-t)
                new_obs.append(discount * states_val[batch,t])
            new_obs = np.sum(new_obs, axis=0)
            new_obs[walls] = -1
            new_states_val.append(new_obs)
        new_states_val = np.array(new_states_val)
        return new_states_val

    def summarizePerformance(self, test_data_set, learning_algo, fname):
        """ Plot of the low-dimensional representation of the environment built by the model
        """

        if fname is None:
            fig_dir = './'
        else:
            fig_dir = f'figs/{fname}/'
            if not os.path.isdir(fig_dir):
                os.makedirs(fig_dir)

        # Only seen states
        observations = test_data_set.observations()[0]
        reward_locs = test_data_set.reward_locs()
        observations_tcm = []
        reward_locs_tcm = []
        color_labels = []
        y_locations = []
        for t in np.arange(self._nstep, observations.shape[0]):
            tcm_obs = observations[t-self._nstep:t].reshape((1, self._nstep, -1))
            tcm_obs = self.make_state_with_history(tcm_obs)
            observations_tcm.append(tcm_obs)
            reward_locs_tcm.append(reward_locs[t-1])
            agent_location = np.argwhere(observations[t-1]==10)[0,1]
            y_location = agent_location % MyEnv.HEIGHT
            y_locations.append(y_location)
            color_label = self._space_label[0, agent_location]
            color_labels.append(color_label)
        observations_tcm = np.array(observations_tcm, dtype='float')[-50:]
        reward_locs_tcm = np.array(reward_locs_tcm, dtype='int')[-50:]
        color_labels = np.array(color_labels, dtype=int)[-50:]
        y_locations = np.array(y_locations)[-50:]
        unique_observations_tcm, unique_idxs = np.unique(
            observations_tcm, axis=0, return_index=True)
        color_labels = color_labels[unique_idxs]
        marker_labels = reward_locs_tcm[unique_idxs].astype(int)
        xxplot = False
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        n = unique_observations_tcm.shape[0]
        with torch.no_grad():
            abs_states = learning_algo.crar.encoder(
                torch.tensor(unique_observations_tcm).float().to(device)
                )
    
        actions = test_data_set.actions()[0:n]
        if self.inTerminalState() == False:
            self._mode_episode_count += 1
        print("== Mean score per episode is {} over {} episodes ==".format(
            self._mode_score/(self._mode_episode_count+0.0001), self._mode_episode_count
            ))
                
        m = cm.ScalarMappable(cmap=cm.jet)
       
        abs_states_np = abs_states.cpu().numpy()
        if len(abs_states_np.shape) == 1:
            abs_states_np = abs_states_np.reshape((1, -1))

        if self._intern_dim == 2:
            x = np.array(abs_states_np)[:,0]
            y = np.array(abs_states_np)[:,1]
            z = np.zeros(y.shape)
        elif self._intern_dim == 3:
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
        ax = fig.add_subplot(111,projection='3d')
        ax.set_xlabel(r'$X_1$')
        ax.set_ylabel(r'$X_2$')
        ax.set_zlabel(r'$X_3$')
                    
        # Plot the estimated transitions
        for i in range(n-1):
            n_actions = 4
            action_colors = ["0.9", "0.65", "0.4", "0.15"]
            for action in range(n_actions):
                action_encoding = np.zeros(n_actions)
                action_encoding[action] = 1
                with torch.no_grad():
                    pred = learning_algo.crar.transition(torch.cat([
                        abs_states[i:i+1], torch.as_tensor([action_encoding])
                        ], dim=1).float().to(device)).cpu().numpy()
                if (self._intern_dim > 3) and (abs_states_np.shape[0] > 2):
                    pred = pca.transform(pred)
                x_transitions = np.concatenate([x[i:i+1],pred[0,:1]])
                y_transitions = np.concatenate([y[i:i+1],pred[0,1:2]])
                if self._intern_dim == 2:
                    z_transitions = np.zeros(y_transitions.shape)
                else:
                    z_transitions = np.concatenate([z[i:i+1],pred[0,2:3]])
                ax.plot(
                    x_transitions, y_transitions, z_transitions,
                    color=action_colors[action], alpha=0.75)
        
        # Plot the dots at each time step depending on the action taken
        colors = [
            'orange', cm.get_cmap('Blues'), cm.get_cmap('Reds'),
            cm.get_cmap('Purples')
            ]
        color_steps = np.linspace(0.25, 1., MyEnv.HEIGHT, endpoint=True)
        markers = ['s', '^', 'o']
        central_stem = np.array(color_labels==3)
        not_central_stem = np.logical_not(central_stem)
        for i in range(x.size):
            if central_stem[i]:
                marker = markers[marker_labels[i]]
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
        
        # Plot the legend for transition estimates
        box1b = TextArea(" Estimated transitions (action 0, 1, 2 and 3): ", textprops=dict(color="k"))
        box2b = DrawingArea(90, 20, 0, 0)
        el1b = Rectangle((5, 10), 15,2, fc="0.9", alpha=0.75)
        el2b = Rectangle((25, 10), 15,2, fc="0.65", alpha=0.75) 
        el3b = Rectangle((45, 10), 15,2, fc="0.4", alpha=0.75)
        el4b = Rectangle((65, 10), 15,2, fc="0.15", alpha=0.75) 
        box2b.add_artist(el1b)
        box2b.add_artist(el2b)
        box2b.add_artist(el3b)
        box2b.add_artist(el4b)
        
        boxb = HPacker(children=[box1b, box2b],
            align="center",
            pad=0, sep=5)
        
        anchored_box = AnchoredOffsetbox(loc=3,
            child=boxb, pad=0.,
            frameon=True,
            bbox_to_anchor=(0., 0.98),
            bbox_transform=ax.transAxes,
            borderpad=0.,
            )        
        ax.add_artist(anchored_box)
        plt.show()
        plt.savefig(f'{fig_dir}latents_{learning_algo.update_counter}.pdf')

        # Plot continuous measure of dimensionality
        if (self._intern_dim > 3) and (abs_states_np.shape[0] > 2):
            variance_curve = np.cumsum(pca.explained_variance_ratio_)
            auc = np.trapz(variance_curve, dx=1/variance_curve.size)
            plt.figure()
            plt.plot(variance_curve)
            plt.title(f'AUC: {auc}')
            plt.savefig(f'{fig_dir}latent_dim_{learning_algo.update_counter}.pdf')
        
        # Plot losses over epochs
        losses, loss_names = learning_algo.get_losses()
        loss_weights = learning_algo._loss_weights
        _, axs = plt.subplots(4, 2, figsize=(7, 10))
        for i in range(8):
            loss = losses[i]; loss_name = loss_names[i]
            ax = axs[i%4][i//4]
            ax.plot(loss)
            ax.set_ylabel(loss_name)
        plt.tight_layout()
        plt.savefig(f'{fig_dir}losses.pdf', dpi=300)
        _, axs = plt.subplots(4, 2, figsize=(7, 10))
        for i in range(8):
            loss = losses[i]; loss_name = loss_names[i]
            ax = axs[i%4][i//4]
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
        if (self._higher_dim_obs==True):
            return [(1, self._width, self._height)]
            return [(1, (self._width+2)*3, (self._height+2)*3)]
        else:
            return [(1, self._height*self._width)]

    def observationType(self, subject):
        return np.float32

    def nActions(self):
        return 4

    def observe(self):
        obs = self.get_observation(self.x, self.y, self._reward_location)
        return [obs]

    def get_observation(self, x, y, reward_location):
        obs = np.zeros((self._width, self._height))
        for _y in np.arange(self._height):
            for _x in np.arange(self._width):
                if not self.in_bounds(_x, _y):
                    obs[_x, _y] = -1

#        if not self._give_rewards:
#            pass
        left_reward = (0, MyEnv.HEIGHT-1)
        right_reward = (MyEnv.WIDTH-1, MyEnv.HEIGHT-1)
        reset_reward = (MyEnv.WIDTH//2, 0)

        if self._show_rewards:
            if reward_location == MyEnv.LEFT:
                obs[left_reward[0], left_reward[1]] = 1
            elif reward_location == MyEnv.RIGHT:
                obs[right_reward[0], right_reward[1]] = 1
            else:
                obs[reset_reward[0], reset_reward[1]] = 1

        #obs = obs + np.random.normal(0, 0.2, size=obs.shape)
        obs[x, y] = 10

        if not self._higher_dim_obs: obs = obs.flatten()

        return obs

    def inTerminalState(self):
        return False

