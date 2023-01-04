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

        self._mode = -1
        self._mode_score = 0.0
        self._mode_episode_count = 0
        self._size_maze = kwargs.get("size_maze", 16)
        self._higher_dim_obs = kwargs.get("higher_dim_obs", False)
        self._reward = kwargs.get("reward", False)
        self._plotfig = kwargs.get("plotfig", True)
        self._dimensionality_tracking = []
        self._reward_location = 0 # Just a placeholder
        self.create_map()

    def create_map(self, reset_goal=True):
        self._map=np.zeros((self._size_maze, self._size_maze))
        self._map[-1,:] = 1; self._map[0,:] = 1
        self._map[:,0] = 1; self._map[:,-1] = 1
        midpoint = self._size_maze//2
        self._map[:, midpoint] = 1
        self._map[midpoint-2:midpoint+2, midpoint] = 0
        valid_pos = np.argwhere(self._map != 1)
        self._pos_agent = valid_pos[np.random.choice(len(valid_pos))]
        if reset_goal:
            self._pos_goal = valid_pos[np.random.choice(len(valid_pos))]
                
    def reset(self, mode):
        self.create_map(reset_goal=False)
        if mode == MyEnv.VALIDATION_MODE:
            if self._mode != MyEnv.VALIDATION_MODE:
                self._mode = MyEnv.VALIDATION_MODE
                self._mode_score = 0.0
                self._mode_episode_count = 0
            else:
                self._mode_episode_count += 1
        elif self._mode != -1:
            self._mode = -1
        return [1 * [self._size_maze * [self._size_maze * [0]]]]
        
        
    def act(self, action):
        """Applies the agent action [action] on the environment.

        Parameters
        -----------
        action : int
            The action selected by the agent to operate on the environment. Should be an identifier 
            included between 0 included and nActions() excluded.
        """

        self._cur_action=action
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
        
        if (self._reward) and np.array_equal(self._pos_agent, self._pos_goal):
            reward = 1
        else:
            reward = 0
        self._mode_score += reward
        return reward

    def summarizePerformance(self, test_data_set, learning_algo, fname):
        """
        Plot of the low-dimensional representation of the environment
        built by the model
        """

        if fname is None:
            fig_dir = './'
        else:
            fig_dir = f'figs/{fname}/'
            if not os.path.isdir(fig_dir):
                os.makedirs(fig_dir)

        all_possib_inp = [] 
        labels_maze=[]
        self.create_map(reset_goal=False)
        intern_dim = learning_algo._internal_dim
        for y_a in range(self._size_maze):
            for x_a in range(self._size_maze):                
                state=copy.deepcopy(self._map)
                state[self._size_maze//2,self._size_maze//2]=0
                if state[x_a,y_a] == 0:
                    if self._higher_dim_obs:
                        all_possib_inp.append(self.get_higher_dim_obs([x_a, y_a]))
                    else:
                        state[x_a,y_a] = 0.5
                        all_possib_inp.append(state)
                    
        device = learning_algo.device
        with torch.no_grad():
            all_possib_abs_states=learning_algo.crar.encoder(
                torch.tensor(all_possib_inp).float().to(device)
                ).cpu().numpy()
        if(all_possib_abs_states.ndim==4):
            all_possib_abs_states=np.transpose(all_possib_abs_states, (0, 3, 1, 2))    # data_format='channels_last' --> 'channels_first'
        
        n = 1000
        historics = test_data_set.observations()[0][0:n]
        historics = np.squeeze(historics, axis=1)
        with torch.no_grad():
            abs_states = learning_algo.crar.encoder(
                torch.tensor(historics).float().to(device))
        if abs_states.ndim == 4:
            abs_states = np.transpose(abs_states, (0, 3, 1, 2)) # data_format='channels_last' --> 'channels_first'
    
        actions=test_data_set.actions()[0:n]
        
        if not self.inTerminalState():
            self._mode_episode_count += 1
        print("== Mean score per episode is {} over {} episodes ==".format(self._mode_score / (self._mode_episode_count+0.0001), self._mode_episode_count))
                
        m = cm.ScalarMappable(cmap=cm.jet)
       
        abs_states_np = abs_states.cpu().numpy()
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
        if intern_dim > 2:
            z = np.array(abs_states)[:,2]
                    
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
        for i in range(n-1):
            with torch.no_grad():
                predicted1 = learning_algo.crar.transition(torch.cat([
                    abs_states[i:i+1],
                    torch.as_tensor([[1,0,0,0]], device=device).float()
                    ], dim=1)).cpu().numpy()
                predicted2 = learning_algo.crar.transition(torch.cat([
                    abs_states[i:i+1],
                    torch.as_tensor([[0,1,0,0]], device=device).float()
                    ], dim=1)).cpu().numpy()
                predicted3 = learning_algo.crar.transition(torch.cat([
                    abs_states[i:i+1],
                    torch.as_tensor([[0,0,1,0]], device=device).float()
                    ], dim=1)).cpu().numpy()
                predicted4 = learning_algo.crar.transition(torch.cat([
                    abs_states[i:i+1],
                    torch.as_tensor([[0,0,0,1]], device=device).float()
                    ], dim=1)).cpu().numpy()
            if intern_dim == 2:
                ax.plot(np.concatenate([x[i:i+1],predicted1[0,:1]]), np.concatenate([y[i:i+1],predicted1[0,1:2]]), color="0.9", alpha=0.75)
                ax.plot(np.concatenate([x[i:i+1],predicted2[0,:1]]), np.concatenate([y[i:i+1],predicted2[0,1:2]]), color="0.65", alpha=0.75)
                ax.plot(np.concatenate([x[i:i+1],predicted3[0,:1]]), np.concatenate([y[i:i+1],predicted3[0,1:2]]), color="0.4", alpha=0.75)
                ax.plot(np.concatenate([x[i:i+1],predicted4[0,:1]]), np.concatenate([y[i:i+1],predicted4[0,1:2]]), color="0.15", alpha=0.75)
            else:
                ax.plot(np.concatenate([x[i:i+1],predicted1[0,:1]]), np.concatenate([y[i:i+1],predicted1[0,1:2]]), np.concatenate([z[i:i+1],predicted1[0,2:3]]), color="0.9", alpha=0.75)
                ax.plot(np.concatenate([x[i:i+1],predicted2[0,:1]]), np.concatenate([y[i:i+1],predicted2[0,1:2]]), np.concatenate([z[i:i+1],predicted2[0,2:3]]), color="0.65", alpha=0.75)
                ax.plot(np.concatenate([x[i:i+1],predicted3[0,:1]]), np.concatenate([y[i:i+1],predicted3[0,1:2]]), np.concatenate([z[i:i+1],predicted3[0,2:3]]), color="0.4", alpha=0.75)
                ax.plot(np.concatenate([x[i:i+1],predicted4[0,:1]]), np.concatenate([y[i:i+1],predicted4[0,1:2]]), np.concatenate([z[i:i+1],predicted4[0,2:3]]), color="0.15", alpha=0.75)            
        
        # Plot the dots at each time step depending on the action taken
        length_block=[[0,18],[18,19],[19,31]]
        for i in range(3):
            colors=['blue','orange','green']
            if intern_dim == 2:
                line3 = ax.scatter(all_possib_abs_states[length_block[i][0]:length_block[i][1],0], all_possib_abs_states[length_block[i][0]:length_block[i][1],1], c=colors[i], marker='x', edgecolors='k', alpha=0.5, s=100)
            else:
                line3 = ax.scatter(all_possib_abs_states[length_block[i][0]:length_block[i][1],0], all_possib_abs_states[length_block[i][0]:length_block[i][1],1] ,all_possib_abs_states[length_block[i][0]:length_block[i][1],2], marker='x', depthshade=True, edgecolors='k', alpha=0.5, s=50)

        if intern_dim == 2:
            axes_lims=[ax.get_xlim(),ax.get_ylim()]
        else:
            axes_lims=[ax.get_xlim(),ax.get_ylim(),ax.get_zlim()]
        
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
        
        boxb = HPacker(children=[box1b, box2b], align="center", pad=0, sep=5)
        
        anchored_box = AnchoredOffsetbox(
            loc=3, child=boxb, pad=0., frameon=True,
            bbox_to_anchor=(0., 0.98), bbox_transform=ax.transAxes,
            borderpad=0.)        
        ax.add_artist(anchored_box)

        if self._plotfig:
            plt.savefig(f'{fig_dir}latents_{learning_algo.update_counter}.pdf')
        else:
            plt.savefig(f'{fig_dir}latents.pdf')

        # Plot continuous measure of dimensionality
        if (intern_dim > 3) and (abs_states_np.shape[0] > 2):
            variance_curve = np.cumsum(pca.explained_variance_ratio_)
            auc = np.trapz(variance_curve, dx=1/variance_curve.size)
            self._dimensionality_tracking.append(auc)
            plt.figure()
            plt.plot(self._dimensionality_tracking)
            plt.ylabel('AUC of PCA Explained Variance Ratio')
            plt.xlabel('Epochs')
            plt.savefig(f'{fig_dir}dimensionality.pdf')

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
        if self._plotfig:
            plt.savefig(f'{fig_dir}losses.pdf', dpi=300)
        _, axs = plt.subplots(4, 2, figsize=(7, 10))
        for i in range(8):
            loss = losses[i]; loss_name = loss_names[i]
            ax = axs[i%4][i//4]
            ax.plot(np.array(loss)*loss_weights[i])
            ax.set_ylabel(loss_name)
        plt.tight_layout()
        if self._plotfig:
            plt.savefig(f'{fig_dir}scaled_losses.pdf', dpi=300)
        plt.figure()
        plt.plot(losses[-1])
        plt.title('Total Loss')
        if self._plotfig:
            plt.savefig(f'{fig_dir}total_losses.pdf', dpi=300)
        matplotlib.pyplot.close("all") # avoid memory leaks

    def inputDimensions(self):
        if self._higher_dim_obs:
            return [(1, self._size_maze*6, self._size_maze*6)]
        else:
            return [(1,self._size_maze,self._size_maze)]

    def observationType(self, subject):
        return np.float

    def nActions(self):
        return 4

    def observe(self):
        obs = copy.deepcopy(self._map)
        #obs[self._pos_goal[0],self._pos_goal[1]] = 8
        obs[self._pos_agent[0],self._pos_agent[1]] = 0.5
        if self._higher_dim_obs:
            obs = self.get_higher_dim_obs(self._pos_agent)
        return [obs]
    
    def get_higher_dim_obs(self, pos_agent):
        """
        Go from box-visualization to a humanoid agent representation
        """

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
        if self._reward:
            return np.array_equal(self._pos_agent, self._pos_goal)
        else:
            return False

if __name__ == "__main__":
    pass
