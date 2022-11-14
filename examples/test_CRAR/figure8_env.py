import numpy as np
import torch
from deer.base_classes import Environment

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
    HEIGHT = 3 #3 #6
    WIDTH = 5 #7 # Must be odd!

    def __init__(self, give_rewards=False, intern_dim=2, **kwargs):
        self._give_rewards = give_rewards
        self._mode = -1
        self._mode_score = 0.0
        self._mode_episode_count = 0

        self._height = MyEnv.HEIGHT
        self._width = MyEnv.WIDTH # Must be odd!
        self._higher_dim_obs = kwargs['higher_dim_obs']
        self._show_rewards = kwargs.get('show_rewards', True)
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
                elif x == midpoint and y == MyEnv.HEIGHT-1:
                    space_labels[x,y] = 0
                elif x < midpoint:
                    space_labels[x,y] = 1
                elif x > midpoint:
                    space_labels[x,y] = 2
                elif x == midpoint:
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
                    elif ((self.x, self.y) == reset_reward) and (action==1):
                        pass
                    else:
                        self.x = new_x; self.y = new_y
                else:
                    if ((self.x, self.y) == right_reward) and (action==0):
                        pass
                    elif ((self.x, self.y) == reset_reward) and (action==0):
                        pass
                    else:
                        self.x = new_x; self.y = new_y
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

    def summarizePerformance(self, test_data_set, learning_algo, *args, **kwargs):
        """ Plot of the low-dimensional representation of the environment built by the model
        """

#        with torch.no_grad():
#            all_possib_inp = [] # Will store all possible observations
#            color_labels = []
#            marker_labels = []
#
#            # Only seen states
#            observations = np.unique(test_data_set.observations()[0], axis=0)
#
#            # All possible states
#            for _x in range(self._width):
#                for _y in range(self._height):
#                    if not self.in_bounds(_x, _y): continue
#                    for idx, r_loc in enumerate([MyEnv.LEFT, MyEnv.RIGHT, MyEnv.RESET]):
#                        state = self.get_observation(_x, _y, r_loc)
#
#                        observed = False
#                        for obs in observations:
#                            if np.all(state == obs):
#                                observed = True
#
#                        if observed:
#                            all_possib_inp.append(state)
#                            color_labels.append(
#                                self._space_label[0, np.argwhere(state==10)[0,0]]
#                                )
#                            marker_labels.append(idx)
#
#            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#            all_possib_inp = np.expand_dims(
#                np.array(all_possib_inp,dtype='float'), axis=1
#                )
#            all_possib_abs_states = learning_algo.crar.encoder(
#                torch.tensor(all_possib_inp).float().to(device)
#                ).cpu().numpy()
#
#            n = observations.shape[0]
#            historics = observations
#            abs_states = learning_algo.crar.encoder(
#                torch.tensor(historics).float().to(device)
#                )
#        
#            actions = test_data_set.actions()[0:n]
#            if self.inTerminalState() == False:
#                self._mode_episode_count += 1
#            print("== Mean score per episode is {} over {} episodes ==".format(self._mode_score / (self._mode_episode_count+0.0001), self._mode_episode_count))
#                    
#            m = cm.ScalarMappable(cmap=cm.jet)
#           
#            abs_states_np = abs_states.cpu().numpy()
#            if len(abs_states_np.shape) == 1:
#                abs_states_np = abs_states_np.reshape((1, -1))
#            x = np.array(abs_states_np)[:,0]
#            y = np.array(abs_states_np)[:,1]
#            if(self._intern_dim>2):
#                z = np.array(abs_states_np)[:,2]
#                        
#            fig = plt.figure()
#            if(self._intern_dim==2):
#                ax = fig.add_subplot(111)
#                ax.set_xlabel(r'$X_1$')
#                ax.set_ylabel(r'$X_2$')
#            else:
#                ax = fig.add_subplot(111,projection='3d')
#                ax.set_xlabel(r'$X_1$')
#                ax.set_ylabel(r'$X_2$')
#                ax.set_zlabel(r'$X_3$')
#                        
#            # Plot the estimated transitions
#            for i in range(n-1):
#                predicted1 = learning_algo.crar.transition(torch.cat([
#                        abs_states[i:i+1],
#                        torch.as_tensor([[1,0,0,0]], device=device).float()
#                        ], dim=1)).cpu().numpy()
#                predicted2 = learning_algo.crar.transition(torch.cat([
#                        abs_states[i:i+1],
#                        torch.as_tensor([[0,1,0,0]], device=device).float()
#                        ], dim=1)).cpu().numpy()
#                predicted3 = learning_algo.crar.transition(torch.cat([
#                        abs_states[i:i+1],
#                        torch.as_tensor([[0,0,1,0]], device=device).float()
#                        ], dim=1)).cpu().numpy()
#                predicted4 = learning_algo.crar.transition(torch.cat([
#                        abs_states[i:i+1],
#                        torch.as_tensor([[0,0,0,1]], device=device).float()
#                        ], dim=1)).cpu().numpy()
#                if(self._intern_dim==2):
#                    ax.plot(np.concatenate([x[i:i+1],predicted1[0,:1]]), np.concatenate([y[i:i+1],predicted1[0,1:2]]), color="0.9", alpha=0.75)
#                    ax.plot(np.concatenate([x[i:i+1],predicted2[0,:1]]), np.concatenate([y[i:i+1],predicted2[0,1:2]]), color="0.65", alpha=0.75)
#                    ax.plot(np.concatenate([x[i:i+1],predicted3[0,:1]]), np.concatenate([y[i:i+1],predicted3[0,1:2]]), color="0.4", alpha=0.75)
#                    ax.plot(np.concatenate([x[i:i+1],predicted4[0,:1]]), np.concatenate([y[i:i+1],predicted4[0,1:2]]), color="0.15", alpha=0.75)
#                else:
#                    ax.plot(np.concatenate([x[i:i+1],predicted1[0,:1]]), np.concatenate([y[i:i+1],predicted1[0,1:2]]), np.concatenate([z[i:i+1],predicted1[0,2:3]]), color="0.9", alpha=0.75)
#                    ax.plot(np.concatenate([x[i:i+1],predicted2[0,:1]]), np.concatenate([y[i:i+1],predicted2[0,1:2]]), np.concatenate([z[i:i+1],predicted2[0,2:3]]), color="0.65", alpha=0.75)
#                    ax.plot(np.concatenate([x[i:i+1],predicted3[0,:1]]), np.concatenate([y[i:i+1],predicted3[0,1:2]]), np.concatenate([z[i:i+1],predicted3[0,2:3]]), color="0.4", alpha=0.75)
#                    ax.plot(np.concatenate([x[i:i+1],predicted4[0,:1]]), np.concatenate([y[i:i+1],predicted4[0,1:2]]), np.concatenate([z[i:i+1],predicted4[0,2:3]]), color="0.15", alpha=0.75)            
#            
#            # Plot the dots at each time step depending on the action taken
#            colors = ['orange','blue', 'red', 'purple']
#            markers = ['x', 'o', '*']
#            color_labels = np.array(color_labels)
#            marker_label = np.array(marker_labels)
#
#            if(self._intern_dim==2):
#                for m_idx in np.unique(marker_labels):
#                    _states = all_possib_abs_states[marker_labels==m_idx]
#                    _colors = color_labels[marker_labels==m_idx]
#                    line3 = ax.scatter(
#                        _states[:,0], _states[:,1],
#                        c=[colors[i] for i in _colors],
#                        marker=markers[m_idx],
#                        edgecolors='k', alpha=0.5, s=100
#                        )
#            else:
#                for m_idx in np.unique(marker_labels):
#                    _states = all_possib_abs_states[marker_labels==m_idx]
#                    _colors = color_labels[marker_labels==m_idx]
#                    line3 = ax.scatter(
#                        _states[:,0], _states[:,1], _states[:,2],
#                        c=[colors[i] for i in _colors],
#                        depthshade=True, edgecolors='k',
#                        alpha=0.5, s=50, marker=markers[m_idx],
#                        )
#    
#            if(self._intern_dim==2):
#                axes_lims=[ax.get_xlim(),ax.get_ylim()]
#            else:
#                axes_lims=[ax.get_xlim(),ax.get_ylim(),ax.get_zlim()]
#            
#            # Plot the legend for transition estimates
#            box1b = TextArea(" Estimated transitions (action 0, 1, 2 and 3): ", textprops=dict(color="k"))
#            box2b = DrawingArea(90, 20, 0, 0)
#            el1b = Rectangle((5, 10), 15,2, fc="0.9", alpha=0.75)
#            el2b = Rectangle((25, 10), 15,2, fc="0.65", alpha=0.75) 
#            el3b = Rectangle((45, 10), 15,2, fc="0.4", alpha=0.75)
#            el4b = Rectangle((65, 10), 15,2, fc="0.15", alpha=0.75) 
#            box2b.add_artist(el1b)
#            box2b.add_artist(el2b)
#            box2b.add_artist(el3b)
#            box2b.add_artist(el4b)
#            
#            boxb = HPacker(children=[box1b, box2b],
#                align="center",
#                pad=0, sep=5)
#            
#            anchored_box = AnchoredOffsetbox(loc=3,
#                child=boxb, pad=0.,
#                frameon=True,
#                bbox_to_anchor=(0., 0.98),
#                bbox_transform=ax.transAxes,
#                borderpad=0.,
#                )        
#            ax.add_artist(anchored_box)
#            plt.show()
#            plt.savefig('fig_base'+str(learning_algo.update_counter)+'.pdf')

        plt.figure()
        plt.plot(np.log(learning_algo.tracked_losses))
        plt.savefig('tracked_losses.pdf')

        plt.figure()
        plt.plot(np.log(learning_algo.tracked_disamb1))
        plt.savefig('tracked_disamb1.pdf')

        plt.figure()
        plt.plot(np.log(learning_algo.tracked_disamb2))
        plt.savefig('tracked_disamb2.pdf')

        plt.figure()
        plt.plot(np.log(learning_algo.tracked_disentang))
        plt.savefig('tracked_disentang.pdf')   

        plt.figure()
        plt.plot(np.log(learning_algo.tracked_T_err))
        plt.savefig('tracked_T_err.pdf')   
        matplotlib.pyplot.close("all") # avoids memory leaks

    def inputDimensions(self):
        if (self._higher_dim_obs==True):
            return [(1, self._width, self._height)]
            return [(1, (self._width+2)*3, (self._height+2)*3)]
        else:
            return [(1, self._height*self._width)]

#    def singleInputDimensions(self):
#        if (self._higher_dim_obs==True):
#            return [(self._width, self._height)]
#            return [((self._width+2)*3, (self._height+2)*3)]
#        else:
#            return [(1, self._height*self._width)]

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

