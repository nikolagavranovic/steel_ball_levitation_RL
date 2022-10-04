import numpy as np
import math
import random as rand
from animation import BallAnimation
from model import BallLevitationModel
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import copy


class Agent():
    def __init__(self):
        # setting up model constants
        self.T = 0.02
        self.alpha = 0.95
        self.gamma = 0.9  # discount factor
        self.setpoint = 0.5    # setpoint can be modified
        self.anim = BallAnimation(speed = 50, y_min_real=0.0, y_max_real=1.0, setpoint=self.setpoint)
        self.best_score = 1   # best score according to average miss 
        self.best_scores = []
        
       
               
# region Parameter initialization

    def init_agent_params(self):
        """Initializes agent parameters, q table, discretising state variables: position, velocity, current
        """
        
        self.disc_pos = np.linspace(0.05, 1, 20)
        self.disc_vel = np.linspace(-2, 2, 11) 
        self.make_disc_actions()
        self.make_disc_curr()
        # initializing q matrix
        self.initialize_q_matrix()
        self.explored_fields = np.zeros(shape = (len(self.disc_pos), len(self.disc_vel), len(self.disc_curr), len(self.disc_action)))
    
    def init_model_params(self, initial_conditions):
        self.model = BallLevitationModel(9.81, 0.01, 0.1, 0.001, 1, 10, initial_conditions, self.T)

    def save_agent_params(self):
        """Saves parameters that defines agent to 6 seperate files.
        """
        np.save("q_matrix", self.q_table)
        np.save("discrete_position", self.disc_pos)
        np.save("discrete_velocity", self.disc_vel)
        np.save("discrete_current", self.disc_curr)
        np.save("discrete_actions", self.disc_action)
        np.save("explored_fields", self.explored_fields)
        
    def load_agent_params(self):
        """Load agent parameters previously saved to files.
        """
        self.q_table = np.load("q_matrix.npy")        
        self.disc_pos = np.load("discrete_position.npy")
        self.disc_vel = np.load("discrete_velocity.npy")
        self.disc_curr = np.load("discrete_current.npy")
        self.disc_action = np.load("discrete_actions.npy")
        self.explored_fields = np.load("explored_fields.npy")

    def make_disc_curr(self):
        """Make discrete current states
        """
        max_curr = self.model.find_maximum_current(self.model.find_minimum_action())
        self.disc_curr = np.linspace(0, max_curr, 21)

    def make_disc_actions(self):
        """ Descrete actions (voltage intensity): [0.0, ..., Vmax]
        will pull ball towards coil or let the ball fall to the other end
        how many descrete states are there is training parameter, and can be optimized
        """
        
        Vmax = self.model.find_minimum_action()
        self.disc_action = np.linspace(0, Vmax, 5)  

    def initialize_q_matrix(self):
        """ Q matrix is initialized with rewards of each state regardles of action for first iteration.
        This is better than initializing all with zeros, although it should work both ways.
        """        
        self.q_table = np.zeros(shape = (len(self.disc_pos), len(self.disc_vel), len(self.disc_curr), len(self.disc_action)))
        for i in range(len(self.disc_pos)):
            for j in range(len(self.disc_vel)):
                for k in range(len(self.disc_curr)):
                    for n in range(len(self.disc_action)):
                        self.q_table[i, j, k, n] = self.cost_function(self.setpoint, self.disc_pos[i], self.disc_vel[j]) 

# endregion
    


# region Learning

    def get_action(self, epsilon, optimal_action):
        """ Chooses an action according to epsilon greedy policy.

        Args:
            epsilon (float): Probability that random action is choosen
            optimal_action (float): Optimal action

        Returns:
            action, action_ind: Action agent is taking and it's index.
        """
        c = np.random.choice([0, 1], p = [epsilon, 1 - epsilon])

        if c == 0:
            action_ind = np.random.choice(range(len(self.disc_action))) # choose random action
        else:
            action_ind = optimal_action

        action = self.disc_action[action_ind] 
        return action, action_ind

    def play_episode_bellman(self, epsilon = 0.05, niter = 2000, animation = False, bonus_flag = False, show_info_flag = False):
        """ Simulates one continous episode of training with bellman equation.

        Args:
            epsilon (float): Probability of choosing random action, due to exploration. Defaults to 0.05.
            niter (int): Number of iterations (with sample time T) in episode. Defaults to 2000.
            plot_pendulum (bool): Show animation. Defaults to False.
            bonus_flag (bool): Award bonus if setpoint state with 0 velocity. Defaults to False.
            show_info_flag (bool): Show info after 500th iteration. Defaults to False.
        """
        self.model.reset_initial_conditions()
        
        for i in range(niter):
            
            x, dx, curr = self.model.get_states()
            # find the discrete state sistem is closest to
            x_disc_ind = np.argmin(abs(self.disc_pos - x))   # smallest difference of current state to the closest discrete state
            dx_disc_ind = np.argmin(abs(self.disc_vel - dx))
            curr_disc_ind = np.argmin(abs(self.disc_curr - curr))

            reward = self.cost_function(self.setpoint, self.disc_pos[x_disc_ind], self.disc_vel[dx_disc_ind])
            
            # epsilon is probability that random action is choosen
            c = np.random.choice([0, 1], p = [epsilon, 1 - epsilon])

            if c == 0:
                action_ind = np.random.choice(range(len(self.disc_action))) # choose random action
            else:
                # finding index of minimal action for given states value (it can be changed so that the best action is for maximum
                # if cost function is defined differently)
                action_ind = int(np.argmin(self.q_table[x_disc_ind, dx_disc_ind, curr_disc_ind, :])) 

            action = self.disc_action[action_ind] 

            self.model.set_action(action)

            xnew, dxnew, currnew = self.model.next_states()
            xnew_disc_ind = np.argmin(abs(self.disc_pos - xnew))   # smallest difference of current state to the closest discrete state
            dxnew_disc_ind = np.argmin(abs(self.disc_vel - dxnew))
            currnew_disc_ind = np.argmin(abs(self.disc_curr - currnew))

            bonus = 0
            if bonus_flag:
                # reward for the state in which velocity is 0 and angle is pi
                # computed state variables has to be in some (discrete) range - there is some tolerance  
                if  self.disc_vel[dxnew_disc_ind] == 0:
                    if abs(self.disc_pos[xnew_disc_ind] - self.setpoint) < 0.01:  # due to numerical error, condition is defined this way
                        bonus = -1 # setting big negative reward for state and action that led to this state
                        # so that all other states would be motivated to get here

            # updating values for action in state is done according to Belmann's equation. 
            self.q_table[x_disc_ind, dx_disc_ind, curr_disc_ind, action_ind] = reward + self.gamma * min(self.q_table[xnew_disc_ind, dxnew_disc_ind, currnew_disc_ind, :]) + bonus
            self.explored_fields[x_disc_ind, dx_disc_ind, curr_disc_ind, action_ind] = 1

            # if ball is in desired condition, break the loop and start episode again
            # if reward < 10e-10:
            #     break

            if animation:
                self.anim.plot_ball_position(xnew)

            if show_info_flag:
                if (i + 1)% 500 == 0:
                    print(f"Iteration {i}, epsilon {epsilon}")
                    
                    min_value = min(self.q_table[self.q_table != np.NAN])                    
                    
                    x, dx, curr, a = np.where(self.q_table == min_value)
                    print(f"Minimum q value: {min_value} is for\n position {self.disc_pos[x]}, velocity {self.disc_vel[dx]} and action {self.disc_action[a]}")
                    print("-----------------------------------")
                    print(f"number of unexplored fields is {len(self.explored_fields[self.explored_fields == 0])}") 

    def play_episode_qlearning(self, epsilon = 0.05, niter = 2000, animation = False, bonus_flag = False, show_info_flag = False, goal_break=False):
        """ Simulates one continous episode of training with Q-learning.

        Args:
            epsilon (float): Probability of choosing random action, due to exploration. Defaults to 0.05.
            niter (int): Number of iterations (with sample time T) in episode. Defaults to 2000.
            plot_pendulum (bool): Show animation. Defaults to False.
            bonus_flag (bool): Award bonus if setpoint state with 0 velocity. Defaults to False.
            show_info_flag (bool): Show info after 500th iteration. Defaults to False.
        """
        self.model.reset_initial_conditions()
        
        for i in range(niter):
            
            x, dx, curr = self.model.get_states()
            # find the discrete state sistem is closest to
            x_disc_ind = np.argmin(abs(self.disc_pos - x))   # smallest difference of current state to the closest discrete state
            dx_disc_ind = np.argmin(abs(self.disc_vel - dx))
            curr_disc_ind = np.argmin(abs(self.disc_curr - curr))

            reward = self.cost_function(self.setpoint, self.disc_pos[x_disc_ind], self.disc_vel[dx_disc_ind])
            
            # epsilon is probability that random action is choosen
            c = np.random.choice([0, 1], p = [epsilon, 1 - epsilon])

            if c == 0:
                action_ind = np.random.choice(range(len(self.disc_action))) # choose random action
            else:
                # finding index of minimal action for given states value (it can be changed so that the best action is for maximum
                # if cost function is defined differently)
                action_ind = int(np.argmin(self.q_table[x_disc_ind, dx_disc_ind, curr_disc_ind, :])) 

            action = self.disc_action[action_ind] 

            self.model.set_action(action)

            xnew, dxnew, currnew = self.model.next_states()
            xnew_disc_ind = np.argmin(abs(self.disc_pos - xnew))   # smallest difference of current state to the closest discrete state
            dxnew_disc_ind = np.argmin(abs(self.disc_vel - dxnew))
            currnew_disc_ind = np.argmin(abs(self.disc_curr - currnew))

            bonus = 0
            if bonus_flag:
                # reward for the state in which velocity is 0 and angle is pi
                # computed state variables has to be in some (discrete) range - there is some tolerance  
                if  self.disc_vel[dxnew_disc_ind] == 0:
                    if abs(self.disc_pos[xnew_disc_ind] - self.setpoint) < 0.01:  # due to numerical error, condition is defined this way
                        bonus = -1 # setting big negative reward for state and action that led to this state
                        # so that all other states would be motivated to get here

            # this equation is just little modification of bellman's equation, because of parameter alpha
            # which has a function that previous values has some impact on current value, although it is very small
            self.q_table[x_disc_ind, dx_disc_ind, curr_disc_ind, action_ind] += self.alpha*(
                reward + self.gamma * min(self.q_table[xnew_disc_ind, dxnew_disc_ind, currnew_disc_ind, :]) - 
                self.q_table[x_disc_ind, dx_disc_ind, curr_disc_ind, action_ind] + bonus)

            self.explored_fields[x_disc_ind, dx_disc_ind, curr_disc_ind, action_ind] = 1

            # if ball is in desired condition, break the loop and start episode again
            if goal_break:
                if reward < 10e-10:
                    break

            if animation:
                self.anim.plot_ball_position(xnew)

            if show_info_flag:
                if (i + 1)% 500 == 0:
                    print(f"Iteration {i}, epsilon {epsilon}")
                    
                    min_value = min(self.q_table[self.q_table != np.NAN])                    
                    
                    x, dx, curr, a = np.where(self.q_table == min_value)
                    print(f"Minimum q value: {min_value} is for\n position {self.disc_pos[x]}, velocity {self.disc_vel[dx]} current {self.disc_curr[curr]} and action {self.disc_action[a]}")
                    print("-----------------------------------")
                    print(f"number of unexplored fields is {len(self.explored_fields[self.explored_fields == 0])}") 
 
    def play_episode_sarsa(self, epsilon = 0.05, niter = 2000, animation = False, bonus_flag = False, show_info_flag = False, goal_break=False):
        """ Simulates one continous episode of training with SARSA learning.

        Args:
            epsilon (float): Probability of choosing random action, due to exploration. Defaults to 0.05.
            niter (int): Number of iterations (with sample time T) in episode. Defaults to 2000.
            plot_pendulum (bool): Show animation. Defaults to False.
            bonus_flag (bool): Award bonus if setpoint state with 0 velocity. Defaults to False.
            show_info_flag (bool): Show info after 500th iteration. Defaults to False.
        """
        self.model.reset_initial_conditions()
        
        x, dx, curr = self.model.get_states()        
        # find the discrete state sistem is closest to
        x_disc_ind = np.argmin(abs(self.disc_pos - x))   # smallest difference of current state to the closest discrete state
        dx_disc_ind = np.argmin(abs(self.disc_vel - dx))
        curr_disc_ind = np.argmin(abs(self.disc_curr - curr))

        
        # finding index of minimal action for given states value (it can be changed so that the best action is for maximum
        # if cost function is defined differently)
        optimal_action = int(np.argmin(self.q_table[x_disc_ind, dx_disc_ind, curr_disc_ind, :])) 

        action, action_ind = self.get_action(epsilon, optimal_action)

        for i in range(niter):            

            reward = self.cost_function(self.setpoint, self.disc_pos[x_disc_ind], self.disc_vel[dx_disc_ind])
            
            self.model.set_action(action)

            # geting next state
            xnew, dxnew, currnew = self.model.next_states()
            xnew_disc_ind = np.argmin(abs(self.disc_pos - xnew))   # smallest difference of current state to the closest discrete state
            dxnew_disc_ind = np.argmin(abs(self.disc_vel - dxnew))
            currnew_disc_ind = np.argmin(abs(self.disc_curr - currnew))

            
            optimal_action = int(np.argmin(self.q_table[x_disc_ind, dx_disc_ind, curr_disc_ind, :])) 
            actionnew, actionnew_ind = self.get_action(epsilon, optimal_action)
            bonus = 0
            if bonus_flag:
                # reward for the state in which velocity is 0 and angle is pi
                # computed state variables has to be in some (discrete) range - there is some tolerance  
                if  self.disc_vel[dxnew_disc_ind] == 0:
                    if abs(self.disc_pos[xnew_disc_ind] - self.setpoint) < 0.01:  # due to numerical error, condition is defined this way
                        bonus = -1 # setting big negative reward for state and action that led to this state
                        # so that all other states would be motivated to get here

            # this equation is equation for sarsa on-policy
            self.q_table[x_disc_ind, dx_disc_ind, curr_disc_ind, action_ind] += self.alpha*(
                reward + self.gamma * self.q_table[xnew_disc_ind, dxnew_disc_ind, currnew_disc_ind, actionnew_ind] - 
                self.q_table[x_disc_ind, dx_disc_ind, curr_disc_ind, action_ind] + bonus)

            self.explored_fields[x_disc_ind, dx_disc_ind, curr_disc_ind, action_ind] = 1

            # if ball is in desired condition, break the loop and start episode again
            if goal_break:
                if reward < 10e-10:
                    break

            if animation:
                self.anim.plot_ball_position(xnew)

            if show_info_flag:
                if (i + 1)% 500 == 0:
                    print(f"Iteration {i}, epsilon {epsilon}")
                    
                    min_value = min(self.q_table[self.q_table != np.NAN])                    
                    
                    x, dx, curr, a = np.where(self.q_table == min_value)
                    print(f"Minimum q value: {min_value} is for\n position {self.disc_pos[x]}, velocity {self.disc_vel[dx]} and action {self.disc_action[a]}")
                    print("-----------------------------------")
                    print(f"number of unexplored fields is {len(self.explored_fields[self.explored_fields == 0])}") 


            # updating state and action
            x = copy.deepcopy(xnew)
            dx = copy.deepcopy(dxnew) 
            curr = copy.deepcopy(currnew)     
            # find the discrete state sistem is closest to
            x_disc_ind = np.argmin(abs(self.disc_pos - x))   # smallest difference of current state to the closest discrete state
            dx_disc_ind = np.argmin(abs(self.disc_vel - dx))
            curr_disc_ind = np.argmin(abs(self.disc_curr - curr))

            action = copy.deepcopy(actionnew)
            action_ind = copy.deepcopy(actionnew_ind)



    def learn(self, type="",  max_episodes = 10000, epsilon_start = 0.5, tolerance_error=1, iteration_to_validate = 100):
        """ Main method for agent training.

        Args:
            type (str): Type of learning, must be "bellman", "qlearning", "sarsa". Defaults to "".
            max_episodes (int): Maximum episodes agent is allowed to reach. Defaults to 10000.
            epsilon_start (float): Epsilon value in first iteretion. This parameter lineary degrades. Defaults to 0.5.
            tolerance_error (int): Average error after agent stops learning session. Defaults to 1.
            iteration_to_validate (int): Iteration after validation of agent behaviour will be done. Defaults to 100.
        """
        if type not in ["bellman", "qlearning", "sarsa"]:
            print("type argument error")
            exit()

        epsilon_stop = 0.0
        epsilon_step = (epsilon_start - epsilon_stop)/(max_episodes)
        epsilon = epsilon_start
        show_info = False
        # for i in range(n_of_epizodes):
        i = 0
        while self.best_score > tolerance_error and i < max_episodes:
            if type == "bellman":
                self.play_episode_bellman(niter = 500, epsilon=epsilon, show_info_flag=show_info, animation=False, bonus_flag=True) 
            elif type == "qlearning":
                self.play_episode_qlearning(niter = 500, epsilon=epsilon, show_info_flag=show_info, animation=False, bonus_flag=True) 
            elif type == "sarsa":
                self.play_episode_sarsa(niter = 500, epsilon=epsilon, show_info_flag=show_info, animation=False, bonus_flag=True) 
            
                
            show_info = False            
            epsilon -= epsilon_step

            if (i+1) % 1000 == 0:
                show_info = True
                print(f"Episode {i + 1} over...")
                print(f"Average value of q matrix is: {np.average(self.q_table)}")
                
            
            if (i+1) % 100 == 0:
                Xp, err, avg_err = self.validate()
                if self.best_score > avg_err:
                    self.best_score = copy.deepcopy(avg_err)
                    self.save_agent_params()

            self.best_scores.append(self.best_score)

            i += 1


# endregion   

    def control(self, simulation_length = 1000, export_to_gif = False):
        """ Performs ball control according to optimal policy.

        Args:
            simulation_length (int): Simulation length in time periods. Defaults to 1000.
            export_to_gif (bool): Flag, if true animation will be exported to GIF. Defaults to False.

        Returns:
            x (List(float)): List of position during the simulation.
        """
        self.model.reset_initial_conditions()
        retX = []
        for i in range(simulation_length):
            x, dx, curr = self.model.get_states()
            retX.append(x)
            x_disc_ind = np.argmin(abs(self.disc_pos - x))   # smallest difference of current state to the closest discrete state
            dx_disc_ind = np.argmin(abs(self.disc_vel - dx))
            inew_disc_ind = np.argmin(abs(self.disc_curr - curr))

            action_ind = int(np.argmin(self.q_table[x_disc_ind, dx_disc_ind, inew_disc_ind, :])) # nalazimo indeks minimalne vrednosti stanja (one koja je najmanje kaznjiva, ovo se moze promeniti kasnije da bude nagrada umesto kazne)
            action = self.disc_action[action_ind] 

            self.model.set_action(action)

            xnew, dxnew, inew = self.model.next_states()
            self.anim.plot_ball_position(xnew)
            self.anim.plot_current(curr, self.disc_curr[-1])
            self.anim.plot_voltage(action, self.disc_action[-1])
            self.anim.update(export_to_gif = export_to_gif)
            
        if export_to_gif:
            self.anim.export_to_gif()

        return retX

# region Plotting
    def plot_q_table(self):
        q_to_plot = pd.DataFrame(np.min(self.q_table, 2))  ## minimum po dimenziji koja se odnosi na akcija, odnosno nalazimo vrednosti svakog stanja
        
        s = sns.heatmap(q_to_plot, annot=True, fmt = ".0f", cmap='Blues', xticklabels=np.round(self.disc_vel, 3), yticklabels=np.round(self.disc_angle, 4))
        s.set(xlabel='angular speed', ylabel='angular position')
        plt.show()
    
    def plot_optimal_policy(self):
        q_to_plot = pd.DataFrame(np.argmin(np.argmin(self.q_table, axis = 3), 2))  ## minimum po dimenziji koja se odnosi na akcija, odnosno nalazimo vrednosti svakog stanja
        
        s = sns.heatmap(q_to_plot, annot=True, fmt = ".2f", cmap='Blues', xticklabels=np.round(self.disc_vel, 3), yticklabels=np.round(self.disc_pos, 4))
        s.set(xlabel='angular speed', ylabel='angular position')
        plt.show()

    def plot_table(self, table):       
        s = sns.heatmap(pd.DataFrame(table) , annot=True, fmt = ".2f", cmap='Blues', xticklabels=np.round(self.disc_vel, 2), yticklabels=np.round(self.disc_pos, 2))
        s.set(xlabel='speed', ylabel='position')
        plt.show()   

    def plot_pos_vel_table(self):    
        q_to_plot = pd.DataFrame(np.min(np.min(self.q_table, 3), 2))  ## minimum po dimenziji koja se odnosi na akcija, odnosno nalazimo vrednosti svakog stanja
           
        s = sns.heatmap(q_to_plot , annot=True, fmt = ".2f", cmap='Blues', xticklabels=np.round(self.disc_vel, 2), yticklabels=np.round(self.disc_pos, 2))
        s.set(xlabel='speed', ylabel='position')
        plt.show() 

    def plot_pos_curr_table(self):    
        q_to_plot = pd.DataFrame(np.min(np.min(self.q_table, 3), 1))  ## minimum po dimenziji koja se odnosi na akcija, odnosno nalazimo vrednosti svakog stanja
           
        s = sns.heatmap(q_to_plot , annot=True, fmt = ".2f", cmap='Blues', xticklabels=np.round(self.disc_curr, 3), yticklabels=np.round(self.disc_pos, 2))
        s.set(xlabel='current', ylabel='position')
        plt.show() 
            
    def plot_vel_curr_table(self):    
        q_to_plot = pd.DataFrame(np.min(np.min(self.q_table, 3), 0))  ## minimum po dimenziji koja se odnosi na akcija, odnosno nalazimo vrednosti svakog stanja
           
        s = sns.heatmap(q_to_plot , annot=True, fmt = ".2f", cmap='Blues', xticklabels=np.round(self.disc_curr, 3), yticklabels=np.round(self.disc_vel, 2))
        s.set(xlabel='current', ylabel='velocity')
        plt.show() 
# endregion  


    def cost_function(self, setpoint, x, dx):     
        """ Penality is less if ball is closer to setpoint and if velocity is 0
        although is more important that position is nearer.

        Args:
            setpoint (float): Setpoint value.
            x (float): Position value.
            dx (float): Velocity value.

        Returns:
            cost (float): Value that indicates how good is state.
        """
        
        # NOTE: there is no current in cost function. Maybe this should be modified!
        cost = 10*(setpoint - x) ** 2 + 0.5*(dx)**2        
        return cost

    def validate(self, simulation_length = 1000):
        """Runs a simulation and calculates position, cumulative errror and average error.

        Args:
            simulation_length (int): (int): Simulation length in time periods. Defaults to 1000.

        Returns:
            retX, error, avg_error: Position. Cumulative error. Average error.
        """
        self.model.reset_initial_conditions()
        error = 0
        retX = []
        for i in range(simulation_length):
            x, dx, curr = self.model.get_states()
            error += abs(self.setpoint - x)  # greska je apsolutna razlika 
            retX.append(x)
            x_disc_ind = np.argmin(abs(self.disc_pos - x))   # smallest difference of current state to the closest discrete state
            dx_disc_ind = np.argmin(abs(self.disc_vel - dx))
            inew_disc_ind = np.argmin(abs(self.disc_curr - curr))

            action_ind = int(np.argmin(self.q_table[x_disc_ind, dx_disc_ind, inew_disc_ind, :])) # nalazimo indeks minimalne vrednosti stanja (one koja je najmanje kaznjiva, ovo se moze promeniti kasnije da bude nagrada umesto kazne)
            action = self.disc_action[action_ind] 

            self.model.set_action(action)

            xnew, dxnew, inew = self.model.next_states()
            
        avg_err = error/len(retX)
        print(f"Average error is {avg_err}")
        
        return retX, error, avg_err

    
    def write_list_to_file(self, list, filename):
        with open(filename, 'w') as f:
            for el in list:
                f.write(f"{el},")

