from agent import Agent
import numpy as np
import matplotlib.pyplot as plt


def read_el_from_file(filename):
    with open(filename, 'r') as f:
        s = str.split(f.read(), ',')
        return [float(x) for x in s]

# loads error from file and plots it. Needed for documentation
def plot_error():
    l = read_el_from_file("best_scores_q_learning_1.txt")
    plt.plot(range(len(l[100000:])), l[100000:])
    plt.title("Q learning control, model 2")
    plt.xlabel("Episode")
    plt.ylabel("Best average error")
    plt.ylim((0.0, 1.05))
    plt.grid(visible = True, linestyle='-')
    plt.show()

# animates ball control and plots ball position
def plot_displacement(a):
    x = a.control(simulation_length=1000)
    t = np.linspace(0, len(x)*0.02, len(x))
    plt.plot(t, x)
    plt.plot([t[0], t[-1]], [0.5, 0.5], color='r', linestyle='dashed')
    plt.title("Q learning control, model 2")
    plt.xlabel("Time [s]")
    plt.ylim((0.0, 1.05))
    plt.ylabel("Displacement [m]")
    plt.grid(visible = True, linestyle='-')
    plt.show()


if __name__ == '__main__':

    ######################################################################
    ######## how main should look like if there is a trained model? ######
    ######################################################################
    # # make an object
    a = Agent()
    # # load agent parameters
    # a.load_agent_params()
    # # initialize model parameters
    # a.init_model_params([1.0, 0.0, 0.0]) 
    # # optionaly, demonstrate control with animation, or more leraning episodes
    # a.control()
    # # a.learn(type = "qlearning", epsilon_start=0.1, max_episodes=10000, tolerance_error=0.05)
    
    # after learning, if information about best score is needed, save it to file
    # a.write_list_to_file(a.best_scores, f"best_scores_q_learning_{1}.txt")



    ########################################################################
    ######## how main should look like if training starts from zero? ######
    #######################################################################
    # # make object
    # a = Agent()
    # # initialize model parameters
    # a.init_model_params(initial_conditions=[1.0, 0.0, 0.0])
    # # initial agent parameters
    # a.init_agent_params()
    # # then learn
    # a.learn(type = "qlearning", n_of_epizodes=3001, epsilon_start=0.5)
   

    ################
    ## optionally ##
    ################
    # # plot table that shows best states for given state variables
    # a.plot_pos_vel_table()
    # a.plot_pos_curr_table()
    # a.plot_vel_curr_table()
    


