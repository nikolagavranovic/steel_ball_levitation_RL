import numpy as np
import math
import random as rand

class BallLevitationModel():

    def __init__(self, g, m, k_fr, ki, Ta, kc, initial_conditions, T):
        """ Initializes system parameters 

        Args:
            g (float): Gravity constant
            m (float): Ball mass
            k_fr (float): Friction coefitient
            ki (float): Coil gain
            Ta (float): Coil time constant    
            kc (float): Coil inductance [V/A]        
            initial_conditions (list): Initial conditions of system
            T (float): Time period of discrete system
        """
        if len(initial_conditions) != 3:
            print("System is third order so three initial conditions must be passed!")
            exit(1)

        # setting properties
        self.g = g
        self.k_fr = k_fr
        self.m = m  # 10 gr
        self.ki = ki
        self.Ta = Ta
        self.kc = kc
        self.T = T
        self.initial_conditions = initial_conditions
        self.states = initial_conditions
        self.V = 0   # action/input voltage    
        

    def reset_initial_conditions(self):
            # nececcary to copy initial_conditions BY VALUE, otherwise every change in states
            # will affect initial_conditions
            self.states = self.initial_conditions[:]  # reseting initial conditions to default

    def find_minimum_action(self):
        """ Finds minimum action that needs to be applied to ball to get the ball from bottom to middle 
        (position = 0 to position = 0.5). This function can be used before agent starts learning to find
        range of input actions.
        """
        V = 0.5  # starting voltage is 0.5V
        IsReached = False
        while not IsReached: 
            self.reset_initial_conditions()
            self.set_action(V)
            t = 0
            while t < 1:  # this period is 1 seconds if T = 0.02, in other words, checks if within 1 second ball can levitate at position 0.5
                pos = self.next_states()[0]
                if pos <= 0.5:
                    IsReached = True
                t += self.T

            V += 0.5  # step for increasing voltage is 0.5
        
        self.reset_initial_conditions()
        return V

    def find_maximum_current(self, V):     
        """Finds maximum current for specific input voltage. Needed for current discretisation.
        """   
        self.reset_initial_conditions()
        self.set_action(V)
        t = 0
        while t < 1: 
            curr = self.next_states()[2]
            t += self.T
            
        self.reset_initial_conditions()
        return curr

    def get_states(self):
        return self.states

    def set_action(self, V):
        """ Sets voltage as input action.

        Args:
            V (float): Input voltage
        """
        self.V = V
      
    def next_states(self):
        """Simulates next state of state variables (position, velocity and current). 
        Model is given with reccurence equation, and for discretisation is used Euler I
        """
                
        # For computing next state, multiple steps procedure is used (fixed to 5, but can be modified). 
        # In case of not doing this, input action (voltage) will not affect
        # neither ball position or ball velocity immediatly, because it apears only in third
        # equation (position is only affected by velocity).
        # This will lead to confusion in Q(s, a) matrix, because some voltage apliance will not affect position/velocity change.
        # Agent should eventualy learn optimal policy, but it will be much harder.
        x = self.states
        g = self.g
        m = self.m
        k_fr = self.k_fr
        kc = self.kc
        Ta = self.Ta
        ki = self.ki
        V = self.V
        x0 = 0  # offset of coil relative to 0 position (coil is set to be at 0, so the offset is 0. Can be changed.)
        
        A = kc/m        

        for i in range(5):

            T_temp = (self.T/5)
            
            self.states[0] +=  T_temp* x[1]
            self.states[1] += T_temp*(g - k_fr/m*x[1] - A*x[2]**2/(x[0] - x0)**2)
            self.states[2] += T_temp*(1/Ta*(-x[2] + ki*V))

            if abs(self.states[2] > 100000):
                print("danger")

        # saturation, ball cannot pass through plates, which are on positions 0.05 (upper) and 1.0 (lower)
        if (self.states[0] < 0.05):
            self.states[0] = 0.05
            self.states[1] = 0
        elif (self.states[0] > 1):
            self.states[0] = 1
            self.states[1] = 0

        return self.states


