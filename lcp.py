import pandas as pd
import numpy as np


class DataManager:
    """
    This class manage the data.
    In particular, provide r and v for one time point.
    """
    def file_load(self, fname):
        self.df = pd.read_csv(fname)

    def set_time(self, t):
        selected_df = self.df.loc[self.df.t == t]
        self.r = selected_df[["x", "y", "z"]]
        self.v = selected_df[["vx", "vy", "vz"]]


class Decider:
    """
    This class calculate difference between simulation and real data,
    and decide whether simulation results is acceptable.
    """
    def __init__(self, length, min_eps=1.0e-4, order=2):
        """
        Initialize epx vec, which represents accetance threshold for each round
        Waring:: Eps vec should be considered later
        """
        self.eps_vec = min_eps * np.arange(length, 1)**order

    def load_data(self, dm):
        """
        Aquire the real data velocity used in evaluation
        """
        self.v = dm.v

    def decide(self, v, t):
        """
        Decide accetance
        """
        eps = self.eps_vec[t]
        acceptance = np.linalg.norm(self.v - v) < eps
        return(acceptance)


class SampleRegister:
    """
    This class manage sampled parameters and weight for each of them.
    """
    def __init__(self, prior, transition):
        """
        Set prior function and transition function of parameters.
        Initialize vector of new_theta and new_w
        """
        self.prior = prior
        self.transition = transition
        self.new_w_vec = np.array([])
        self.new_theta_vec = np.array([])

    def register_new(self, new_theta, t):
        """
        Register a new parameter.
        Simultaneously it calculate weight for the new parameter,
        and register it.
        """
        self.new_theta_vec = np.append(self.new_theta_vec, new_theta)
        if t > 0:
            w = self.prior(new_theta)/np.sum(
                [w * self.transition(theta, new_theta)
                 for theta, w in zip(self.theta_vec, self.w_vec)])
        else:
            w = 1
        self.new_w_vec = np.append(self.new_w_vec, w)

    def load_new(self):
        """
        Refresh theta_vec and w_vec
        """
        self.theta_vec = self.new_theta_vec
        self.new_theta_vec = np.array([])
        self.w_vec = self.new_w_vec
        self.new_w_vec = np.array([])

    def choice(self):
        """
        Randomly choose parameters based on weight
        """
        return(
            np.random.choice(
                self.theta_vec,
                size=1, p=self.w_vec/np.sum(self.w_vec)))
