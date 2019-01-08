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
