import pandas as pd
import numpy as np
from scipy import stats
import math


class DataManager:
    """
    This class manage the data.
    In particular, provide r and v for one time point.
    """
    def file_load(self, fname):
        self.df = pd.read_csv(fname)

    def set_time(self, t):
        selected_df = self.df.loc[self.df.t == t]
        self.r = selected_df[["x", "y", "z"]].values
        self.v = selected_df[["vx", "vy", "vz"]].values


class Decider:
    """
    This class calculate difference between simulation and real data,
    and decide whether simulation results is acceptable.
    """
    def __init__(self, length, v, min_eps=1.0e-4, order=2):
        """
        Initialize epx vec, which represents accetance threshold for each round
        Aquire the real data velocity used in evaluation
        Waring:: Eps vec should be considered later
        """
        self.eps_vec = min_eps * np.arange(length, 1)**order
        self.v = v

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
                size=1, p=self.w_vec/np.sum(self.w_vec))[0])


class PriorSampler:
    """
    Sample parameters from prior distribution
    """
    def __init__(self, theta_key_vec, boundary_dict, K_dict):
        self.theta_key_vec = theta_key_vec
        self.boundary_dict = boundary_dict
        self.K_dict = K_dict
        point_num = K_dict[theta_key_vec[0]].shape[0]
        self.point_num_vec = np.full(point_num, 1.0)

    def sample_param(self):
        """
        Sample parameters based on prior distribution
        Mean of each parameter sampled from uniform distribution
        Local values are sampled from multivariate gaussian,
        which reflect spatial location of each cell
        """
        theta = {}
        for theta_key in self.theta_key_vec:
            theta[theta_key] = {}
            theta[theta_key]["mean"] = np.random.uniform(
                self.boundary_dict[theta_key][0],
                self.boundary_dict[theta_key][1],
                size=1)[0]
            # loca parameters sampled from mean
            theta[theta_key]["local"] = np.random.multivariate_normal(
                theta[theta_key]["mean"] * self.point_num_vec,
                self.K_dict[theta_key])
        return(theta)

    def calc_prior_prob(self, theta):
        """
        Calculate the likelihood of theta based on prior distribution
        """
        prob = 1
        for theta_key in self.theta_key_vec:
            interval = self.boundary_dict[theta_key][1] - self.boundary_dict[theta_key][0]
            mean_prob = 1.0/interval
            local_prob = stats.multivariate_normal(
                theta[theta_key]["mean"] * self.point_num_vec,
                self.K_dict[theta_key]).pdf(theta[theta_key]["local"])
            prob *= mean_prob * local_prob
        return(prob)


class SequentialSampler(PriorSampler):
    """
    Parameter sampling based on previous parameter
    """
    def __init__(self, theta_key_vec, var_dict, K_dict):
        self.theta_key_vec = theta_key_vec
        self.var_dict = var_dict
        self.K_dict = K_dict
        point_num = K_dict[theta_key_vec[0]].shape[0]
        self.point_num_vec = np.full(point_num, 1.0)

    def sample_param(self, pre_theta):
        """
        Sample parameters from previous theta
        """
        theta = {}
        for theta_key in self.theta_key_vec:
            theta[theta_key] = {}
            # mean parameter sampled from previous mean
            theta[theta_key]["mean"] = np.random.normal(
                pre_theta[theta_key]["mean"],
                self.var_dict[theta_key],
                size=1)[0]
            # local parameters sampled from previous local
            theta[theta_key]["local"] = np.random.multivariate_normal(
                pre_theta[theta_key]["local"],
                self.K_dict[theta_key])
        return(theta)

    def calc_transition_prob(self, pre_theta, theta):
        """
        Calculate transition probability from pre_theta to theta
        """
        prob = 1
        for theta_key in self.theta_key_vec:
            # mean parameter sampled from previous mean
            mean_prob = stats.norm.pdf(
                theta[theta_key]["mean"],
                loc=pre_theta[theta_key]["mean"],
                scale=math.sqrt(self.var_dict[theta_key]))
            # local parameters sampled from previous local
            local_prob = stats.multivariate_normal(
                pre_theta[theta_key]["local"],
                self.K_dict[theta_key]).pdf(theta[theta_key]["local"])
            prob *= mean_prob * local_prob
        return(prob)
