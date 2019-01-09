from lcp import DataManager, Decider, SampleRegister, PriorSampler, SequentialSampler
import pandas as pd
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
import numpy as np
from unittest.mock import MagicMock


class TestDataManager(object):
    def test_set_time(self):
        dm = DataManager()
        dm.df = pd.DataFrame(
            {"vx": [3, 2],
             "vy": [3, 2],
             "vz": [3, 2],
             "x": [1, 2],
             "y": [1, 2],
             "z": [1, 2],
             "t": [1, 2]})
        dm.set_time(1)
        assert_array_equal(
            dm.r,
            np.array([[1, 1, 1]]))
        assert_array_equal(
            dm.v,
            np.array([[3, 3, 3]]))


class TestDecider(object):
    def test_decide(self):
        dc = Decider(10, 1)
        dc.eps_vec = np.array([3, 2, 1])
        dc.v = np.array([0, 0])
        v1 = np.array([1.5, 0])
        accept0 = dc.decide(v1, 0)
        accept2 = dc.decide(v1, 2)
        assert accept0
        assert not accept2


class TestSampleRegister(object):
    def test_register_new(self):
        sr = SampleRegister(lambda x: 5, lambda x, y: 1)
        sr.theta_vec = np.array([1, 2])
        sr.w_vec = np.array([3, 4])
        sr.new_theta_vec = np.array([1])
        sr.new_w_vec = np.array([5])
        new_theta = 5
        sr.register_new(new_theta, 1)
        assert_array_equal(sr.new_theta_vec, np.array([1, 5]))
        assert_array_equal(sr.new_w_vec, np.array([5, 5/7]))

    def test_load_new(self):
        sr = SampleRegister(1, 1)
        sr.w_vec = np.array([0, 1])
        sr.theta_vec = np.array([1, 0])
        assert sr.choice() == 0


class TestPriorSampler(object):
    def test_sample_param(self):
        theta_key_vec = np.array(["a", "b"])
        boundary_dict = {
            "a": (0, 1),
            "b": (90, 100)
        }
        K_dict = {
            "a": np.array([[1, 0], [0, 1]]),
            "b": np.array([[1, 0], [0, 1]])
        }
        ps = PriorSampler(theta_key_vec, boundary_dict, K_dict)
        theta = ps.sample_param()
        assert theta.keys() == boundary_dict.keys()
        assert theta["a"]["local"].shape == (2,)
        assert theta["b"]["mean"] > 80
        assert theta["a"]["mean"] < 1.1

    def test_calc_prior_prob(self):
        theta_key_vec = np.array(["a", "b"])
        boundary_dict = {
            "a": (0, 1),
            "b": (90, 100)
        }
        K_dict = {
            "a": np.array([[1, 0], [0, 1]]),
            "b": np.array([[1, 0], [0, 1]])
        }
        ps = PriorSampler(theta_key_vec, boundary_dict, K_dict)
        theta = ps.sample_param()
        prior_prob = ps.calc_prior_prob(theta)
        assert prior_prob > 0
        assert prior_prob < 1


class TestSequentialSampler(object):
    def test_sample_param(self):
        theta_key_vec = np.array(["a", "b"])
        var_dict = {
            "a": 1,
            "b": 1
        }
        K_dict = {
            "a": np.array([[1, 0], [0, 1]]),
            "b": np.array([[1, 0], [0, 1]])
        }
        pre_theta = {
            "a": {
                "mean": 2,
                "local": np.array([0, 1])
            },
            "b":  {
                "mean": 2,
                "local": np.array([0, 1])
            }
        }
        ss = SequentialSampler(theta_key_vec, var_dict, K_dict)
        theta = ss.sample_param(pre_theta)
        assert theta.keys() == var_dict.keys()
        assert theta["a"]["local"].shape == (2,)

    def test_calc_prior_prob(self):
        theta_key_vec = np.array(["a", "b"])
        var_dict = {
            "a": 1,
            "b": 1
        }
        K_dict = {
            "a": np.array([[1, 0], [0, 1]]),
            "b": np.array([[1, 0], [0, 1]])
        }
        pre_theta = {
            "a": {
                "mean": 2,
                "local": np.array([0, 1])
            },
            "b":  {
                "mean": 2,
                "local": np.array([0, 1])
            }
        }
        ss = SequentialSampler(theta_key_vec, var_dict, K_dict)
        theta = ss.sample_param(pre_theta)
        transition_prob = ss.calc_transition_prob(pre_theta, theta)
        assert transition_prob > 0
        assert transition_prob < 1
