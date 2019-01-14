from lcp import DataManager, Decider, SampleRegister
from lcp import PriorSampler, SequentialSampler, Simulator
from lcp import chart2polar3d, get_random_dev_vec
import pandas as pd
import math
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
import numpy as np
from unittest.mock import MagicMock
import pytest

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
             "track": [1, 1],
             "t": [1, 2]})
        dm.set_time(2)
        assert_array_equal(
            dm.r,
            np.array([[2, 2, 2]]))
        assert_array_equal(
            dm.v,
            np.array([[2, 2, 2]]))
        assert_array_equal(
            dm.pre_v,
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


def test_chart2polar3d():
    x = np.array([1, 1, 0])
    polar = chart2polar3d(x)
    assert polar[0] == math.sqrt(2)
    assert polar[1] == math.pi/2
    assert polar[1] == math.pi/2


def test_get_random_dev_vec():
    x = np.array([1, 2, 3])
    random_dev_vec = get_random_dev_vec(x, np.pi/3)
    cos = np.dot(x, random_dev_vec)/(np.linalg.norm(x)*np.linalg.norm(random_dev_vec))
    assert cos == pytest.approx(np.cos(np.pi/3))


theta = {
    "re": {
        "mean": 1,
        "local": np.arange(2)
    },
    "dr0": {
        "mean": 1,
        "local": np.arange(2)
    },
    "eta": {
        "mean": 1,
        "local": np.arange(2)
    },
    "v0": {
        "mean": 1,
        "local": np.arange(2)
    },
    "fa": {
        "mean": 1,
        "local": np.arange(2)
    },
    "fr": {
        "mean": 1,
        "local": np.arange(2)
    }
}
x = np.array([[0, 0, 1],
              [0, 0, 0]])
pre_v = np.arange(6).reshape((2, 3))
cell_num = 2
sim = Simulator(x, pre_v)


class TestSimulator(object):
    def test_get_neighbor(self):
        n = sim.get_neighbor(0, theta)
        assert_array_equal(n, np.array([1]))

    def test_get_action(self):
        n = sim.get_neighbor(0, theta)[0]
        outer_f = sim.get_action(0, n, theta)
        assert outer_f.shape == (3,)

    def test_get_self_v(self):
        self_v = sim.get_self_v(0, theta)
        assert self_v.shape == (3,)

    def test_conduct_each(self):
        est_v = sim.conduct_each_cell(0, theta)
        assert est_v.shape == (3,)

    def test_conduct(self):
        est_v = sim.conduct(theta)
        assert est_v.shape == (2, 3)
