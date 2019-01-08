from lcp import DataManager, Decider, SampleRegister
import pandas as pd
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
import numpy as np
import unittest


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
        dc = Decider(10)
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
