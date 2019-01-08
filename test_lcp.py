from lcp import DataManager, Decider
import pandas as pd
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
import numpy as np


def test_DataManager_set_time():
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


def test_Decider_decide():
    dc = Decider(10)
    dc.eps_vec = np.array([3, 2, 1])
    dc.v = np.array([0, 0])
    v1 = np.array([1.5, 0])
    accept0 = dc.decide(v1, 0)
    accept2 = dc.decide(v1, 2)
    assert accept0
    assert not accept2
