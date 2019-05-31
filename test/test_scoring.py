# TODO: generate more exhaustive tests
import numpy as np

from ..src import scoring


def test_rss():
    y = np.ones(10)
    y_hat = np.arange(1, 11)
    assert scoring.rss(y, y_hat) == 285.0


def test_tss():
    y = np.arange(1, 11)
    assert scoring.tss(y) == 82.5


def test_ess():
    y = np.ones(10)
    y_hat = np.arange(1, 11)
    assert scoring.ess(y, y_hat) == 285.0


def test_r2():
    y = np.arange(1, 11)
    y_hat = np.arange(1, 11)
    assert scoring.r2(y, y_hat) == 1.0
