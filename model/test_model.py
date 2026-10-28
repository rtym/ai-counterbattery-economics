"""Tests of the article model: python -m pytest research/model/test_model.py"""
import numpy as np

import model as M


def test_gun_outcomes_counts_each_gun_once():
    rng = np.random.default_rng(1)
    # 10 detected guns, all engaged by artillery and by drones, both always destroy: 10, not 20
    des, dam, ea, ed = M.gun_outcomes(rng, np.array([10]), np.array([10]), 1.0, 1.0, np.array([10]), 1.0, 1.0, 0.0)
    assert des[0] == 10 and dam[0] == 0
    # 2 guns hit by both means: never more destroyed + damaged than detected, across many draws
    det = np.full(20000, 10)
    des, dam, _, _ = M.gun_outcomes(rng, det, np.full(20000, 2), 1.0, 1.0, np.full(20000, 2), 1.0, 1.0, 0.0)
    assert np.all(des + dam <= det)
    # expected destroyed = 10 * (1 - (1 - 0.2)(1 - 0.2)) = 3.6 (overlap removed), not 4
    assert abs(des.mean() - 3.6) < 0.05


def test_gun_outcomes_zero_detections():
    des, dam, _, _ = M.gun_outcomes(np.random.default_rng(0), np.array([0]), np.array([0]), 0.5, 0.6, np.array([0]),
                                    0.4, 0.1, 0.4)
    assert des[0] == 0 and dam[0] == 0


def test_psi_reduces_to_independent_rounds_without_interval():
    p0 = M.psi(np.array([1, 5]), np.array([10.0]), np.array([40.0]), np.array([12.0]), 0.0)
    p1 = M.psi(np.array([1, 5]), np.array([10.0]), np.array([40.0]), np.array([12.0]), 0.5)
    assert np.allclose(p0[:, 0], p1[:, 0]) and p1[0, 1] < p0[0, 1]
