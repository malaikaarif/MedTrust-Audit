"""
Tests for evaluators/calibration.py and evaluators/high_conf_errors.py
Run with: pytest tests/
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from evaluators.calibration import compute_ece


def test_ece_is_zero_for_perfectly_calibrated_predictions():
    """If confidence always exactly matches actual accuracy, ECE should be ~0."""
    # 100 predictions, all correct, all at 100% confidence -> perfectly calibrated
    y_true = np.array([0] * 100)
    y_pred_probs = np.tile([1.0, 0.0], (100, 1))
    ece = compute_ece(y_true, y_pred_probs, n_bins=10)
    assert ece == 0.0


def test_ece_is_positive_for_overconfident_predictions():
    """A model that's always 99% confident but only right half the time should have high ECE."""
    y_true = np.array([0, 1] * 50)  # alternating true labels
    # model always predicts class 0 with 99% confidence, regardless of truth
    y_pred_probs = np.tile([0.99, 0.01], (100, 1))
    ece = compute_ece(y_true, y_pred_probs, n_bins=10)
    assert ece > 0.3  # should be substantially miscalibrated


def test_ece_returns_value_between_zero_and_one():
    np.random.seed(0)
    y_true = np.random.randint(0, 4, size=200)
    y_pred_probs = np.random.dirichlet(np.ones(4), size=200)
    ece = compute_ece(y_true, y_pred_probs, n_bins=10)
    assert 0.0 <= ece <= 1.0


def test_high_confidence_error_rate_matches_manual_calculation():
    """Regression check against the real verified numbers from the independent
    MobileNetV2 reproduction: 39 high-confidence errors out of 1,465 high-confidence
    predictions (threshold >= 0.9)."""
    from evaluators.high_conf_errors import find_high_confidence_errors

    y_true = np.load(os.path.join(os.path.dirname(__file__), '..', 'y_true.npy'))
    y_pred = np.load(os.path.join(os.path.dirname(__file__), '..', 'y_pred.npy'))
    y_pred_probs = np.load(os.path.join(os.path.dirname(__file__), '..', 'y_pred_probs.npy'))

    result = find_high_confidence_errors(y_true, y_pred, y_pred_probs)
    assert result['total_high_conf_predictions'] == 1465
    assert result['high_conf_errors'] == 39
    assert abs(result['high_confidence_error_rate'] - 0.0266) < 0.001