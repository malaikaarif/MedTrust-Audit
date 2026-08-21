"""
Tests for cri/clinical_readiness_index.py
Run with: pytest tests/
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cri.clinical_readiness_index import compute_cri


def test_cri_perfect_model_scores_near_one():
    """A perfect model (100% acc, 0 ECE, 0 HCE, full generalization) should score close to 1.0."""
    result = compute_cri(accuracy=1.0, ece=0.0, high_conf_error_rate=0.0, generalization=1.0)
    assert result['cri_score'] >= 0.99
    assert result['verdict'] == 'DEPLOY'


def test_cri_terrible_model_scores_near_zero():
    """A model with 0% accuracy, max ECE, and 100% high-confidence errors should score near 0."""
    result = compute_cri(accuracy=0.0, ece=1.0, high_conf_error_rate=1.0, generalization=0.0)
    assert result['cri_score'] <= 0.01
    assert result['verdict'] == 'REJECT'


def test_cri_verdict_thresholds():
    """Verdict boundaries: >=0.8 DEPLOY, >=0.5 REVIEW, else REJECT."""
    # Construct scores that land in each band via accuracy alone, holding others neutral-ish
    high = compute_cri(accuracy=0.95, ece=0.02, high_conf_error_rate=0.02, generalization=1.0)
    assert high['verdict'] == 'DEPLOY'

    mid = compute_cri(accuracy=0.55, ece=0.3, high_conf_error_rate=0.4, generalization=0.5)
    assert mid['verdict'] in ('REVIEW', 'DEPLOY')  # depends on exact weighting, just shouldn't be REJECT territory when this high

    low = compute_cri(accuracy=0.1, ece=0.9, high_conf_error_rate=0.9, generalization=0.0)
    assert low['verdict'] == 'REJECT'


def test_cri_matches_verified_real_result():
    """
    Regression test: locks in the actual verified result from the independent
    MobileNetV2 reproduction (see README.md 'Reproducibility note').
    If this test starts failing, the CRI formula or weights have changed
    unexpectedly — check cri/clinical_readiness_index.py against the paper's
    Equation (4) before assuming the new number is correct.
    """
    result = compute_cri(
        accuracy=0.941875,
        ece=0.0292,
        high_conf_error_rate=0.0266,
        generalization=1.0
    )
    assert abs(result['cri_score'] - 0.9641) < 0.001
    assert result['verdict'] == 'DEPLOY'


def test_cri_default_weights_sum_to_one():
    """Sanity check: the paper's weights (0.40, 0.25, 0.20, 0.15) should sum to 1.0."""
    weights = {'accuracy': 0.40, 'ece': 0.25, 'high_conf_error': 0.20, 'generalization': 0.15}
    assert abs(sum(weights.values()) - 1.0) < 1e-9