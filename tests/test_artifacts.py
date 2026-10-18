import numpy as np
import pytest

from eegmr.artifacts import WeightedDecoding, flag_outliers, outlier_scores, trial_features
from eegmr.decoding import DecodingVariant


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    n, d = 120, 16
    F = rng.standard_normal((n, d))
    eeg = rng.standard_normal((n, 17, 150)) + 0.5 * (F[:, :1, None] * np.sin(np.arange(150) / 7))
    return F, eeg


def test_trial_features_shape_and_spike_detection():
    rng = np.random.default_rng(1)
    raw = rng.standard_normal((60, 17, 750))
    raw[7, 3, 100] += 80.0  # one big spike
    f = trial_features(raw)
    assert f.shape == (60, 17 * 5) and np.isfinite(f).all()
    scores = outlier_scores(f)
    assert scores.shape == (60,) and np.argmax(scores) == 7
    assert flag_outliers(scores, 0.05).sum() == 3 and flag_outliers(scores, 0.05)[7]


def test_unit_weights_equal_unweighted_decoder(toy):
    F, eeg = toy
    kw = dict(dec_alpha=100.0, whiten_input=False, window=slice(5, 60), device="cpu")
    base = DecodingVariant(**kw).fit(F[:100], eeg[:100]).similarity(eeg[100:], F[100:])
    unit = WeightedDecoding(**kw, sample_weight=np.ones(100)).fit(F[:100], eeg[:100]).similarity(eeg[100:], F[100:])
    np.testing.assert_allclose(unit, base, atol=1e-5)


def test_zero_weight_equals_dropping(toy):
    F, eeg = toy
    kw = dict(dec_alpha=100.0, whiten_input=False, window=slice(5, 60), device="cpu")
    w = np.ones(100)
    w[:10] = 0.0
    weighted = WeightedDecoding(**kw, sample_weight=w).fit(F[:100], eeg[:100]).similarity(eeg[100:], F[100:])
    dropped = DecodingVariant(**kw).fit(F[10:100], eeg[10:100]).similarity(eeg[100:], F[100:])
    np.testing.assert_allclose(weighted, dropped, atol=1e-5)
