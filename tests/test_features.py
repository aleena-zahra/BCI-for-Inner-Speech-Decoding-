import numpy as np
import pytest

from voice.preprocessing.features import EEGPatchTokenizer, SpectralFeatureExtractor


def test_patch_tokenizer_shape_and_timing():
    signals = np.arange(2 * 3 * 32).reshape(2, 3, 32)
    tokenizer = EEGPatchTokenizer(patch_len=8, stride=4)
    patches, count = tokenizer.tokenize(signals)
    assert patches.shape == (2, 7, 24)
    assert count == 7
    metadata = tokenizer.get_token_metadata(32, sfreq=128)
    assert metadata["patch_duration_ms"] == 62.5


def test_spectral_bands_are_trial_channel_aligned():
    rng = np.random.default_rng(5)
    signals = rng.normal(size=(4, 3, 256))
    powers, frequencies, psd = SpectralFeatureExtractor(sfreq=128).compute_band_powers(signals)
    assert set(powers) == {"Delta", "Theta", "Alpha", "Beta", "Gamma"}
    assert all(values.shape == (4, 3) for values in powers.values())
    assert psd.shape == (4, 3, len(frequencies))
    assert frequencies.max() <= 64


def test_revin_normalizes_independent_trials():
    torch = pytest.importorskip("torch")
    from voice.models.layers import RevIN

    values = torch.randn(2, 3, 64) * 4 + 12
    normalized = RevIN(3, affine=False)(values)
    assert torch.allclose(normalized.mean(-1), torch.zeros(2, 3), atol=1e-5)
    assert torch.allclose(normalized.std(-1, unbiased=False), torch.ones(2, 3), atol=1e-4)
