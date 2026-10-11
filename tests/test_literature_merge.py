import mne
import numpy as np
import pytest

from voice.preprocessing import EEGPreprocessor, LiteraturePreprocessingPipeline, zscore_normalize
from voice.evaluation.latent_space import LatentSpaceAuditor
from voice.evaluation.splits import evaluation_folds


def test_zscore_constant_channels_and_microvolt_quality_gate():
    time = np.arange(1024) / 128
    trial = np.sin(2 * np.pi * 10 * time)
    signals = np.tile(trial, (3, 2, 1)) * np.array([1, 500, 2])[:, None, None]
    processor = EEGPreprocessor(sfreq=128, target_sfreq=128, t_start=1, t_end=3,
                                artifact_threshold=100, zscore=True)
    processed, labels, kept = processor.fit_transform_with_indices(signals, np.array([1, 2, 3]))
    assert kept.tolist() == [0, 2]
    assert labels.tolist() == [1, 3]
    assert np.allclose(processed.mean(-1), 0, atol=1e-6)
    assert np.allclose(processed.std(-1), 1, atol=1e-5)
    assert np.array_equal(zscore_normalize(np.ones((2, 3, 64))), np.zeros((2, 3, 64)))


def test_continuous_fir_resampling_preserves_event_times_and_source_raw():
    rng = np.random.default_rng(6)
    raw = mne.io.RawArray(rng.normal(size=(3, 5120)) * 1e-6,
                         mne.create_info(["F3", "F4", "Cz"], 256, "eeg"),
                         first_samp=512, verbose=False)
    events = np.array([[1024, 0, 1], [2560, 0, 2], [4864, 0, 1]])
    original_data = raw.get_data().copy()
    original_events = events.copy()
    pipeline = LiteraturePreprocessingPipeline(target_sfreq=128, l_freq=1, h_freq=40,
                                                zscore=False)
    values, synced, epochs = pipeline.run_pipeline(raw, events, {"a": 1, "b": 2}, tmin=0, tmax=1)
    assert values.shape == (3, 3, 129)
    original_times = (events[:, 0] - raw.first_samp) / raw.info["sfreq"]
    new_times = (synced[:, 0] - round(raw.first_samp * 128 / 256)) / epochs.info["sfreq"]
    assert np.allclose(original_times, new_times, atol=1 / 128)
    assert np.array_equal(synced[:, 2], events[:, 2])
    assert np.allclose(values.mean(axis=1), 0, atol=1e-6)
    assert np.array_equal(raw.get_data(), original_data)
    assert np.array_equal(events, original_events)


def test_trial_fir_profile_retains_passband_and_suppresses_out_of_band_alias():
    time = np.arange(4096) / 512
    values = np.sin(2 * np.pi * 10 * time) + np.sin(2 * np.pi * 180 * time)
    signals = np.tile(values, (2, 2, 1))
    processor = EEGPreprocessor(sfreq=512, target_sfreq=250, lowcut=1, highcut=100,
                                t_start=2, t_end=4, filter_method="fir")
    processed = processor.fit_transform(signals)
    assert processed.shape == (2, 2, 500)
    spectrum = np.abs(np.fft.rfft(processed[0, 0]))
    freq = np.fft.rfftfreq(500, 1 / 250)
    assert spectrum[np.argmin(abs(freq - 10))] > 50 * spectrum[np.argmin(abs(freq - 70))]


def test_ica_exclusions_change_signal_and_transform_reuses_training_fit():
    signals = np.random.default_rng(4).normal(size=(6, 3, 1024)).astype(np.float32)
    config = dict(sfreq=128, target_sfreq=128, t_start=0, t_end=8, ica_components=2)
    processor = EEGPreprocessor(**config, ica_exclude=(0,))
    cleaned = processor.fit_transform(signals[:4])
    filtered = EEGPreprocessor(**{key: value for key, value in config.items() if key != "ica_components"}).transform(signals[:4])
    assert not np.allclose(cleaned, filtered)
    fitted = processor.ica_
    heldout = processor.transform(signals[4:])
    assert processor.ica_ is fitted
    assert heldout.shape == (2, 3, 1024)
    assert np.isfinite(heldout).all()
    with pytest.raises(RuntimeError, match="Fit the preprocessor"):
        EEGPreprocessor(**config).transform(signals)
    with pytest.raises(ValueError, match="exclusions"):
        EEGPreprocessor(**config, ica_exclude=(3,)).fit(signals)


def test_ica_review_folds_exclude_validation_and_heldout_trials():
    labels = np.tile([1, 2], 30)
    subjects = np.repeat(["s1", "s2", "s3"], 20)
    for _, train, validation, test in evaluation_folds(labels, subjects, strategy="loso", deep=True):
        assert not set(train) & set(test)
        assert not set(train) & set(validation)
        assert not set(validation) & set(test)
        assert len(train) + len(validation) + len(test) == len(labels)


def test_word_and_subject_audit_distinguishes_undefined_subject_metric():
    features = np.random.default_rng(5).normal(size=(12, 4))
    labels = np.tile([1, 2], 6)
    audited = LatentSpaceAuditor.audit(features, labels, np.repeat("s1", 12))
    assert audited["projections"]["pca"].shape == (12, 2)
    assert audited["class_metrics"]["silhouette_score"] is not None
    assert audited["subject_metrics"]["silhouette_score"] is None
