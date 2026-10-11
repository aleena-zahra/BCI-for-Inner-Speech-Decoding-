import numpy as np
import pytest

torch = pytest.importorskip("torch")

from voice.models.eegconformer import EEGConformer
from voice.models.eegnet import EEGNet
from voice.models.training import evaluate_deep_model
from voice.evaluation.interpretability import conformer_attention_summary, eegnet_channel_saliency
from voice.preprocessing.pipeline import EEGPreprocessor


@pytest.mark.parametrize("architecture", ["EEGNet", "EEGConformer"])
def test_deep_model_training_returns_heldout_predictions(architecture):
    rng = np.random.default_rng(7)
    signals = rng.normal(size=(24, 8, 128)).astype(np.float32)
    labels = np.tile([0, 1, 2], 8)
    subjects = np.repeat(["s1", "s2", "s3"], 8)
    result = evaluate_deep_model(
        signals,
        labels,
        subjects,
        model_name=architecture,
        strategy="trial",
        epochs=1,
        batch_size=8,
        patch_len=16,
        stride=8,
        preprocessor=EEGPreprocessor(sfreq=128, target_sfreq=128, t_start=0, t_end=1, zscore=True),
    )
    assert len(result["y_true"]) == len(result["y_pred"])
    assert result["features"].shape[0] == len(result["y_true"])
    assert result["confusion_matrix"].shape == (3, 3)
    assert len(result["fold_test_subjects"]["trial holdout"]) == len(result["y_true"])
    assert result["fold_preprocessors"]["trial holdout"].zscore


def test_conformer_patch_shape_and_eegnet_forward():
    signals = torch.randn(2, 8, 128)
    eegnet = EEGNet(n_classes=3, n_channels=8, n_timepoints=128)
    conformer = EEGConformer(
        n_classes=3, n_channels=8, n_timepoints=128,
        d_model=16, n_heads=4, n_layers=1, patch_len=16, stride=8,
    )
    assert eegnet(signals).shape == (2, 3)
    assert conformer(signals).shape == (2, 3)
    attention = conformer.attention_maps(signals)
    assert attention[0].shape == (2, 4, 15, 15)
    importance = eegnet_channel_saliency(eegnet, signals)
    assert importance.shape == (8,)
    assert np.isclose(importance.sum(), 1.0)
    head_one = conformer_attention_summary(conformer, signals.numpy(), head=0, batch_size=1)
    head_batch = conformer_attention_summary(conformer, signals.numpy(), head=0, batch_size=2)
    assert np.allclose(head_one[0], head_batch[0], atol=1e-6)
    assert np.allclose(head_one[0].sum(axis=-1), 1.0, atol=1e-6)
