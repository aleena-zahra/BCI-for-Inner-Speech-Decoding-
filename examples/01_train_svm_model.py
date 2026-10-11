import numpy as np
from voice import InnerSpeechPipeline


def main() -> None:
    """Run the complete preprocessing, CSP, and SVM workflow."""
    np.random.seed(42)
    n_trials, n_channels, n_timepoints = 100, 64, 512
    signals = np.random.randn(n_trials, n_channels, n_timepoints)
    labels = np.random.choice([0, 1], size=n_trials)

    pipeline = InnerSpeechPipeline(test_size=0.2, random_state=42)
    metrics = pipeline.fit_evaluate(signals, labels)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("Confusion matrix:\n", metrics["confusion_matrix"])

if __name__ == "__main__":
    main()
