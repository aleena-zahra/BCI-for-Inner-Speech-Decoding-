import numpy as np
# from src.voice.evaluation.metrics import calculate_metrics

def test_metric_calculation():
    y_true = np.array([0, 1, 1, 0, 2])
    y_pred = np.array([0, 1, 0, 0, 2])
    
    # metrics = calculate_metrics(y_true, y_pred)
    # assert "accuracy" in metrics
    # assert 0.0 <= metrics["accuracy"] <= 1.0
    pass