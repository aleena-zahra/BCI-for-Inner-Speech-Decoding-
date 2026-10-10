import numpy as np
from src.voice.models.svm import CSPSVMClassifier

def main():
    # 1. Create synthetic BCI feature data (e.g., 100 trials, 20 extracted features)
    np.random.seed(42)
    n_samples, n_features = 100, 20
    
    X = np.random.randn(n_samples, n_features)
    # Binary target for inner speech classes (e.g., class 0 vs class 1)
    y = np.random.choice([0, 1], size=n_samples)

    # Train/Test split
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 2. Instantiate and fit the model
    print("Training CSPSVMClassifier...")
    clf = CSPSVMClassifier(C=1.0)
    clf.fit(X_train, y_train)

    # 3. Evaluate and output metrics
    metrics = clf.evaluate(X_test, y_test)
    
    print("\n--- Evaluation Results ---")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"ROC-AUC  : {metrics['auc'] if metrics['auc'] is not None else 'N/A'}")
    print("Confusion Matrix:\n", metrics['confusion_matrix'])

if __name__ == "__main__":
    main()