import numpy as np
from voice import InnerSpeechPipeline

# Shape: trials, channels, timepoints
X = np.random.randn(40, 64, 512)
y = np.array([0, 1] * 20)

pipeline = InnerSpeechPipeline()

metrics = pipeline.fit_evaluate(X, y)

print(metrics["accuracy"])
print(metrics["confusion_matrix"])