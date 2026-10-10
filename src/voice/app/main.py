import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split

from voice.preprocessing.pipeline import EEGPreprocessor
from voice.analysis.features import MultiBandCSPExtractor
from voice.models.baseline_svm import CSPSVMClassifier

st.title("🧠 Voice: Inner Speech EEG Analyzer")

# Model Architecture Choice
model_choice = st.sidebar.selectbox(
    "Decoding Model Architecture",
    ["CSP + Linear SVM (Baseline)", "EEG-Conformer (Deep Learning)", "EEGNet Baseline"]
)

if model_choice == "CSP + Linear SVM (Baseline)":
    st.subheader("Classical Machine Learning Pipeline (CSP + SVM)")
    
    # Process when data is in session_state
    if "raw_data" in st.session_state and st.session_state.raw_data:
        lang = st.selectbox("Select Active Language", list(st.session_state.raw_data.keys()))
        raw, events, _, cn, meta = st.session_state.raw_data[lang]
        
        if st.button("Run CSP + SVM Baseline"):
            # 1. Preprocess
            prep = EEGPreprocessor(sfreq=raw.info['sfreq'], target_sfreq=128.0)
            X_raw = raw.get_data() # (n_channels, n_times)
            
            # Simple epoching mockup
            X_trials = np.array([X_raw[:, i:i+256] for i in range(0, X_raw.shape[1]-256, 256)])[:len(events)]
            y_trials = events[:, 2][:len(X_trials)]

            # 2. Train/Test Split
            X_train, X_test, y_train, y_test = train_test_split(X_trials, y_trials, test_size=0.3, random_state=23)

            # 3. Extract CSP Features
            csp_extractor = MultiBandCSPExtractor(n_components=4)
            X_train_csp = csp_extractor.fit_transform(X_train, y_train)
            X_test_csp = csp_extractor.transform(X_test)

            # 4. Evaluate Classifier
            clf = CSPSVMClassifier()
            clf.fit(X_train_csp, y_train)
            metrics = clf.evaluate(X_test_csp, y_test)

            st.success(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
            if metrics['auc']:
                st.metric("AUC Score", f"{metrics['auc']:.4f}")