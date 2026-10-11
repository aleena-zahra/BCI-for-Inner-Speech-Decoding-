import streamlit as st

st.header("Training")

if not st.session_state.tokenized_data:
    st.info("Extract features first.")
    st.stop()

model_name = st.selectbox("Model", ["EEGNet", "EEGConformer", "CSP + SVM"], key="model_name")
epochs = st.number_input("Epochs", min_value=1, max_value=100, value=10, key="epochs")

if st.button("Train model", type="primary"):
    st.session_state.trained_models["active"] = {
        "model": model_name,
        "epochs": epochs,
    }
    st.success("Training settings recorded. Connect model training here next.")