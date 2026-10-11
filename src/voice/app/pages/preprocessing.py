import streamlit as st

st.header("Preprocessing")

if not st.session_state.raw_data:
    st.info("Load a dataset on the Dataset page first.")
    st.stop()

st.number_input("Low cutoff (Hz)", min_value=0.1, value=0.5, key="low_cutoff")
st.number_input("High cutoff (Hz)", min_value=1.0, value=100.0, key="high_cutoff")
st.checkbox("Apply common average reference", value=True, key="apply_car")

if st.button("Run preprocessing", type="primary"):
    st.session_state.proc_data["active"] = {
        "low_cutoff": st.session_state.low_cutoff,
        "high_cutoff": st.session_state.high_cutoff,
        "apply_car": st.session_state.apply_car,
    }
    st.success("Preprocessing settings recorded. Connect the library pipeline here next.")