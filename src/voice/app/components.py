import streamlit as st
import pandas as pd


def render_sidebar():
    """Renders the navigation/settings sidebar."""
    st.sidebar.title("🧠 Voice BCI Settings")
    st.sidebar.markdown("---")
    data_dir = st.sidebar.text_input(
        "BIDS Dataset Path",
        placeholder="/path/to/BIDS_dataset",
        help="Path to the root directory of your BIDS formatted EEG dataset.",
    )
    st.sidebar.markdown("---")
    return {"data_dir": data_dir}


def render_bids_summary(runs_data: list):
    """Displays dataset statistics and summary metrics."""
    if not runs_data:
        st.warning("No BIDS runs or EEG files found in the specified path.")
        return

    st.subheader("📊 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Recordings", len(runs_data))
    
    # Example metrics based on dataset contents
    subjects = set(item.get("subject") for item in runs_data if "subject" in item)
    col2.metric("Subjects", len(subjects) if subjects else "N/A")
    
    tasks = set(item.get("task") for item in runs_data if "task" in item)
    col3.metric("Tasks", len(tasks) if tasks else "N/A")

    st.markdown("### Scanned File Manifest")
    df = pd.DataFrame(runs_data)
    st.dataframe(df, use_container_width=True)


def render_eeg_viewer(eeg_data=None):
    """Placeholder or interactive widget for rendering EEG signals or power spectra."""
    st.subheader("📈 Signal Inspection")
    if eeg_data is None:
        st.info("Select a run from the dataset summary above to inspect signals.")
    else:
        st.line_chart(eeg_data)