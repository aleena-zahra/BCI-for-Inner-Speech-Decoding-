import streamlit as st
from voice.app.components import render_sidebar, render_bids_summary, render_eeg_viewer
from voice.data.bids_scanner import scan_bids_eeg_dataset


def main():
    # 1. Page Configuration
    st.set_page_config(
        page_title="Voice BCI - Inner Speech Analyzer",
        page_icon="🧠",
        layout="wide",
    )

    st.title("Inner Speech BIDS Data Analyzer")
    st.caption("Clean, inspect, and analyze EEG BIDS datasets for inner speech decoding.")

    # 2. Render Sidebar & Fetch Controls
    config = render_sidebar()
    data_dir = config["data_dir"]

    # 3. Main Dashboard Body
    if data_dir:
        st.info(f"Scanning directory: `{data_dir}`")
        try:
            # Call your backend logic from src/voice/data/
            runs = scan_bids_eeg_dataset(data_dir)
            
            # Render UI components with the backend results
            render_bids_summary(runs)
            
            st.markdown("---")
            render_eeg_viewer()

        except Exception as e:
            st.error(f"Error loading dataset: {e}")
    else:
        st.notice("Please enter a valid BIDS dataset directory path in the sidebar to begin.")


if __name__ == "__main__":
    main()