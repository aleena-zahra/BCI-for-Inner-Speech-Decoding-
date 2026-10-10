
# src/voice/cli.py
import os
import sys
import subprocess

def launch_app():
    """Launch the Streamlit UI programmatically or via CLI."""
    # Find the path to run_streamlit_app.py inside the package
    app_path = os.path.join(os.path.dirname(__file__), "app", "run_streamlit_app.py")
    
    # Fallback if run_streamlit_app.py is in the root examples folder
    if not os.path.exists(app_path):
        app_path = os.path.join(os.getcwd(), "examples", "run_streamlit_app.py")

    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    subprocess.run(cmd)

if __name__ == "__main__":
    launch_app()
