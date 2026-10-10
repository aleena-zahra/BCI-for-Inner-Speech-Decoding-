import sys
import subprocess
from pathlib import Path

def run_app():
    """Launches the Streamlit app programmatically."""
    app_path = Path(__file__).parent / "app" / "main.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])