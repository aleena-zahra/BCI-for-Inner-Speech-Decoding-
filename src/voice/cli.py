
# src/voice/cli.py
from pathlib import Path
import sys
import subprocess

def launch_app():
    """Launch the Streamlit UI programmatically or via CLI."""
    app_path = Path(__file__).resolve().parent / "app" / "main.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), *sys.argv[1:]]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    launch_app()
