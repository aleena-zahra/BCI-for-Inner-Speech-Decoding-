# Voice: EEG Inner Speech Decoding

Voice is a Python library for EEG trial preprocessing, feature extraction, classification, and analysis. It also includes a Streamlit dashboard for inspecting the current workflow and recording its settings.

## Run the dashboard on Windows

From the repository root, create a virtual environment and install the library with its dashboard dependencies:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[ui]"
```

Start the dashboard:

```powershell
voice-ui --server.address=127.0.0.1 --server.port=8501
```

Open <http://127.0.0.1:8501>. Stop the server by pressing `Ctrl+C` in the terminal where it is running.

If the virtual environment is not activated, launch it directly:

```powershell
.\.venv\Scripts\voice-ui.exe --server.address=127.0.0.1 --server.port=8501
```

The launcher works from other working directories too. The editable install (`-e`) makes the local `src/voice` package available to the command.

## What the dashboard currently does

The dashboard has Dataset, Preprocessing, Features, Training, and Interpretability pages. At present, the pages let you register a dataset path and record pipeline settings. They do not yet load recordings into the library, run preprocessing, train models, or calculate latent-space analyses. The interface displays this status when it opens.

## Library contents

The `src/voice` package contains the trial preprocessing pipeline, multiband CSP feature extraction, an SVM classifier, EEGNet and EEGConformer model definitions, evaluation utilities, and the `InnerSpeechPipeline` API. The Streamlit dashboard lives under `src/voice/app`; `voice-ui` launches its `main.py` entry point.

The separate `POC/app.py` is an experimental application with its own pipeline implementation. It is not the dashboard launched by `voice-ui`.
