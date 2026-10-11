from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

import voice
from voice import cli


APP_PATH = Path(cli.__file__).resolve().parent / "app" / "main.py"


def test_launcher_uses_packaged_app_from_another_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.sys, "argv", ["voice-ui", "--server.port=8502"])
    with patch.object(cli.subprocess, "run") as run:
        voice.launch_app()
    command = run.call_args.args[0]
    assert command[:4] == [cli.sys.executable, "-m", "streamlit", "run"]
    assert Path(command[4]) == APP_PATH
    assert command[5:] == ["--server.port=8502"]
    assert run.call_args.kwargs == {"check": True}
    assert APP_PATH.is_file()


@pytest.mark.parametrize(
    "page,message",
    [
        ("preprocessing", "Load a dataset"),
        ("features", "Run preprocessing first"),
        ("training", "Extract features first"),
        ("interpretability", "Train a model first"),
    ],
)
def test_pages_explain_missing_prerequisites(page, message):
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    app.switch_page(f"pages/{page}.py").run(timeout=30)
    assert not app.exception
    assert any(message in item.value for item in app.info)


def test_settings_workflow_and_initial_render(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    assert not app.exception
    assert app.header[0].value == "Dataset"
    assert "not yet connected" in app.info[0].value

    app.button[0].click().run()
    assert app.warning
    assert not app.session_state["raw_data"]

    app.text_input(key="dataset_path").set_value(str(tmp_path))
    app.button[0].click().run()
    assert app.session_state["raw_data"]["active"]["path"] == str(tmp_path)

    app.switch_page("pages/preprocessing.py").run()
    app.button[0].click().run()
    assert app.session_state["proc_data"]["active"]["apply_car"] is True

    app.switch_page("pages/features.py").run()
    app.button[0].click().run()
    assert app.session_state["tokenized_data"]["active"]["patch_length"] == 25

    app.switch_page("pages/training.py").run()
    app.button[0].click().run()
    assert app.session_state["trained_models"]["active"]["model"] == "EEGNet"

    app.switch_page("pages/interpretability.py").run()
    app.button[0].click().run()
    assert app.session_state["latent_embeddings"]["active"]["projection"] == "PCA"
    assert not app.exception
