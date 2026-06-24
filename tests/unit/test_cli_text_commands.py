import json

from cli import main as cli_main
from config.settings import Experiment, PipelineConfig


def make_run(tmp_path) -> None:
    experiment = Experiment(
        name="exp1",
        run_dir=tmp_path / "exp1",
        config=PipelineConfig(runs_dir=tmp_path),
    )
    experiment.run_dir.mkdir(parents=True, exist_ok=True)


def test_run_ocr_command_invokes_stage(monkeypatch, tmp_path, capsys) -> None:
    make_run(tmp_path)
    calls = []

    def fake_run_ocr(experiment, force):
        calls.append((experiment.name, force))
        return 3

    monkeypatch.setattr(cli_main, "run_ocr", fake_run_ocr)

    status = cli_main.main(
        ["run-ocr", "--runs-dir", str(tmp_path), "--experiment-name", "exp1", "--force"]
    )

    assert status == 0
    assert calls == [("exp1", True)]
    assert json.loads(capsys.readouterr().out)["ocr_records"] == 3


def test_run_asr_command_invokes_stage(monkeypatch, tmp_path, capsys) -> None:
    make_run(tmp_path)
    calls = []

    def fake_run_asr(experiment, force):
        calls.append((experiment.name, force))
        return 2

    monkeypatch.setattr(cli_main, "run_asr", fake_run_asr)

    status = cli_main.main(["run-asr", "--runs-dir", str(tmp_path), "--experiment-name", "exp1"])

    assert status == 0
    assert calls == [("exp1", False)]
    assert json.loads(capsys.readouterr().out)["asr_segments"] == 2


def test_build_text_index_command_invokes_stage(monkeypatch, tmp_path, capsys) -> None:
    make_run(tmp_path)
    calls = []

    def fake_build_text_index_from_artifacts(experiment, force):
        calls.append((experiment.name, force))
        return 5

    monkeypatch.setattr(
        cli_main,
        "build_text_index_from_artifacts",
        fake_build_text_index_from_artifacts,
    )

    status = cli_main.main(
        ["build-text-index", "--runs-dir", str(tmp_path), "--experiment-name", "exp1"]
    )

    assert status == 0
    assert calls == [("exp1", False)]
    assert json.loads(capsys.readouterr().out)["text_documents"] == 5
