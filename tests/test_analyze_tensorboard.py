"""Tests for TensorBoard training metric analysis."""
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

from torch.utils.tensorboard import SummaryWriter

from analyze_tensorboard import (
    build_csv_rows,
    build_tensorboard_command,
    build_json_payload,
    discover_run_dirs,
    filter_runs,
    launch_tensorboard,
    load_runs,
    render_tag_list,
    render_text_summary,
)


def _write_sample_event_log(run_dir: Path) -> None:
    writer = SummaryWriter(log_dir=str(run_dir))
    writer.add_scalar("train/loss", 3.0, 0)
    writer.add_scalar("train/loss", 2.0, 1)
    writer.add_scalar("train/loss", 1.5, 2)
    writer.add_scalar("eval/loss", 2.5, 1)
    writer.close()


def _write_text_event_log(run_dir: Path) -> None:
    writer = SummaryWriter(log_dir=str(run_dir))
    writer.add_scalar("train/loss", 1.0, 0)
    writer.add_text("hparams/config", '{"output_dir": "./models/run"}', 0)
    writer.close()


def test_discover_and_summarize_tensorboard_runs(tmp_path):
    run_dir = tmp_path / "runs" / "exp1"
    run_dir.mkdir(parents=True)
    _write_sample_event_log(run_dir)

    discovered = discover_run_dirs(tmp_path)
    assert discovered == [run_dir]

    runs = load_runs(discovered)
    assert set(runs[run_dir]) == {"train/loss", "eval/loss"}

    filtered = filter_runs(runs, r"train/.*")
    text = render_text_summary(filtered, tmp_path)
    assert "train/loss" in text
    assert "final" in text
    assert "1.500000" in text


def test_exports_include_raw_points(tmp_path):
    run_dir = tmp_path / "logs" / "run2"
    run_dir.mkdir(parents=True)
    _write_sample_event_log(run_dir)

    runs = load_runs(discover_run_dirs(tmp_path))
    json_payload = build_json_payload(runs, tmp_path)
    csv_rows = build_csv_rows(runs, tmp_path)
    tag_list = render_tag_list(runs, tmp_path)

    assert json_payload[0]["run"] == "logs/run2"
    assert json_payload[0]["metrics"]["train/loss"]["summary"]["count"] == 3
    assert len(csv_rows) == 4
    assert "eval/loss" in tag_list


def test_ignores_non_numeric_tensor_summaries(tmp_path):
    run_dir = tmp_path / "logs" / "run3"
    run_dir.mkdir(parents=True)
    _write_text_event_log(run_dir)

    runs = load_runs(discover_run_dirs(tmp_path))

    assert "train/loss" in runs[run_dir]
    assert all(not tag.startswith("hparams/") for tag in runs[run_dir])


def test_build_tensorboard_command_uses_localhost(tmp_path):
    command = build_tensorboard_command(tmp_path, "127.0.0.1", 6006)
    assert command[:3] == [command[0], "-m", "tensorboard.main"]
    assert "--logdir" in command
    assert str(tmp_path) in command
    assert "--host" in command
    assert "127.0.0.1" in command


def test_launch_tensorboard_waits_for_foreground_process(tmp_path):
    process = Mock()
    command = build_tensorboard_command(tmp_path, "127.0.0.1", 6006)
    process.wait.side_effect = [subprocess.TimeoutExpired(cmd="tensorboard", timeout=2), 0]
    with patch("analyze_tensorboard.subprocess.Popen", return_value=process) as popen:
        exit_code = launch_tensorboard(tmp_path, "127.0.0.1", 6006, open_browser=False)

    assert exit_code == 0
    popen.assert_called_once_with(command)
    process.wait.assert_any_call(timeout=2)
    assert process.wait.call_args_list[-1].kwargs == {}


def test_launch_tensorboard_detach_keeps_background_behavior(tmp_path):
    process = Mock()
    with patch("analyze_tensorboard.subprocess.Popen", return_value=process) as popen:
        exit_code = launch_tensorboard(
            tmp_path,
            "127.0.0.1",
            6006,
            open_browser=False,
            detach=True,
        )

    assert exit_code == 0
    popen.assert_called_once_with(
        build_tensorboard_command(tmp_path, "127.0.0.1", 6006),
        start_new_session=True,
    )
