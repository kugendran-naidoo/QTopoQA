import pytest

from qtdaqa.new_dynamic_features.model_training import monitor_best_model, train_cli


def test_monitor_top_lists_runs(tmp_path, monkeypatch, capsys):
    entries = [
        (
            "selection_metric",
            -0.71,
            {
                "run_name": "alpha",
                "best_selection_metric": -0.71,
                "best_val_loss": 0.051,
                "best_checkpoint": str(tmp_path / "alpha.ckpt"),
            },
        ),
        (
            "val_loss",
            0.048,
            {
                "run_name": "beta",
                "best_selection_metric": -0.60,
                "best_val_loss": 0.048,
                "best_checkpoint": str(tmp_path / "beta.ckpt"),
            },
        ),
    ]
    monkeypatch.setattr(train_cli, "rank_runs", lambda root: entries)

    exit_code = monitor_best_model.main(["--top", "2", "--root", str(tmp_path)])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Top 2 runs" in output
    assert "1. alpha" in output
    assert "2. beta" in output
    assert "selection_metric=-0.71" in output


def test_monitor_top_rejects_conflicting_flags(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        monitor_best_model.main(["--top", "1", "--follow", "--root", str(tmp_path)])
    assert excinfo.value.code == 2

    with pytest.raises(SystemExit) as excinfo:
        monitor_best_model.main(["--top", "1", "--run-id", "foo", "--root", str(tmp_path)])
    assert excinfo.value.code == 2
