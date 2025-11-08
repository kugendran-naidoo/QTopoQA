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
                "selection_metric_enabled": True,
                "best_selection_val_spearman": -0.65,
                "selection_primary_metric": "selection_metric",
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
                "selection_metric_enabled": False,
                "selection_primary_metric": "val_loss",
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
    assert "primary_metric: selection_metric = -0.71" in output
    assert "secondary_metric: val_spearman_corr = -0.65" in output
    assert "secondary_metric: None" in output


def test_monitor_top_rejects_conflicting_flags(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        monitor_best_model.main(["--top", "1", "--follow", "--root", str(tmp_path)])
    assert excinfo.value.code == 2

    with pytest.raises(SystemExit) as excinfo:
        monitor_best_model.main(["--top", "1", "--run-id", "foo", "--root", str(tmp_path)])
    assert excinfo.value.code == 2


def test_render_table_includes_metric_block(tmp_path, capsys):
    summary = {
        "run_name": "alpha",
        "run_dir": "training_runs/alpha",
        "best_summary_line": None,
        "best_checkpoint_name": "checkpoint.ckpt",
        "best_checkpoint_path": "training_runs/alpha/checkpoint.ckpt",
        "selection_primary_metric": "val_loss",
        "best_val_loss": 0.05,
        "best_selection_metric": -0.6,
        "selection_metric_enabled": True,
        "best_selection_val_spearman": -0.62,
        "warnings": [],
        "checkpoint_symlinks": [],
        "learning_parameters": {},
    }
    monitor_best_model._render_table(summary, metrics_limit=None)
    output = capsys.readouterr().out
    assert "primary_metric: val_loss = 0.05" in output
    assert "secondary_metric: val_spearman_corr = -0.62" in output
    assert "selection_metric: -0.6" in output
    assert "checkpoint: training_runs/alpha/checkpoint.ckpt" in output
