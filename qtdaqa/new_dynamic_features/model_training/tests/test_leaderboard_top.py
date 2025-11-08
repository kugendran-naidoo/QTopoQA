import argparse
from pathlib import Path

from qtdaqa.new_dynamic_features.model_training import train_cli


def _make_run(tmp_path: Path, name: str, summary: dict, registry: dict) -> None:
    run_dir = tmp_path / name
    run_dir.mkdir()
    (run_dir / "run_metadata.json").write_text("{}", encoding="utf-8")
    payload = {"run_name": name}
    payload.update(summary)
    registry[name] = payload


def test_rank_runs_respects_primary_metric(tmp_path, monkeypatch):
    summaries = {}
    _make_run(
        tmp_path,
        "run_a",
        {
            "selection_primary_metric": "selection_metric",
            "best_selection_metric": -0.72,
            "best_val_loss": 0.052,
        },
        summaries,
    )
    _make_run(
        tmp_path,
        "run_b",
        {
            "selection_primary_metric": "val_loss",
            "best_selection_metric": -0.50,
            "best_val_loss": 0.041,
        },
        summaries,
    )
    _make_run(
        tmp_path,
        "run_c",
        {
            "selection_primary_metric": "selection_metric",
            "best_selection_metric": None,
            "best_val_loss": 0.070,
        },
        summaries,
    )
    _make_run(
        tmp_path,
        "run_missing",
        {
            "selection_primary_metric": "selection_metric",
            "best_selection_metric": None,
            "best_val_loss": None,
        },
        summaries,
    )

    def fake_summarise(run_dir: Path):
        return dict(summaries[run_dir.name])

    monkeypatch.setattr(train_cli, "_summarise_run", fake_summarise)

    ranked = train_cli.rank_runs(tmp_path)
    ranked_names = [entry[2]["run_name"] for entry in ranked]
    assert ranked_names == ["run_a", "run_b", "run_c"]
    # run_a chooses selection_metric, run_b uses val_loss, run_c falls back to val_loss
    assert [entry[0] for entry in ranked] == [
        "selection_metric",
        "val_loss",
        "val_loss",
    ]


def test_cmd_leaderboard_honours_top_count(tmp_path, monkeypatch, capsys):
    entries = [
        (
            "selection_metric",
            -0.70,
            {
                "run_name": "primary",
                "best_selection_metric": -0.70,
                "best_val_loss": 0.050,
                "best_checkpoint": tmp_path / "primary.ckpt",
                "selection_metric_enabled": True,
                "best_selection_val_spearman": -0.5,
            },
        ),
        (
            "val_loss",
            0.045,
            {
                "run_name": "secondary",
                "best_selection_metric": -0.60,
                "best_val_loss": 0.045,
                "best_checkpoint": tmp_path / "secondary.ckpt",
                "selection_metric_enabled": False,
            },
        ),
        (
            "val_loss",
            0.060,
            {
                "run_name": "tertiary",
                "best_selection_metric": -0.55,
                "best_val_loss": 0.060,
                "best_checkpoint": tmp_path / "tertiary.ckpt",
                "selection_metric_enabled": False,
            },
        ),
    ]
    monkeypatch.setattr(train_cli, "rank_runs", lambda root: entries)

    args = argparse.Namespace(root=tmp_path, top=2, limit=5)
    train_cli.cmd_leaderboard(args)
    output = capsys.readouterr().out
    assert "1. primary" in output
    assert "2. secondary" in output
    assert "tertiary" not in output
    assert "primary_metric: selection_metric = -0.7" in output
    assert "secondary_metric: val_spearman_corr = -0.5" in output
    assert "secondary_metric: None" in output
