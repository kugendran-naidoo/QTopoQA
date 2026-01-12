from pathlib import Path

from qtdaqa.new_dynamic_features.graph_builder2 import graph_builder2 as gb2


def test_validation_default_on_writes_manifest(monkeypatch, tmp_path):
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = graph_dir / "graph_manifest.json"

    def fake_validate(
        graph_dir_arg,
        manifest_arg,
        create_manifest,
        sample,
        ignore_metadata,
        *,
        workers=1,
        progress_interval=15.0,
    ):
        manifest_arg.write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.setattr(gb2, "validate_graphs", fake_validate)
    result = gb2._maybe_run_graph_validation(
        enabled=True,
        graph_dir=graph_dir,
        workers=0,
        progress_interval=0.0,
    )
    assert manifest_path.exists()
    assert result.get("exit_code") == 0


def test_validation_opt_out_skips(monkeypatch, tmp_path):
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = graph_dir / "graph_manifest.json"

    def fake_validate(
        graph_dir_arg,
        manifest_arg,
        create_manifest,
        sample,
        ignore_metadata,
        *,
        workers=1,
        progress_interval=15.0,
    ):
        manifest_arg.write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.setattr(gb2, "validate_graphs", fake_validate)
    result = gb2._maybe_run_graph_validation(
        enabled=False,
        graph_dir=graph_dir,
        workers=0,
        progress_interval=0.0,
    )
    assert not manifest_path.exists()
    assert result.get("enabled") is False
