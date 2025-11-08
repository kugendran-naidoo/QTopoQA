from __future__ import annotations

import json
from pathlib import Path

from qtdaqa.new_dynamic_features.model_training.run_metadata import record_selection_metadata


def test_record_selection_metadata_updates_run_metadata(tmp_path: Path) -> None:
    save_dir = tmp_path / "run"
    save_dir.mkdir()
    metadata_path = save_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps({"previous": "keep"}), encoding="utf-8")

    record_selection_metadata(
        save_dir,
        primary_metric="selection_metric",
        use_val_spearman=False,
        spearman_weight=0.5,
        spearman_min_delta=0.01,
    )

    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert data["previous"] == "keep"
    assert data["selection_primary_metric"] == "selection_metric"
    selection_block = data["selection"]
    assert selection_block["primary_metric"] == "selection_metric"
    assert selection_block["use_val_spearman"] is False
    assert selection_block["spearman_weight"] == 0.5
    assert selection_block["spearman_min_delta"] == 0.01
