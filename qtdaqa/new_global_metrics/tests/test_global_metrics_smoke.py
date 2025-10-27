from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pytest

from qtdaqa.new_global_metrics import global_metrics


DATASET_ROOT = Path(
    "/Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/datasets/training/adjusted/pilot_batch_Dockground_MAF2"
)


def _copy_sample_pdbs(destination: Path, limit: int = 2) -> None:
    copied = 0
    nested_dir = destination / "nested"
    for pdb_path in sorted(DATASET_ROOT.rglob("*.pdb")):
        target_dir = destination if copied == 0 else nested_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdb_path, target_dir / pdb_path.name)
        copied += 1
        if copied >= limit:
            break
    if copied == 0:
        raise RuntimeError("No PDB files were copied into the sample dataset")


def test_interface_contact_count_smoke(tmp_path: Path) -> None:
    if not DATASET_ROOT.exists():
        pytest.skip(f"Sample dataset missing: {DATASET_ROOT}")

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    _copy_sample_pdbs(dataset_dir, limit=2)

    work_dir = tmp_path / "work"
    graph_dir = tmp_path / "graph"
    log_dir = tmp_path / "logs"

    exit_code = global_metrics.main(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--work-dir",
            str(work_dir),
            "--graph-dir",
            str(graph_dir),
            "--log-dir",
            str(log_dir),
            "--output-csv",
            "metrics.csv",
        ]
    )
    assert exit_code == 0

    output_csv = work_dir / "metrics.csv"
    assert output_csv.is_file()

    with output_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert rows, "Expected at least one metrics row"
    for row in rows:
        assert "MODEL" in row
        value = float(row["interface_contact_count"])
        assert value >= 0.0
