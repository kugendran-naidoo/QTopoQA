from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pytest

from qtdaqa.new_global_metrics import global_metrics

try:  # pragma: no cover - optional dependency checks
    import MDAnalysis  # noqa: F401
    from MDAnalysis.analysis import sasa  # noqa: F401
except ImportError:  # pragma: no cover
    HAS_MDANALYSIS = False
else:  # pragma: no cover
    HAS_MDANALYSIS = True


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

    base_args = [
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
    if not HAS_MDANALYSIS:
        base_args.append("--no-buried-sasa")

    exit_code = global_metrics.main(base_args)
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
        residue_value = float(row["interface_residue_count"])
        assert residue_value >= 0.0
        centroid_median = float(row["interface_centroid_distance_median"])
        centroid_softmin = float(row["interface_centroid_distance_softmin"])
        centroid_weighted = float(row["interface_centroid_distance_weighted_mean"])
        assert centroid_median >= 0.0
        assert centroid_softmin >= 0.0
        assert centroid_weighted >= 0.0
        rg_mean = float(row["chain_radius_of_gyration_mean"])
        rg_median = float(row["chain_radius_of_gyration_median"])
        rg_std = float(row["chain_radius_of_gyration_std"])
        assert rg_mean >= 0.0
        assert rg_median >= 0.0
        assert rg_std >= 0.0
        seq_largest = float(row["sequence_largest_fraction"])
        seq_top2 = float(row["sequence_top2_fraction"])
        seq_log = float(row["sequence_log_L1_L2"])
        seq_ratio = float(row["sequence_max_min_ratio_clipped"])
        seq_cv = float(row["sequence_cv"])
        seq_med = float(row["sequence_median_length"])
        seq_mean = float(row["sequence_mean_length"])
        seq_p75 = float(row["sequence_p75_length"])
        seq_long = float(row["sequence_n_long_ge_1p5med"])
        seq_short = float(row["sequence_n_short_le_0p5med"])
        seq_count = float(row["sequence_chain_count"])
        seq_total = float(row["sequence_total_length"])
        assert 0.0 <= seq_largest <= 1.0
        assert 0.0 <= seq_top2 <= 1.0
        assert seq_ratio >= 0.0
        assert seq_cv >= 0.0
        assert seq_med >= 0.0
        assert seq_mean >= 0.0
        assert seq_p75 >= 0.0
        assert seq_long >= 0.0
        assert seq_short >= 0.0
        assert seq_count >= 0.0
        assert seq_total >= 0.0
        if HAS_MDANALYSIS:
            sasa_value = float(row["interface_buried_sasa"])
            assert sasa_value >= 0.0
        else:
            assert "interface_buried_sasa" not in row

    # Re-run skipping residue count to ensure CLI toggle works.
    contacts_args = [
        "--dataset-dir",
        str(dataset_dir),
        "--work-dir",
        str(work_dir),
        "--graph-dir",
        str(graph_dir),
        "--log-dir",
        str(log_dir),
        "--output-csv",
        "metrics_contacts_only.csv",
        "--no-interface-residue-count",
    ]
    if not HAS_MDANALYSIS:
        contacts_args.append("--no-buried-sasa")

    exit_code = global_metrics.main(contacts_args)
    assert exit_code == 0

    contacts_only_csv = work_dir / "metrics_contacts_only.csv"
    assert contacts_only_csv.is_file()

    with contacts_only_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert rows, "Expected contact-only metrics"
    assert "interface_residue_count" not in reader.fieldnames
    if HAS_MDANALYSIS:
        assert "interface_buried_sasa" in reader.fieldnames
    assert "interface_centroid_distance_median" in reader.fieldnames
    assert "interface_centroid_distance_softmin" in reader.fieldnames
    assert "interface_centroid_distance_weighted_mean" in reader.fieldnames
    assert "chain_radius_of_gyration_mean" in reader.fieldnames
    assert "chain_radius_of_gyration_median" in reader.fieldnames
    assert "chain_radius_of_gyration_std" in reader.fieldnames
    assert "sequence_largest_fraction" in reader.fieldnames
    assert "sequence_top2_fraction" in reader.fieldnames
    assert "sequence_log_L1_L2" in reader.fieldnames
    assert "sequence_max_min_ratio_clipped" in reader.fieldnames
    assert "sequence_cv" in reader.fieldnames
    assert "sequence_median_length" in reader.fieldnames
    assert "sequence_mean_length" in reader.fieldnames
    assert "sequence_p75_length" in reader.fieldnames
    assert "sequence_n_long_ge_1p5med" in reader.fieldnames
    assert "sequence_n_short_le_0p5med" in reader.fieldnames
    assert "sequence_chain_count" in reader.fieldnames
    assert "sequence_total_length" in reader.fieldnames

    if HAS_MDANALYSIS:
        # Re-run skipping buried SASA to ensure CLI toggle works.
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
                "metrics_no_sasa.csv",
                "--no-buried-sasa",
            ]
        )
        assert exit_code == 0

        no_sasa_csv = work_dir / "metrics_no_sasa.csv"
        assert no_sasa_csv.is_file()

        with no_sasa_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
        rows = list(reader)

        assert rows, "Expected metrics rows when SASA disabled"
        assert "interface_buried_sasa" not in reader.fieldnames

    # Re-run skipping centroid distance to ensure CLI toggle works.
    centroid_skip_args = [
        "--dataset-dir",
        str(dataset_dir),
        "--work-dir",
        str(work_dir),
        "--graph-dir",
        str(graph_dir),
        "--log-dir",
        str(log_dir),
        "--output-csv",
        "metrics_no_centroid.csv",
        "--no-interface-centroid-distance",
    ]
    if not HAS_MDANALYSIS:
        centroid_skip_args.append("--no-buried-sasa")

    exit_code = global_metrics.main(centroid_skip_args)
    assert exit_code == 0

    no_centroid_csv = work_dir / "metrics_no_centroid.csv"
    assert no_centroid_csv.is_file()

    with no_centroid_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert rows, "Expected metrics rows when centroid distance disabled"
    assert "interface_centroid_distance_median" not in reader.fieldnames
    assert "interface_centroid_distance_softmin" not in reader.fieldnames
    assert "interface_centroid_distance_weighted_mean" not in reader.fieldnames

    # Re-run skipping radius of gyration to ensure CLI toggle works.
    rg_skip_args = [
        "--dataset-dir",
        str(dataset_dir),
        "--work-dir",
        str(work_dir),
        "--graph-dir",
        str(graph_dir),
        "--log-dir",
        str(log_dir),
        "--output-csv",
        "metrics_no_rg.csv",
        "--no-radius-gyration",
    ]
    if not HAS_MDANALYSIS:
        rg_skip_args.append("--no-buried-sasa")

    exit_code = global_metrics.main(rg_skip_args)
    assert exit_code == 0

    no_rg_csv = work_dir / "metrics_no_rg.csv"
    assert no_rg_csv.is_file()

    with no_rg_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert rows, "Expected metrics rows when radius of gyration disabled"
    assert "chain_radius_of_gyration_mean" not in reader.fieldnames
    assert "chain_radius_of_gyration_median" not in reader.fieldnames
    assert "chain_radius_of_gyration_std" not in reader.fieldnames

    # Re-run skipping sequence ratio to ensure CLI toggle works.
    seq_skip_args = [
        "--dataset-dir",
        str(dataset_dir),
        "--work-dir",
        str(work_dir),
        "--graph-dir",
        str(graph_dir),
        "--log-dir",
        str(log_dir),
        "--output-csv",
        "metrics_no_seq.csv",
        "--no-sequence-ratio",
    ]
    if not HAS_MDANALYSIS:
        seq_skip_args.append("--no-buried-sasa")

    exit_code = global_metrics.main(seq_skip_args)
    assert exit_code == 0

    no_seq_csv = work_dir / "metrics_no_seq.csv"
    assert no_seq_csv.is_file()

    with no_seq_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert rows, "Expected metrics rows when sequence ratio disabled"
    for column in [
        "sequence_largest_fraction",
        "sequence_top2_fraction",
        "sequence_log_L1_L2",
        "sequence_max_min_ratio_clipped",
        "sequence_cv",
        "sequence_median_length",
        "sequence_mean_length",
        "sequence_p75_length",
        "sequence_n_long_ge_1p5med",
        "sequence_n_short_le_0p5med",
        "sequence_chain_count",
        "sequence_total_length",
    ]:
        assert column not in reader.fieldnames
