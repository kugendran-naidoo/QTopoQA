from __future__ import annotations

from importlib import import_module
from pathlib import Path

cif = import_module(
    "qtdaqa.new_dynamic_features.graph_builder2.helpers.compare_interface_features.compare_interface_features"
)
cnf = import_module(
    "qtdaqa.new_dynamic_features.graph_builder2.helpers.compare_node_features.compare_node_features"
)
ctf = import_module(
    "qtdaqa.new_dynamic_features.graph_builder2.helpers.compare_topo_features.compare_topo_features"
)


def test_compare_interface_features_reports_identical(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline" / "sample.interface.txt"
    candidate = tmp_path / "candidate" / "sample.interface.txt"
    baseline.parent.mkdir(parents=True, exist_ok=True)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    content = "A 1.0 2.0 3.0\nB 4.0 5.0 6.0\n"
    baseline.write_text(content, encoding="utf-8")
    candidate.write_text(content, encoding="utf-8")

    same, diffs = cif.compare_files(
        baseline,
        candidate,
        tolerance=0.0,
        rel_tolerance=0.0,
    )

    assert same
    assert diffs == []


def test_compare_node_features_detects_value_mismatch(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline" / "nodes.csv"
    candidate = tmp_path / "candidate" / "nodes.csv"
    baseline.parent.mkdir(parents=True, exist_ok=True)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text("ID,f1,f2\nA,1.0,2.0\n", encoding="utf-8")
    candidate.write_text("ID,f1,f2\nA,1.0,3.0\n", encoding="utf-8")

    same, diffs = cnf.compare_files(
        baseline,
        candidate,
        abs_tol=0.0,
        rel_tol=0.0,
    )

    assert not same
    assert any("value[1]" in entry for entry in diffs)


def test_compare_topo_features_flags_numeric_difference(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / "sample.topology.csv").write_text(
        "ID,feat\nA,1.0\n",
        encoding="utf-8",
    )
    (candidate_dir / "sample.topology.csv").write_text(
        "ID,feat\nA,1.5\n",
        encoding="utf-8",
    )

    results, missing_ref, missing_cand = ctf.compare_directories(
        baseline_dir,
        candidate_dir,
        tolerance=0.0,
    )

    assert len(results) == 1
    assert not missing_ref
    assert not missing_cand
    assert results[0].exceeded
