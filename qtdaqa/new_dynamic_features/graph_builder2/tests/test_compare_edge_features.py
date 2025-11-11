from __future__ import annotations

import csv
from pathlib import Path

import pytest

from qtdaqa.new_dynamic_features.graph_builder2.helpers.compare_edge_features import (
    compare_edge_features as cef,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_compare_directories_detects_matches_and_differences(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_csv(
        baseline / "edges.csv",
        [
            {"src_idx": 0, "dst_idx": 1, "distance": 3.5, "band": "short"},
            {"src_idx": 1, "dst_idx": 0, "distance": 3.5, "band": "short"},
        ],
    )
    _write_csv(
        candidate / "edges.csv",
        [
            {"src_idx": 0, "dst_idx": 1, "distance": 3.6, "band": "short"},
            {"src_idx": 1, "dst_idx": 0, "distance": 3.5, "band": "short"},
        ],
    )
    _write_csv(
        baseline / "extra.csv",
        [{"src_idx": 2, "dst_idx": 3, "distance": 7.2, "band": "long"}],
    )
    _write_csv(
        candidate / "other.csv",
        [{"src_idx": 4, "dst_idx": 5, "distance": 1.0, "band": "short"}],
    )

    results, missing_ref, missing_cand = cef.compare_directories(baseline, candidate, tolerance=0.0)

    assert len(results) == 1
    assert missing_ref == [baseline / "extra.csv"]
    assert missing_cand == [candidate / "other.csv"]

    result = results[0]
    assert result.relative_path == Path("edges.csv")
    assert pytest.approx(result.max_abs_diff, rel=0, abs=1e-9) == 0.1
    assert result.exceeded and result.exceeded[0][1] == "distance"


def test_main_writes_reports(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_csv(
        baseline / "edges.csv",
        [
            {"src_idx": 0, "dst_idx": 1, "distance": 3.5},
        ],
    )
    _write_csv(
        candidate / "edges.csv",
        [
            {"src_idx": 0, "dst_idx": 1, "distance": 3.5},
        ],
    )

    diff_report = tmp_path / "diff.txt"
    same_report = tmp_path / "same.txt"
    missing_report = tmp_path / "missing.txt"
    run_report = tmp_path / "run.log"

    rc = cef.main(
        [
            str(baseline),
            str(candidate),
            "--diff-report",
            str(diff_report),
            "--same-report",
            str(same_report),
            "--missing-report",
            str(missing_report),
            "--run-report",
            str(run_report),
        ]
    )
    assert rc == 0
    assert "edges.csv" in same_report.read_text()
    assert diff_report.read_text() == ""
    assert missing_report.read_text() == ""
    assert run_report.read_text().startswith("=== compare_edge_features run ===")
