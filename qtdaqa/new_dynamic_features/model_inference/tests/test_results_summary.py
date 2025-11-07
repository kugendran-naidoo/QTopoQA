import sys
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qtdaqa.new_dynamic_features.model_inference import results_summary  # noqa: E402


def _make_summary(tmp_path: Path, target: str, rows: list[str]) -> None:
    target_dir = tmp_path / target
    target_dir.mkdir(parents=True)
    (target_dir / f"{target}.summary_metrics.csv").write_text("\n".join(rows), encoding="utf-8")


def test_generate_dockq_and_hit_rate_summaries(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _make_summary(
        results_dir,
        "3SE8",
        [
            "header",
            "3SE8,1,2,3,0.25,4,5,6",
            "3SE8,1,2,3,0.1049,7,8,9",
        ],
    )
    _make_summary(results_dir, "5ABC", ["# note", "5ABC,x,y,z,0.5,1,2,3"])

    dockq_path = results_summary.generate_dockq_summary(results_dir)
    dockq_lines = dockq_path.read_text(encoding="utf-8").strip().splitlines()
    assert dockq_lines[0] == "target,dockq"
    assert "3SE8,0.105" in dockq_lines[1:]
    assert "5ABC,0.500" in dockq_lines[1:]

    hit_rate_path = results_summary.generate_hit_rate_summary(results_dir)
    hit_lines = hit_rate_path.read_text(encoding="utf-8").strip().splitlines()
    assert hit_lines[0] == "target,hit_rate"
    assert "3SE8,7/8/9" in hit_lines[1:]
    assert "5ABC,1/2/3" in hit_lines[1:]


def test_generate_summary_no_data(tmp_path: Path) -> None:
    results_dir = tmp_path / "empty"
    results_dir.mkdir()
    with pytest.raises(RuntimeError):
        results_summary.generate_dockq_summary(results_dir)
