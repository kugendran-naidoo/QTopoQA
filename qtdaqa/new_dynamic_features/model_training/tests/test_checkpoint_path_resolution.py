from pathlib import Path

from qtdaqa.new_dynamic_features.model_training.run_metadata import (
    resolve_checkpoint_path,
)


def test_resolve_existing_checkpoint_returns_existing_path(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoint.val-0.01_epoch001.chkpt"
    ckpt.write_text("ok", encoding="utf-8")
    resolved = resolve_checkpoint_path(str(ckpt))
    assert resolved == ckpt.resolve()


def test_resolve_existing_checkpoint_falls_back_to_chkpt(tmp_path: Path) -> None:
    legacy = tmp_path / "checkpoint.val-0.02_epoch002.ckpt"
    legacy.write_text("legacy", encoding="utf-8")
    target = legacy.with_suffix(".chkpt")
    legacy.rename(target)

    resolved = resolve_checkpoint_path(str(legacy))
    assert resolved == target.resolve()


def test_resolve_existing_checkpoint_handles_missing() -> None:
    missing = "/tmp/nonexistent/checkpoint.val-0.03_epoch003.ckpt"
    resolved = resolve_checkpoint_path(missing)
    assert resolved == Path(missing)
