import logging
import sys
import types
from pathlib import Path

import pytest

# Minimal torch stub to avoid loading the real library during unit tests.
torch_stub = types.ModuleType("torch")
torch_stub.load = lambda *_args, **_kwargs: {}
sys.modules.setdefault("torch", torch_stub)

from qtdaqa.new_dynamic_features.model_training.common import resume_guard as guard


def _mock_ckpt(path: Path, epoch: int, monkeypatch: pytest.MonkeyPatch) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")

    def _fake_load(_p, map_location=None):
        return {"epoch": epoch}

    monkeypatch.setattr(guard.torch, "load", _fake_load)
    return path


def test_resume_checkpoint_rejects_over_max(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ckpt = _mock_ckpt(tmp_path / "model_checkpoints" / "ckpt_over.ckpt", epoch=5, monkeypatch=monkeypatch)
    with pytest.raises(RuntimeError) as exc:
        guard.validate_resume_checkpoint(
            tmp_path,
            tmp_path / "model_checkpoints",
            str(ckpt),
            max_epochs=3,
            logger=logging.getLogger("test"),
        )
    assert "Refusing to resume" in str(exc.value)


def test_dirty_run_dir_without_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_ckpt(tmp_path / "model_checkpoints" / "ckpt_exist.ckpt", epoch=2, monkeypatch=monkeypatch)
    with pytest.raises(RuntimeError) as exc:
        guard.validate_resume_checkpoint(
            tmp_path,
            tmp_path / "model_checkpoints",
            None,
            max_epochs=5,
            logger=logging.getLogger("test"),
        )
    msg = str(exc.value)
    assert "already contains checkpoints" in msg
    assert "no --resume-from was provided" in msg


def test_resume_checkpoint_accepts_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ckpt = _mock_ckpt(tmp_path / "model_checkpoints" / "ckpt_ok.ckpt", epoch=2, monkeypatch=monkeypatch)
    resolved = guard.validate_resume_checkpoint(
        tmp_path,
        tmp_path / "model_checkpoints",
        str(ckpt),
        max_epochs=5,
        logger=logging.getLogger("test"),
    )
    assert resolved == str(ckpt.resolve())
