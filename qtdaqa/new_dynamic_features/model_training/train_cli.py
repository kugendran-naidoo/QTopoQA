#!/usr/bin/env python3
"""
Unified CLI for dynamic TopoQA training workflows.

Commands
--------
run         Launch a single training job from a YAML config file.
batch       Execute multiple jobs described by a YAML manifest.
resume      Continue a paused training run from its run directory/ID.
summarise   Report metrics and artifacts produced by a completed run.

All commands default to the existing training implementation
(`model_train_topoqa_cpu.py`) and preserve deterministic environment
settings that were previously managed by bash wrappers.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_ROOT = SCRIPT_DIR / "training_runs"
TRAINING_SCRIPT = SCRIPT_DIR / "model_train_topoqa_cpu.py"
REPO_ROOT = SCRIPT_DIR.parents[2]

DEFAULT_ENV = {
    "PYTHONHASHSEED": "222",
    "PL_SEED_WORKERS": "1",
    "TORCH_USE_DETERMINISTIC_ALGORITHMS": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":16:8",
}

TIMESTAMP_FMT = "%Y-%m-%d_%H-%M-%S"
RUN_PREFIX = "training_run"


class CLIError(RuntimeError):
    """Raised when a user-facing error occurs."""


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise CLIError(f"YAML root must be a mapping (file: {path})")
    return data


def _dump_yaml(data: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _timestamp() -> str:
    return _dt.datetime.now().strftime(TIMESTAMP_FMT)


def _deep_set(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = config
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _parse_override(raw: str) -> Tuple[str, Any]:
    if "=" not in raw:
        raise CLIError(f"Overrides must use key=value syntax (received '{raw}')")
    key, raw_value = raw.split("=", 1)
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise CLIError(f"Unable to parse override value '{raw_value}': {exc}") from exc
    return key.strip(), value


def _apply_overrides(config: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    if not overrides:
        return config
    updated = json.loads(json.dumps(config))  # deep copy via JSON
    applied: Dict[str, Any] = {}
    for raw in overrides:
        key, value = _parse_override(raw)
        _deep_set(updated, key, value)
        applied[key] = value
    return updated


def _ensure_run_root(root: Path) -> Tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    history_dir = root / "history"
    history_dir.mkdir(exist_ok=True)
    latest_link = root / "latest"
    return history_dir, latest_link


def _prepare_run_dir(root: Path, run_name: Optional[str]) -> Path:
    _history_dir, latest_link = _ensure_run_root(root)
    if run_name is None:
        run_name = f"{RUN_PREFIX}_{_timestamp()}"
        counter = 1
        candidate = root / run_name
        while candidate.exists():
            run_name = f"{RUN_PREFIX}_{_timestamp()}_{counter}"
            candidate = root / run_name
            counter += 1
    run_dir = root / run_name
    if run_dir.exists():
        raise CLIError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    for subdir in ("config", "logs", "metrics", "artifacts"):
        (run_dir / subdir).mkdir(exist_ok=True)
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(run_dir)
    return run_dir


def _resolve_git_info(repo_root: Path) -> Tuple[str, bool]:
    git = shutil.which("git")
    if not git:
        return "unknown", False
    try:
        commit = subprocess.check_output(
            [git, "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        commit = "unknown"
    try:
        status = subprocess.check_output(
            [git, "-C", str(repo_root), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        dirty = bool(status.strip())
    except subprocess.CalledProcessError:
        dirty = False
    return commit or "unknown", dirty


def _write_run_metadata(
    run_dir: Path,
    *,
    run_name: str,
    source_config: Path,
    overrides: Dict[str, Any],
    notes: Optional[str],
    trial_label: Optional[str],
    command: Sequence[str],
    env: Dict[str, str],
) -> None:
    metadata = {
        "run_name": run_name,
        "created": _dt.datetime.now().isoformat(),
        "source_config": str(source_config),
        "overrides": overrides,
        "notes": notes,
        "trial_label": trial_label,
        "command": list(command),
        "environment": {key: env[key] for key in sorted(env)},
    }
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if notes:
        (run_dir / "notes.txt").write_text(notes, encoding="utf-8")


def _stream_subprocess(cmd: Sequence[str], log_path: Path, cwd: Path, env: Dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
        return process.wait()


def _post_process_run(run_dir: Path) -> None:
    training_log = run_dir / "training.log"
    console_log = run_dir / "training_console.log"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    if training_log.exists():
        target = logs_dir / "training.log"
        if not target.exists():
            training_log.rename(target)
        else:
            target = logs_dir / f"training.log.{_timestamp()}"
            training_log.rename(target)
        symlink = run_dir / "training.log"
        if symlink.exists():
            symlink.unlink()
        symlink.symlink_to(target)
    if console_log.exists():
        target = logs_dir / console_log.name
        if not target.exists():
            console_log.rename(target)
        else:
            target = logs_dir / f"{console_log.stem}.{_timestamp()}.log"
            console_log.rename(target)
        console_symlink = run_dir / console_log.name
        if console_symlink.exists():
            console_symlink.unlink()
        console_symlink.symlink_to(target)

    metrics_root = run_dir / "metrics"
    metrics_root.mkdir(exist_ok=True)
    for metrics_csv in run_dir.glob("cpu_training/**/metrics.csv"):
        target = metrics_root / f"{metrics_csv.parent.parent.name}_{metrics_csv.parent.name}.csv"
        shutil.copy2(metrics_csv, target)


def _build_training_command(
    config_path: Path,
    run_dir: Path,
    *,
    git_commit: str,
    git_dirty: bool,
    trial_label: Optional[str],
    resume_from: Optional[Path],
    fast_dev_run: bool,
    limit_train_batches: Optional[float],
    limit_val_batches: Optional[float],
    log_lr: bool,
    extra_args: Optional[Sequence[str]] = None,
) -> List[str]:
    cmd = [
        sys.executable,
        str(TRAINING_SCRIPT),
        "--config",
        str(config_path),
        "--run-dir",
        str(run_dir),
        "--git-commit",
        git_commit,
    ]
    if git_dirty:
        cmd.append("--git-dirty")
    if trial_label:
        cmd.extend(["--trial-label", trial_label])
    if resume_from:
        cmd.extend(["--resume-from", str(resume_from)])
    if fast_dev_run:
        cmd.append("--fast-dev-run")
    if limit_train_batches is not None:
        cmd.extend(["--limit-train-batches", str(limit_train_batches)])
    if limit_val_batches is not None:
        cmd.extend(["--limit-val-batches", str(limit_val_batches)])
    if log_lr:
        cmd.append("--log-lr")
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def _normalise_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.update(DEFAULT_ENV)
    return env


def cmd_run(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    if not config_path.exists():
        raise CLIError(f"Config file not found: {config_path}")

    raw_config = _load_yaml(config_path)
    updated_config = _apply_overrides(raw_config, args.override or [])

    overrides_dict: Dict[str, Any] = {}
    if args.override:
        for raw in args.override:
            key, value = _parse_override(raw)
            overrides_dict[key] = value

    trainer_passthrough: List[str] = []
    handled_trainer_config: Dict[str, Any] = {}
    for raw in args.trainer_arg or []:
        if "=" in raw:
            key, raw_value = raw.split("=", 1)
            normalised = key.strip().replace("-", "_")
            if normalised in {"progress_bar_refresh_rate", "log_every_n_steps"}:
                try:
                    handled_trainer_config[normalised] = yaml.safe_load(raw_value)
                except yaml.YAMLError:
                    handled_trainer_config[normalised] = raw_value
                continue
        trainer_passthrough.append(raw)

    for key, value in handled_trainer_config.items():
        updated_config[key] = value
        overrides_dict[key] = value

    run_root = (args.output_root or RUN_ROOT).resolve()
    run_dir = _prepare_run_dir(run_root, args.run_name)

    config_dir = run_dir / "config"
    config_dir.mkdir(exist_ok=True)

    original_config_path = config_dir / "original_config.yaml"
    shutil.copy2(config_path, original_config_path)
    final_config_path = config_dir / "config.yaml"
    _dump_yaml(updated_config, final_config_path)
    if overrides_dict:
        overrides_path = config_dir / "applied_overrides.yaml"
        _dump_yaml(overrides_dict, overrides_path)

    git_commit, git_dirty = _resolve_git_info(REPO_ROOT)
    env = _normalise_env()

    command = _build_training_command(
        final_config_path,
        run_dir,
        git_commit=git_commit,
        git_dirty=git_dirty,
        trial_label=args.trial_label,
        resume_from=args.resume_from.resolve() if args.resume_from else None,
        fast_dev_run=args.fast_dev_run,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        log_lr=args.log_lr,
        extra_args=trainer_passthrough,
    )

    log_path = run_dir / "training_console.log"
    _write_run_metadata(
        run_dir,
        run_name=run_dir.name,
        source_config=config_path,
        overrides={k: v for k, v in overrides_dict.items()},
        notes=args.notes,
        trial_label=args.trial_label,
        command=command,
        env=env,
    )

    print(f"[train_cli] run_id={run_dir.name} | command={' '.join(command)}")
    ret = _stream_subprocess(command, log_path, cwd=REPO_ROOT, env=env)
    if ret != 0:
        raise CLIError(f"Training command exited with status {ret}")
    _post_process_run(run_dir)
    print(f"[train_cli] completed run {run_dir.name} -> {run_dir}")


def _load_manifest(manifest_path: Path) -> Dict[str, Any]:
    data = _load_yaml(manifest_path)
    if "jobs" not in data or not isinstance(data["jobs"], list):
        raise CLIError("Manifest must contain a top-level 'jobs' list")
    return data


def _merge_dicts(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    result = json.loads(json.dumps(base))
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def cmd_batch(args: argparse.Namespace) -> None:
    manifest_path = args.manifest.resolve()
    if not manifest_path.exists():
        raise CLIError(f"Manifest file not found: {manifest_path}")
    manifest = _load_manifest(manifest_path)
    shared = manifest.get("shared", {})
    shared_overrides = shared.get("overrides") or {}
    shared_notes = shared.get("notes")
    shared_trial = shared.get("trial_label")
    shared_output_root = Path(shared["output_root"]).resolve() if shared.get("output_root") else None
    shared_fast_dev = bool(shared.get("fast_dev_run", False))
    shared_limit_train = shared.get("limit_train_batches")
    shared_limit_val = shared.get("limit_val_batches")
    shared_log_lr = bool(shared.get("log_lr", False))
    shared_trainer_args = list(shared.get("trainer_args", []) or [])

    continue_on_error = bool(manifest.get("continue_on_error", False))

    total = len(manifest["jobs"])
    for index, job in enumerate(manifest["jobs"], start=1):
        job = job or {}
        job_name = job.get("name") or f"job_{index}"
        config_path = job.get("config") or shared.get("config")
        if not config_path:
            raise CLIError(f"Job '{job_name}' is missing 'config' (and no shared config provided)")
        output_root = Path(
            job.get("output_root")
            or (shared_output_root if shared_output_root else args.output_root)
        ).resolve()
        run_name = job.get("run_name") or f"{job_name}_{_timestamp()}"

        overrides = []
        merged_overrides = _merge_dicts(shared_overrides, job.get("overrides") or {})
        for key, value in merged_overrides.items():
            if isinstance(value, (dict, list)):
                serialised = yaml.safe_dump(value, default_flow_style=True).strip()
            else:
                serialised = value
            overrides.append(f"{key}={serialised}")

        trainer_args = shared_trainer_args + list(job.get("trainer_args", []) or [])

        job_args = argparse.Namespace(
            config=Path(config_path),
            output_root=output_root,
            run_name=run_name,
            override=overrides,
            notes=job.get("notes", shared_notes),
            trial_label=job.get("trial_label", shared_trial),
            resume_from=Path(job["resume_from"]).resolve() if job.get("resume_from") else None,
            fast_dev_run=bool(job.get("fast_dev_run", shared_fast_dev)),
            limit_train_batches=job.get("limit_train_batches", shared_limit_train),
            limit_val_batches=job.get("limit_val_batches", shared_limit_val),
            log_lr=bool(job.get("log_lr", shared_log_lr)),
            trainer_arg=trainer_args,
        )

        print(f"[train_cli][batch] ({index}/{total}) running job '{job_name}' -> run_name={job_args.run_name}")
        try:
            cmd_run(job_args)
        except CLIError as exc:
            print(f"[train_cli][batch] job '{job_name}' failed: {exc}", file=sys.stderr)
            if not continue_on_error:
                raise
        print(f"[train_cli][batch] job '{job_name}' finished.")


def _resolve_run_dir_argument(run_id: Optional[str], run_dir: Optional[Path]) -> Path:
    if run_dir and run_id:
        raise CLIError("Provide either --run-id or --run-dir, not both.")
    if run_dir:
        resolved = run_dir.resolve()
    elif run_id:
        resolved = RUN_ROOT / run_id
    else:
        raise CLIError("A run identifier is required (use --run-id or --run-dir).")
    if not resolved.exists():
        raise CLIError(f"Run directory does not exist: {resolved}")
    return resolved


def _select_checkpoint(run_dir: Path, preferred: Optional[str]) -> Path:
    if preferred:
        candidate = Path(preferred)
        if not candidate.is_absolute():
            candidate = run_dir / preferred
        if not candidate.exists():
            raise CLIError(f"Checkpoint not found: {candidate}")
        return candidate
    best_symlink = run_dir / "model_checkpoints" / "best.ckpt"
    if best_symlink.exists():
        return best_symlink.resolve()
    candidates = sorted((run_dir / "model_checkpoints").glob("*.chkpt"), reverse=True)
    if not candidates:
        raise CLIError("No checkpoint files found in run directory.")
    return candidates[0]


def cmd_resume(args: argparse.Namespace) -> None:
    run_dir = _resolve_run_dir_argument(args.run_id, args.run_dir)
    config_path = run_dir / "config" / "config.yaml"
    if not config_path.exists():
        raise CLIError(f"Config file missing in run directory: {config_path}")

    checkpoint = _select_checkpoint(run_dir, args.checkpoint)
    print(f"[train_cli][resume] Resuming {run_dir.name} from {checkpoint}")

    git_commit, git_dirty = _resolve_git_info(REPO_ROOT)
    env = _normalise_env()
    command = _build_training_command(
        config_path,
        run_dir,
        git_commit=git_commit,
        git_dirty=git_dirty,
        trial_label=args.trial_label,
        resume_from=checkpoint,
        fast_dev_run=args.fast_dev_run,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        log_lr=args.log_lr,
        extra_args=args.trainer_arg,
    )
    log_path = run_dir / f"resume_{_timestamp()}.log"
    ret = _stream_subprocess(command, log_path, cwd=REPO_ROOT, env=env)
    if ret != 0:
        raise CLIError(f"Resume command exited with status {ret}")
    _post_process_run(run_dir)
    print(f"[train_cli][resume] Completed resume for {run_dir}")


def _extract_best_metric(metrics_path: Path) -> Optional[Tuple[int, float]]:
    import csv

    if not metrics_path.exists():
        return None
    best_epoch = None
    best_value = None
    with metrics_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if "val_loss" not in row or not row["val_loss"]:
                continue
            try:
                val_loss = float(row["val_loss"])
            except ValueError:
                continue
            if best_value is None or val_loss < best_value:
                best_value = val_loss
                best_epoch = int(float(row.get("epoch", 0)))
    if best_value is None:
        return None
    return best_epoch or 0, best_value


def _collect_val_history(metrics_files: Sequence[Path]) -> List[Dict[str, Any]]:
    import csv

    history: List[Dict[str, Any]] = []
    for metrics_file in metrics_files:
        if not metrics_file.exists():
            continue
        with metrics_file.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                val_loss = row.get("val_loss")
                if not val_loss:
                    continue
                try:
                    value = float(val_loss)
                except ValueError:
                    continue
                epoch = row.get("epoch")
                try:
                    epoch_idx = int(float(epoch)) if epoch is not None else None
                except ValueError:
                    epoch_idx = None
                history.append(
                    {
                        "epoch": epoch_idx,
                        "val_loss": value,
                        "source": str(metrics_file),
                    }
                )
    history.sort(key=lambda item: item["val_loss"])
    return history


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes = remainder // 60
    if hours == 0 and minutes == 0 and seconds > 0:
        return "<1m"
    if hours == 0:
        return f"{minutes}m"
    return f"{hours}h {minutes:02d}m"


def _estimate_runtime(
    run_metadata: Dict[str, Any],
    config: Dict[str, Any],
    history: Sequence[Dict[str, Any]],
) -> Dict[str, Optional[str]]:
    created_iso = run_metadata.get("created")
    estimate: Dict[str, Optional[str]] = {
        "elapsed": None,
        "best_case": None,
        "median": None,
        "worst_case": None,
    }
    if not created_iso:
        return estimate
    try:
        start_time = _dt.datetime.fromisoformat(created_iso)
    except ValueError:
        return estimate
    if start_time.tzinfo is None:
        now = _dt.datetime.now()
    else:
        now = _dt.datetime.now(start_time.tzinfo)
    elapsed_seconds = max((now - start_time).total_seconds(), 0.0)
    estimate["elapsed"] = _format_duration(elapsed_seconds)

    if not history:
        return estimate

    completed_epochs = [item["epoch"] for item in history if item.get("epoch") is not None]
    if not completed_epochs:
        return estimate

    max_completed_epoch = max(completed_epochs)
    epochs_completed = max_completed_epoch + 1
    if epochs_completed <= 0:
        return estimate

    avg_epoch_duration = elapsed_seconds / epochs_completed
    total_epochs = int(config.get("num_epochs", epochs_completed))
    remaining_epochs = max(total_epochs - epochs_completed, 0)

    best_case = avg_epoch_duration if remaining_epochs > 0 else 0.0
    median_case = avg_epoch_duration * max(remaining_epochs / 2, 0)
    worst_case = avg_epoch_duration * remaining_epochs

    estimate["best_case"] = _format_duration(best_case)
    estimate["median"] = _format_duration(median_case)
    estimate["worst_case"] = _format_duration(worst_case)
    return estimate


def _summarise_run(run_dir: Path) -> Dict[str, Any]:
    run_metadata = {}
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        run_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    metrics_files = sorted(metrics_dir.glob("*.csv"))
    if not metrics_files:
        # Fall back to Lightning default location
        metrics_files = sorted(run_dir.glob("cpu_training/**/metrics.csv"))
    best_epoch = None
    best_loss = None
    for metrics_file in metrics_files:
        result = _extract_best_metric(metrics_file)
        if result is None:
            continue
        epoch, value = result
        if best_loss is None or value < best_loss:
            best_loss = value
            best_epoch = epoch

    val_history = _collect_val_history(metrics_files)
    top_val_losses = []
    for idx, entry in enumerate(val_history[:3]):
        top_val_losses.append(
            {
                "rank": idx + 1,
                "val_loss": entry["val_loss"],
                "epoch": entry["epoch"],
            }
        )

    checkpoint_dir = run_dir / "model_checkpoints"
    best_checkpoint = None
    best_symlink = checkpoint_dir / "best.ckpt"
    if best_symlink.exists():
        best_checkpoint = str(best_symlink.resolve())
    else:
        candidates = sorted(checkpoint_dir.glob("*.chkpt"))
        if candidates:
            best_checkpoint = str(candidates[0])

    config_path = run_dir / "config" / "config.yaml"
    config_snippet = {}
    if config_path.exists():
        config_snippet = _load_yaml(config_path)

    feature_metadata_path = run_dir / "feature_metadata.json"
    feature_info = {}
    if feature_metadata_path.exists():
        feature_info = json.loads(feature_metadata_path.read_text(encoding="utf-8"))

    summary = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "best_checkpoint": best_checkpoint,
        "config": config_snippet,
        "feature_metadata": feature_info,
        "run_metadata": run_metadata,
        "top_val_losses": top_val_losses,
    }

    runtime_estimate = _estimate_runtime(run_metadata, config_snippet, val_history)
    if runtime_estimate["elapsed"] is not None:
        summary["runtime_estimate"] = runtime_estimate

    return summary


def cmd_summarise(args: argparse.Namespace) -> None:
    run_dir = _resolve_run_dir_argument(args.run_id, args.run_dir)
    summary = _summarise_run(run_dir)
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m train_cli",
        description="Dynamic TopoQA training command line interface.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m train_cli run --config config.yaml\n"
            "  python -m train_cli run --config configs/sched_boost.yaml --override seed=777\n"
            "  python -m train_cli batch --manifest manifests/run_all.yaml\n"
            "  python -m train_cli resume --run-id training_run_2025-10-28_18-16-16\n"
            "  python -m train_cli summarise --run-dir training_runs/history/training_run_2025-10-28_18-16-16\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run",
        help="Launch a single training job.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run_parser.add_argument("--config", type=Path, required=True, help="Path to YAML configuration file.")
    run_parser.add_argument("--output-root", type=Path, default=RUN_ROOT, help="Directory where run folders are created.")
    run_parser.add_argument("--run-name", type=str, help="Custom run directory name (default: timestamp).")
    run_parser.add_argument(
        "--override",
        action="append",
        help="Override YAML values using dotted.path=value syntax (may be specified multiple times).",
    )
    run_parser.add_argument("--notes", type=str, help="Optional free-form notes stored with the run.")
    run_parser.add_argument("--trial-label", type=str, help="Label recorded in training logs.")
    run_parser.add_argument("--resume-from", type=Path, help="Checkpoint path to resume from.")
    run_parser.add_argument("--fast-dev-run", action="store_true", help="Enable Lightning fast_dev_run flag.")
    run_parser.add_argument("--limit-train-batches", type=float, help="Limit training batches per epoch.")
    run_parser.add_argument("--limit-val-batches", type=float, help="Limit validation batches per epoch.")
    run_parser.add_argument("--log-lr", action="store_true", help="Log learning rate schedule each epoch.")
    run_parser.add_argument(
        "--trainer-arg",
        action="append",
        help="Additional trainer arguments; recognised keys progress_bar_refresh_rate and log_every_n_steps override defaults.",
    )
    run_parser.set_defaults(func=cmd_run)

    batch_parser = subparsers.add_parser(
        "batch",
        help="Run multiple jobs defined in a YAML manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    batch_parser.add_argument("--manifest", type=Path, required=True, help="Path to batch manifest YAML.")
    batch_parser.add_argument("--output-root", type=Path, default=RUN_ROOT, help="Root directory for created runs.")
    batch_parser.set_defaults(func=cmd_batch)

    resume_parser = subparsers.add_parser(
        "resume",
        help="Resume a completed/paused training run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    resume_parser.add_argument("--run-id", type=str, help="Run identifier under training_runs/ (e.g., training_run_2025-01-01_12-00-00).")
    resume_parser.add_argument("--run-dir", type=Path, help="Explicit path to run directory.")
    resume_parser.add_argument("--checkpoint", type=str, help="Checkpoint file to resume from (default: best checkpoint).")
    resume_parser.add_argument("--trial-label", type=str, help="Optional label recorded in training logs.")
    resume_parser.add_argument("--fast-dev-run", action="store_true", help="Enable Lightning fast_dev_run flag.")
    resume_parser.add_argument("--limit-train-batches", type=float, help="Limit training batches per epoch.")
    resume_parser.add_argument("--limit-val-batches", type=float, help="Limit validation batches per epoch.")
    resume_parser.add_argument("--log-lr", action="store_true", help="Log learning rate schedule each epoch.")
    resume_parser.add_argument(
        "--trainer-arg",
        action="append",
        help="Additional raw arguments forwarded to model_train_topoqa_cpu.py (may be repeated).",
    )
    resume_parser.set_defaults(func=cmd_resume)

    summarise_parser = subparsers.add_parser(
        "summarise",
        help="Summarise metrics and artifacts for a run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    summarise_parser.add_argument("--run-id", type=str, help="Run identifier under training_runs/.")
    summarise_parser.add_argument("--run-dir", type=Path, help="Explicit path to run directory.")
    summarise_parser.set_defaults(func=cmd_summarise)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help(sys.stderr)
        return 1
    try:
        args.func(args)
    except CLIError as exc:
        print(f"[train_cli] ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
