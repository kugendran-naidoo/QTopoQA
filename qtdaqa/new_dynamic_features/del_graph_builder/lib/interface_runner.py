from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .new_calculate_interface import process_pdb_file
from .stage_common import structure_model_key


def _round_interface_file(path: Path, decimals: int) -> None:
    if decimals < 0:
        return
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return

    def _format(value: float) -> str:
        quant = Decimal(1).scaleb(-decimals)
        decimal_value = Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP)
        text = format(decimal_value.normalize(), "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"

    changed = False
    rewritten: List[str] = []
    for line in raw_lines:
        parts = line.split()
        if len(parts) < 4:
            rewritten.append(line)
            continue
        coords = parts[1:4]
        try:
            rounded = [_format(float(value)) for value in coords]
        except ValueError:
            rewritten.append(line)
            continue
        if rounded != coords:
            changed = True
            rewritten.append(" ".join([parts[0], *rounded, *parts[4:]]))
        else:
            rewritten.append(line)

    if changed:
        path.write_text("\n".join(rewritten) + ("\n" if raw_lines else ""), encoding="utf-8")


def run_interface_stage(
    pdb_paths: Iterable[Path],
    dataset_dir: Path,
    work_dir: Path,
    log_dir: Path,
    cutoff: float,
    coordinate_decimals: int,
    jobs: Optional[int] = None,
) -> Dict[str, object]:
    interface_dir = work_dir / "interface"
    interface_dir.mkdir(parents=True, exist_ok=True)
    interface_log_dir = log_dir / "interface_logs"
    interface_log_dir.mkdir(parents=True, exist_ok=True)

    tasks: List[Tuple[Path, Path, Path]] = []
    for pdb_path in sorted(pdb_paths):
        model_key = structure_model_key(dataset_dir, pdb_path)
        model_rel = Path(model_key)
        output_parent = interface_dir / model_rel.parent
        log_parent = interface_log_dir / model_rel.parent
        output_parent.mkdir(parents=True, exist_ok=True)
        log_parent.mkdir(parents=True, exist_ok=True)
        output_path = output_parent / f"{pdb_path.stem}.interface.txt"
        log_path = log_parent / f"{pdb_path.stem}.log"
        tasks.append((pdb_path, output_path, log_path))

    worker_count = max(1, int(jobs)) if jobs else 1
    success = 0
    failures: List[Tuple[Path, Path, str]] = []
    elapsed = 0.0

    if tasks:
        start = time.perf_counter()
        if worker_count <= 1:
            for pdb_path, output_path, log_path in tasks:
                try:
                    residue_count = process_pdb_file(pdb_path, output_path, cutoff=cutoff)
                    _round_interface_file(output_path, coordinate_decimals)
                    success += 1
                    log_path.write_text(f"Interface residues: {residue_count}\n", encoding="utf-8")
                except Exception as exc:  # pragma: no cover
                    failures.append((pdb_path, log_path, str(exc)))
                    log_path.write_text(f"Failure: {exc}\n", encoding="utf-8")
        else:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                future_to_task = {
                    executor.submit(process_pdb_file, pdb_path, output_path, cutoff): (pdb_path, output_path, log_path)
                    for pdb_path, output_path, log_path in tasks
                }
                for future in as_completed(future_to_task):
                    pdb_path, output_path, log_path = future_to_task[future]
                    try:
                        residue_count = future.result()
                        _round_interface_file(output_path, coordinate_decimals)
                        success += 1
                        log_path.write_text(f"Interface residues: {residue_count}\n", encoding="utf-8")
                    except Exception as exc:  # pragma: no cover
                        failures.append((pdb_path, log_path, str(exc)))
                        log_path.write_text(f"Failure: {exc}\n", encoding="utf-8")
        elapsed = time.perf_counter() - start

    return {
        "output_dir": interface_dir,
        "log_dir": interface_log_dir,
        "success": success,
        "failures": failures,
        "elapsed": elapsed,
        "processed": len(tasks),
    }
