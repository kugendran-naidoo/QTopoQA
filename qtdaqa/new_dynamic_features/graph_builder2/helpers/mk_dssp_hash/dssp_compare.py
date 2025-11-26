#!/usr/bin/env python3
"""
Quick DSSP sanity check to compare outputs across machines.

Usage:
  # single file
  python dssp_compare.py --path-to-pdb /absolute/path/to/file.pdb
  # list of files (one per line, absolute or relative paths)
  python dssp_compare.py --pdb-file-list /path/to/list.txt

With --path-to-pdb, it prints a SHA256 hash of the DSSP records and writes a JSON
snapshot alongside the PDB (same stem, .dssp.json). With --pdb-file-list, it prints
"<filename> <hash>" per line for each file in the list (no JSON outputs).
Run on two machines with the same inputs and compare hashes to see if mkdssp/biopython
outputs differ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
import warnings

from Bio.PDB import DSSP, PDBParser

LOG_PATH = Path.cwd() / "failures_warnings.log"


def _record_warnings(pdb_path: Path, warning_list) -> None:
    if not warning_list:
        return
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        for w in warning_list:
            handle.write(f"WARNING {pdb_path.name}: {w.category.__name__}: {w.message}\n")


def compute_dssp_records(pdb_path: Path):
    parser = PDBParser(QUIET=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        structure = parser.get_structure("test", pdb_path)
        model = structure[0]
        dssp = DSSP(model, pdb_path, file_type="PDB", dssp="mkdssp")
    _record_warnings(pdb_path, caught)

    records = []
    for key in dssp.keys():
        chain_id, res_id = key
        _, resseq, icode = res_id
        ss8, asa, phi, psi = dssp[key][2:6]
        records.append(
            {
                "chain": chain_id,
                "resseq": resseq,
                "icode": (icode or " ").strip() or " ",
                "ss8": ss8,
                "asa": float(asa),
                "phi": float(phi),
                "psi": float(psi),
            }
        )

    records.sort(key=lambda r: (r["chain"], r["resseq"], r["icode"]))
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run DSSP on a PDB and emit a hash + JSON snapshot for cross-machine comparison."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--path-to-pdb",
        dest="pdb_path",
        type=Path,
        help="Absolute or relative path to a single PDB file.",
    )
    group.add_argument(
        "--pdb-file-list",
        dest="pdb_file_list",
        type=Path,
        help="Path to a text file containing PDB paths (one per line).",
    )

    args = parser.parse_args(argv)
    # fresh log for each invocation
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    success = 0
    warn_count = 0
    fail_count = 0
    if args.pdb_path:
        pdb_path: Path = args.pdb_path.expanduser().resolve()
        if not pdb_path.exists():
            print(f"Error: PDB file not found: {pdb_path}", file=sys.stderr)
            return 1
        try:
            records = compute_dssp_records(pdb_path)
            payload = json.dumps(records, sort_keys=True).encode("utf-8")
            digest = hashlib.sha256(payload).hexdigest()
            print(f"PDB: {pdb_path}")
            print(f"Residues: {len(records)}")
            print(f"SHA256 of DSSP records: {digest}")
            out_json = pdb_path.with_suffix(".dssp.json")
            out_json.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
            print(f"Wrote {out_json}")
            success += 1
            # warnings already logged to LOG_PATH if any
        except Exception as exc:  # pragma: no cover - best effort
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LOG_PATH.open("a", encoding="utf-8") as handle:
                handle.write(f"FAILURE {pdb_path.name}: {exc}\n")
            print(f"ERROR: {exc}", file=sys.stderr)
            fail_count += 1
        if LOG_PATH.exists():
            warn_count = sum(1 for line in LOG_PATH.read_text(encoding="utf-8").splitlines() if line.startswith("WARNING "))
        print("Summary:")
        print(f"successes={success}")
        print(f"warnings={warn_count}")
        print(f"failures={fail_count}")
        if warn_count or fail_count:
            print(f"(details in {LOG_PATH.name})")
        return 1 if fail_count else 0

    # list mode
    list_path: Path = args.pdb_file_list.expanduser().resolve()
    if not list_path.exists():
        print(f"Error: list file not found: {list_path}", file=sys.stderr)
        return 1
    lines = list_path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        pdb_path = Path(line).expanduser().resolve()
        if not pdb_path.exists():
            print(f"{pdb_path.name} MISSING", file=sys.stderr)
            fail_count += 1
            continue
        try:
            records = compute_dssp_records(pdb_path)
            payload = json.dumps(records, sort_keys=True).encode("utf-8")
            digest = hashlib.sha256(payload).hexdigest()
            print(f"{pdb_path.name} {digest}")
            success += 1
        except Exception as exc:  # pragma: no cover - best effort
            print(f"{pdb_path.name} ERROR: {exc}", file=sys.stderr)
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LOG_PATH.open("a", encoding="utf-8") as handle:
                handle.write(f"FAILURE {pdb_path.name}: {exc}\n")
            fail_count += 1
    # derive warning count from log (best effort)
    if LOG_PATH.exists():
        warn_count = sum(1 for line in LOG_PATH.read_text(encoding="utf-8").splitlines() if line.startswith("WARNING "))
    print("Summary:")
    print(f"successes={success}")
    print(f"warnings={warn_count}")
    print(f"failures={fail_count}")
    if warn_count or fail_count:
        print(f"(details in {LOG_PATH.name})")
    return 1 if fail_count else 0


if __name__ == "__main__":
    sys.exit(main())
