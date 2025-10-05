#!/usr/bin/env python3
"""
cgal_compare_v3.py
Compare two probe JSONs produced by cgal_probe_v3.py for CGAL/GUDHI stability testing.

Defaults:
  --tolerance 1e-10   (float equality tolerance)
Behavior:
  - Prints usage if run with no arguments or invalid inputs
  - Compares hashes for alpha_filtration, alpha_b1, vr0
  - Exits 0 if identical, 1 if different, 2 if bad arguments
"""

import json, argparse, sys
from pathlib import Path

def load_probe(path: Path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser(description="Compare CGAL/GUDHI probe JSONs.")
    ap.add_argument("probe_a", nargs="?", help="First *_probe.json")
    ap.add_argument("probe_b", nargs="?", help="Second *_probe.json")
    ap.add_argument("--tolerance", type=float, default=1e-10, help="Numeric tolerance for comparison")
    args = ap.parse_args()

    # Handle missing args
    if args.probe_a is None or args.probe_b is None:
        ap.print_help(sys.stderr)
        sys.exit(2)

    A_path, B_path = Path(args.probe_a), Path(args.probe_b)
    if not A_path.exists() or not B_path.exists():
        print(f"[!] Missing input file(s): {A_path if not A_path.exists() else ''} {B_path if not B_path.exists() else ''}")
        ap.print_help(sys.stderr)
        sys.exit(2)

    try:
        A = load_probe(A_path)
        B = load_probe(B_path)
    except Exception as e:
        print(f"Error reading probe files: {e}")
        sys.exit(2)

    print(f"A env: {A.get('platform')} | GUDHI {A.get('gudhi')}")
    print(f"B env: {B.get('platform')} | GUDHI {B.get('gudhi')}")
    print(f"Point counts: {A.get('hashes',{}).get('points') == B.get('hashes',{}).get('points')} (hash equality)")
    
    HA, HB = A.get("hashes", {}), B.get("hashes", {})
    keys = ["alpha_filtration", "alpha_b1", "vr0"]

    diff = []
    for k in keys:
        match = HA.get(k) == HB.get(k)
        print(f"Hash equal [{k}]: {match}")
        if not match:
            diff.append(k)

    if diff:
        print(f"VERDICT: DIFFERENT in {', '.join(diff)}")
        sys.exit(1)
    else:
        print("VERDICT: IDENTICAL (all hashes match).")
        sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(2)

