#!/usr/bin/env python3
"""
cgal_probe_v3.py
Generate reproducible fingerprints of CGAL/GUDHI outcomes for Alpha and VR complexes.
Designed to test CGAL stability between GUDHI builds or environments.

Outputs:
  - *_alpha_filtration.csv : simplex + filtration values (sorted)
  - *_alpha1.csv           : Alpha Betti-1 barcode
  - *_vr0.csv              : VR Betti-0 barcode
  - *_probe.json           : environment metadata + hashes

Default options:
  --outdir output/      (created if missing)
  --grid 1e-8           (VR quantization grid)
  --rmax auto           (max pairwise distance)
  --coeff 2             (Z₂ field)
  --alpha-inexact False (use exact predicates)
  --include-h False     (exclude hydrogens)
  --ca-only False       (use all heavy atoms)
"""

import os, sys, json, argparse, hashlib, platform
from pathlib import Path
from datetime import datetime
import numpy as np

# Determinism
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

try:
    import gudhi
    from gudhi import SimplexTree
    from gudhi.alpha_complex import AlphaComplex
except Exception:
    gudhi = None


# ---------- Utility helpers ----------

def stable_hash_array(a: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()

def stable_hash_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

def sniff_filetype(p: Path) -> str:
    ext = p.suffix.lower()
    if ext in [".xyz", ".pdb", ".npy"]:
        return ext[1:]
    return "unknown"

def _is_float(s: str) -> bool:
    try:
        float(s); return True
    except Exception:
        return False

def load_xyz(path: Path, include_h=False) -> np.ndarray:
    coords = []
    with open(path, "r") as f:
        lines = f.readlines()
    try:
        n = int(lines[0].strip().split()[0])
        body = lines[2:2+n]
    except Exception:
        body = lines
    for line in body:
        parts = line.strip().split()
        if len(parts) < 4: continue
        elem = parts[0].upper()
        if not include_h and elem == "H": continue
        try:
            x, y, z = map(float, parts[-3:])
            coords.append([x, y, z])
        except Exception:
            continue
    if not coords:
        raise ValueError("No points parsed from XYZ")
    return np.unique(np.array(coords, dtype=float), axis=0)

def load_pdb(path: Path, include_h=False, ca_only=False) -> np.ndarray:
    coords = []
    with open(path, "r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            atom_name = line[12:16].strip()
            elem = (line[76:78].strip() or atom_name[:1]).upper()
            if not include_h and elem == "H":
                continue
            if ca_only and atom_name != "CA":
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except Exception:
                parts = line.split()
                floats = [p for p in parts if _is_float(p)]
                if len(floats) >= 3:
                    x, y, z = map(float, floats[-3:])
                else:
                    continue
            coords.append([x, y, z])
    if not coords:
        raise ValueError("No coordinates parsed from PDB")
    return np.unique(np.array(coords, dtype=float), axis=0)

def load_points(path: Path, include_h=False, ca_only=False) -> np.ndarray:
    t = sniff_filetype(path)
    if t == "xyz":
        return load_xyz(path, include_h)
    elif t == "pdb":
        return load_pdb(path, include_h, ca_only)
    elif t == "npy":
        return np.load(path)
    else:
        raise ValueError(f"Unsupported file type {path.suffix}")

# ---------- Core GUDHI builders ----------

def build_alpha(points: np.ndarray, exact=True) -> SimplexTree:
    try:
        ac = AlphaComplex(points=points, exact=exact)
    except TypeError:
        try:
            ac = AlphaComplex(points=points, exact_version=exact)
        except TypeError:
            ac = AlphaComplex(points=points)
    return ac.create_simplex_tree()

def build_vr0(points: np.ndarray, rmax="auto", grid=1e-8) -> SimplexTree:
    st = SimplexTree()
    n = points.shape[0]
    for i in range(n):
        st.insert([i], filtration=0.0)
    D = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    if rmax == "auto":
        r = float(np.max(D))
    else:
        r = float(rmax)
    edges = []
    def q(x): return float(np.round(x / grid) * grid)
    for i in range(n):
        for j in range(i + 1, n):
            d = D[i, j]
            if d <= r:
                edges.append((q(d), (i, j)))
    edges.sort(key=lambda t: (t[0], t[1]))
    for f, (i, j) in edges:
        st.insert([i, j], filtration=f)
    try:
        st.initialize_filtration()
    except Exception:
        pass
    return st

def dump_alpha_filtration(st: SimplexTree):
    items = []
    for s, f in st.get_filtration():
        items.append((tuple(s), float(f)))
    items.sort(key=lambda t: (t[1], len(t[0]), t[0]))
    return items

def save_filtration_csv(items, path: Path):
    with open(path, "w") as f:
        f.write("simplex;filtration\n")
        for s, v in items:
            f.write(f"{','.join(map(str, s))};{v:.12g}\n")

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Probe CGAL/GUDHI outcomes for Alpha complex & VR.")
    ap.add_argument("input", nargs="?", help=".xyz/.pdb/.npy point cloud")
    ap.add_argument("--include-h", action="store_true", help="Include hydrogens (default: exclude)")
    ap.add_argument("--ca-only", action="store_true", help="Use only CA atoms (PDB)")
    ap.add_argument("--grid", type=float, default=1e-8, help="Quantization grid for VR distances")
    ap.add_argument("--rmax", default="auto", help="VR max radius (float or 'auto')")
    ap.add_argument("--coeff", type=int, default=2, help="Field modulus p (prime, default=2)")
    ap.add_argument("--alpha-inexact", action="store_true", help="Use inexact Alpha predicates (not recommended)")
    ap.add_argument("--outdir", default="output", help="Output directory (default=output)")
    args = ap.parse_args()

    # No args provided
    if args.input is None:
        ap.print_help(sys.stderr)
        sys.exit(2)

    # Validate input file
    path = Path(args.input)
    if not path.exists():
        print(f"[!] Input file not found: {path}")
        ap.print_help(sys.stderr)
        sys.exit(2)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if gudhi is None:
        print("ERROR: GUDHI not installed.")
        sys.exit(1)

    pts = load_points(path, include_h=args.include_h, ca_only=args.ca_only)
    print(f"Loaded {pts.shape[0]} points from {path}")

    meta = {
        "timestamp": datetime.utcnow().isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "gudhi": getattr(gudhi, "__version__", "unknown"),
        "coeff": args.coeff,
        "grid": args.grid,
        "rmax": args.rmax,
        "alpha_inexact": args.alpha_inexact,
        "include_h": args.include_h,
        "ca_only": args.ca_only
    }

    # Alpha
    st_a = build_alpha(pts, exact=not args.alpha_inexact)
    st_a.compute_persistence(homology_coeff_field=args.coeff, persistence_dim_max=True)
    alpha_filtration = dump_alpha_filtration(st_a)
    a_b1 = np.array(st_a.persistence_intervals_in_dimension(1), dtype=float)

    # VR
    st_v = build_vr0(pts, rmax=args.rmax, grid=args.grid)
    st_v.compute_persistence(homology_coeff_field=args.coeff, persistence_dim_max=True)
    v_b0 = np.array(st_v.persistence_intervals_in_dimension(0), dtype=float)

    # Save
    alpha_csv = outdir / f"{path.stem}_alpha_filtration.csv"
    alpha1_csv = outdir / f"{path.stem}_alpha1.csv"
    vr0_csv = outdir / f"{path.stem}_vr0.csv"
    probe_json = outdir / f"{path.stem}_probe.json"

    save_filtration_csv(alpha_filtration, alpha_csv)
    np.savetxt(alpha1_csv, a_b1, delimiter=",", header="birth,death", comments="")
    np.savetxt(vr0_csv, v_b0, delimiter=",", header="birth,death", comments="")

    meta["hashes"] = {
        "alpha_filtration": stable_hash_str(json.dumps(alpha_filtration)),
        "alpha_b1": stable_hash_array(a_b1),
        "vr0": stable_hash_array(v_b0),
        "points": stable_hash_array(pts)
    }

    with open(probe_json, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved probe artifacts to: {outdir.resolve()}")

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

