#!/usr/bin/env python3
"""
Deterministic persistent homology harness (v4) for .xyz and .pdb point clouds.

New in v4:
- Default output directory is now "output".
- All screen output is also written to a per-run logfile in the output directory.
- Logs include all chosen parameters and environment settings.
"""

import os, sys, platform, json, hashlib, argparse, io
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# ---------- Determinism: cap threads ----------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

try:
    import gudhi
    from gudhi import SimplexTree
    from gudhi.alpha_complex import AlphaComplex
except Exception:
    gudhi = None

# ---------- Helpers ----------

def get_numpy_blas_info():
    try:
        from numpy.__config__ import get_info
        return {"blas_opt_info": get_info('blas_opt_info'),
                "lapack_opt_info": get_info('lapack_opt_info')}
    except Exception as e:
        return {"error": str(e)}

def stable_hash(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(arr.tobytes())
    return h.hexdigest()

def print_versions_and_settings():
    print("===== Platform & Library Settings for Reproducibility =====")
    print(f"Timestamp (UTC): {datetime.utcnow().isoformat()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}  Processor: {platform.processor()}")
    print(f"NumPy: {np.__version__}")
    print("NumPy BLAS/LAPACK info:")
    print(json.dumps(get_numpy_blas_info(), indent=2, default=str))
    if gudhi is None:
        print("GUDHI: NOT INSTALLED")
    else:
        print(f"GUDHI: {getattr(gudhi, '__version__', 'unknown')}")
        print("CGAL: (not exposed by GUDHI Python; ensure same CGAL at build time)")
    print("Thread env caps (should all be '1' for determinism):")
    for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"]:
        print(f"  {k}={os.environ.get(k)}")
    print("==========================================================\n")

def quantize(x, grid=1e-8):
    return np.round(x / grid) * grid

# ---------- VR & Alpha ----------

def build_vr_simplex_tree(points: np.ndarray, r_max=None, grid=1e-8):
    n = points.shape[0]
    st = SimplexTree()
    for i in range(n):
        st.insert([i], filtration=0.0)
    # pairwise distances
    D = np.linalg.norm(points[:,None,:] - points[None,:,:], axis=-1)
    if r_max is None or (isinstance(r_max, str) and str(r_max).lower() == "auto"):
        r_max = float(np.max(D))
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            d = D[i, j]
            if d <= r_max:
                f = float(quantize(d, grid=grid))
                edges.append((f, 1, (i, j)))
    # Stable sort by (filtration, dimension, vertex-lexicographic)
    edges.sort(key=lambda t: (t[0], t[1], t[2]))
    for f, dim, verts in edges:
        st.insert(list(verts), filtration=f)
    st.initialize_filtration()
    return st

def compute_betti0_barcode_vr(points: np.ndarray, r_max=None, grid=1e-8, coeff=2):
    st = build_vr_simplex_tree(points, r_max=r_max, grid=grid)
    st.compute_persistence(homology_coeff_field=coeff, persistence_dim_max=True)
    b0 = np.array([iv for iv in st.persistence_intervals_in_dimension(0)], dtype=np.float64)
    return b0

def compute_alpha_betti1(points: np.ndarray, exact=True, coeff=2):
    try:
        ac = AlphaComplex(points=points, exact=exact)
    except TypeError:
        try:
            ac = AlphaComplex(points=points, exact_version=exact)
        except TypeError:
            ac = AlphaComplex(points=points)
    st: SimplexTree = ac.create_simplex_tree()
    st.compute_persistence(homology_coeff_field=coeff, persistence_dim_max=True)
    b1 = np.array([iv for iv in st.persistence_intervals_in_dimension(1)], dtype=np.float64)
    return b1

# ---------- File handling ----------

def sniff_filetype(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in [".xyz", ".pdb"]:
        return ext[1:]
    try:
        with open(path, "r", errors="ignore") as f:
            head = [next(f) for _ in range(5)]
        text = "".join(head)
    except Exception:
        return "unknown"
    if text.lstrip().split()[0].isdigit() and len(head) >= 2:
        return "xyz"
    if any(line.startswith(("ATOM", "HETATM")) for line in head):
        return "pdb"
    return "unknown"

def _is_float(s: str) -> bool:
    try:
        float(s); return True
    except Exception:
        return False

def load_xyz(path: Path, include_h=False) -> Tuple[np.ndarray, Dict]:
    coords = []
    elements = []
    with open(path, "r") as f:
        lines = f.readlines()
    if not lines:
        raise ValueError("Empty XYZ file")
    try:
        n = int(lines[0].strip().split()[0])
        start = 2
        body = lines[start:start+n]
    except Exception:
        body = lines
    for line in body:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        elem = parts[0]
        if not include_h and elem.upper() == "H":
            continue
        try:
            x, y, z = map(float, parts[-3:])
            coords.append([x, y, z]); elements.append(elem.upper())
        except Exception:
            continue
    if not coords:
        raise ValueError("No coordinates parsed from XYZ")
    pts = np.asarray(coords, dtype=np.float64)
    meta = {
        "type": "xyz",
        "element_counts": {e: int(sum(1 for ee in elements if ee == e)) for e in sorted(set(elements))}
    }
    return pts, meta

def load_pdb(path: Path, include_h=False, ca_only=False, chain_sel=None) -> Tuple[np.ndarray, Dict, np.ndarray, np.ndarray, np.ndarray]:
    coords = []
    chains = []
    res_ids = []
    elements = []
    with open(path, "r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            chain_id = (line[21:22].strip() or "?")
            if chain_sel and chain_id not in chain_sel:
                continue
            atom_name = line[12:16].strip()
            elem = (line[76:78].strip() or atom_name[:1]).upper()
            if not include_h and elem == "H":
                continue
            if ca_only and atom_name != "CA":
                continue
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except Exception:
                parts = line.split()
                floats = [p for p in parts if _is_float(p)]
                if len(floats) >= 3:
                    x, y, z = map(float, floats[-3:])
                else:
                    continue
            resseq = line[22:26].strip()
            icode = line[26:27].strip()
            resid = f"{chain_id}:{resseq}{icode}"
            coords.append([x, y, z]); chains.append(chain_id); res_ids.append(resid); elements.append(elem)
    if not coords:
        raise ValueError("No coordinates parsed from PDB (filters too strict?)")
    pts = np.asarray(coords, dtype=np.float64)
    chains_arr = np.array(chains)
    res_arr = np.array(res_ids)
    elem_arr = np.array(elements)
    # per-chain stats
    chain_stats = {}
    for c in sorted(set(chains)):
        idx = (chains_arr == c)
        chain_stats[c] = {
            "atom_count": int(np.sum(idx)),
            "unique_residues": int(len(set(res_arr[idx].tolist())))
        }
    meta = {
        "type": "pdb",
        "chain_stats": chain_stats,
        "element_counts": {e: int(np.sum(elem_arr == e)) for e in sorted(set(elem_arr))},
        "filters": {
            "include_h": bool(include_h),
            "ca_only": bool(ca_only),
            "chain_sel": None if chain_sel is None else list(chain_sel),
        }
    }
    return pts, meta, chains_arr, res_arr, elem_arr

def select_interface(points: np.ndarray, chains_arr: np.ndarray, cutoff: float) -> np.ndarray:
    """Return boolean mask selecting atoms within 'cutoff' Å of any atom from a different chain."""
    n = points.shape[0]
    mask = np.zeros(n, dtype=bool)
    # O(n^2) deterministic check
    for i in range(n):
        for j in range(n):
            if chains_arr[i] == chains_arr[j]:
                continue
            dx = points[i,0]-points[j,0]; dy = points[i,1]-points[j,1]; dz = points[i,2]-points[j,2]
            if (dx*dx + dy*dy + dz*dz) <= cutoff*cutoff:
                mask[i] = True; mask[j] = True
    return mask

# ---------- Logging (tee stdout to file) ----------

class Tee(io.TextIOBase):
    def __init__(self, stream, filepath):
        self.stream = stream
        self.file = open(filepath, "w", buffering=1, encoding="utf-8")
    def write(self, s):
        self.stream.write(s)
        self.file.write(s)
        return len(s)
    def flush(self):
        self.stream.flush()
        self.file.flush()
    def close(self):
        try:
            self.file.close()
        except Exception:
            pass

# ---------- Main ----------

def build_argparser():
    ap = argparse.ArgumentParser(
        description="Deterministic PH (VR Betti-0 & Alpha Betti-1) for .xyz/.pdb point clouds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=True
    )
    ap.add_argument("inputs", nargs="+", help="Input .xyz or .pdb files")
    ap.add_argument("--include-h", action="store_true", help="Include hydrogens (default: exclude)")
    ap.add_argument("--ca-only", action="store_true", help="(PDB) Use only Cα atoms")
    ap.add_argument("--chain", type=str, default="", help="(PDB) Comma-separated list of chain IDs to include (e.g., A,B). Default: all chains")
    ap.add_argument("--interface", type=float, default=0.0, help="(PDB) Keep only atoms within X Å of another chain (inter-chain interface)")
    ap.add_argument("--grid", type=float, default=1e-8, help="Quantization grid for filtration values")
    ap.add_argument("--rmax", default="auto", help="Max radius for VR (float or 'auto' = max pairwise distance)")
    ap.add_argument("--coeff", type=int, default=2, help="Homology coefficient field (prime field Z_p). Default 2")
    ap.add_argument("--alpha-inexact", action="store_true", help="Use inexact predicates for AlphaComplex (not recommended)")
    ap.add_argument("--outdir", type=str, default="output", help='Output directory (default: "output")')
    return ap

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = build_argparser()
    if len(argv) == 0:
        parser.print_help(sys.stderr); sys.exit(2)
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        raise

    # Ensure output directory exists first (needed for logfile)
    outdir = Path(args.outdir) if args.outdir else Path("output")
    outdir.mkdir(parents=True, exist_ok=True)

    # Setup tee logging to a per-run logfile
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    logfile = outdir / f"ph_run_{ts}.log"
    tee = Tee(sys.stdout, logfile)
    sys.stdout = tee
    sys.stderr = tee

    print(f"Log file: {logfile}")
    print(f"Command line: {' '.join(sys.argv)}")

    print_versions_and_settings()

    if gudhi is None:
        print("ERROR: GUDHI is not installed. Please install gudhi before running.")
        sys.stdout.flush(); tee.close()
        sys.exit(1)

    # parse rmax
    rmax = args.rmax
    if rmax != "auto":
        try:
            rmax = float(rmax)
        except Exception:
            print("Invalid --rmax value. Use a float or 'auto'.")
            parser.print_help(sys.stderr); sys.stdout.flush(); tee.close(); sys.exit(2)

    # chain selection set
    chain_sel = None
    if args.chain.strip():
        chain_sel = set([c.strip() for c in args.chain.split(",") if c.strip()])

    # Echo chosen parameters to the log
    chosen_params = {
        "inputs": args.inputs,
        "include_h": args.include_h,
        "ca_only": args.ca_only,
        "chain": None if chain_sel is None else sorted(chain_sel),
        "interface_cutoff_A": args.interface,
        "grid": args.grid,
        "rmax": args.rmax,
        "coeff": args.coeff,
        "alpha_inexact": args.alpha_inexact,
        "outdir": str(outdir),
    }
    print("Chosen parameters:")
    print(json.dumps(chosen_params, indent=2))

    for path_str in args.inputs:
        path = Path(path_str)
        if not path.exists():
            print(f"[!] Missing file: {path}")
            continue

        ftype = sniff_filetype(path)
        meta_summary = {
            "input_file": str(path),
            "file_type": ftype,
            "parameters": {
                **chosen_params
            },
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "machine": platform.machine(),
                "numpy": np.__version__,
                "gudhi": None if gudhi is None else getattr(gudhi, "__version__", "unknown"),
                "blas_lapack": get_numpy_blas_info(),
                "threads": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"]},
            }
        }

        try:
            if ftype == "xyz":
                pts, meta = load_xyz(path, include_h=args.include_h)
                chains_arr = None
                interface_applied = False
            elif ftype == "pdb":
                pts_all, meta, chains_arr, res_arr, elem_arr = load_pdb(path, include_h=args.include_h, ca_only=args.ca_only, chain_sel=chain_sel)
                interface_applied = False
                if args.interface and args.interface > 0.0:
                    mask = select_interface(pts_all, chains_arr, cutoff=float(args.interface))
                    pts = pts_all[mask]
                    chains_arr = chains_arr[mask]
                    interface_applied = True
                else:
                    pts = pts_all
            else:
                print(f"[!] Unrecognized file type for {path}"); continue
        except Exception as e:
            print(f"[!] Failed to load {path}: {e}"); continue

        if pts.size == 0:
            print(f"[!] No points after filters for {path}")
            continue
        pts = np.unique(pts, axis=0)

        meta_summary["input_stats"] = {
            "point_count": int(pts.shape[0]),
            "points_hash": stable_hash(pts),
            "raw_metadata": meta
        }
        if ftype == "pdb" and chains_arr is not None:
            chain_stats_filtered = {}
            for c in sorted(set(chains_arr.tolist())):
                idx = (chains_arr == c)
                chain_stats_filtered[c] = {"atom_count": int(np.sum(idx))}
            meta_summary["post_filter_chain_stats"] = chain_stats_filtered
            meta_summary["interface_applied"] = bool(interface_applied)

        print(f"\n=== File: {path.name} ===")
        print(f"Points parsed: {pts.shape[0]}")
        print(f"Points stable hash: {meta_summary['input_stats']['points_hash']}")

        # Compute barcodes
        b0 = compute_betti0_barcode_vr(pts, r_max=rmax, grid=args.grid, coeff=args.coeff)
        b1 = compute_alpha_betti1(pts, exact=not args.alpha_inexact, coeff=args.coeff)

        print("----- Vietoris–Rips (Betti-0) -----")
        print(f"Intervals:\n{b0}")
        print(f"Count: {len(b0)}  | Stable hash: {stable_hash(b0)}")

        print("----- Alpha complex (Betti-1) -----")
        print(f"Intervals:\n{b1}")
        print(f"Count: {len(b1)}  | Stable hash: {stable_hash(b1)}")

        # Outputs
        base = path.stem
        out_dir_use = outdir
        out_dir_use.mkdir(parents=True, exist_ok=True)
        b0_path = out_dir_use / f"{base}_vr_b0.csv"
        b1_path = out_dir_use / f"{base}_alpha_b1.csv"
        meta_path = out_dir_use / f"{base}_ph_metadata.json"

        np.savetxt(b0_path, b0, delimiter=",", header="birth,death", comments="")
        np.savetxt(b1_path, b1, delimiter=",", header="birth,death", comments="")
        with open(meta_path, "w") as f:
            json.dump(meta_summary, f, indent=2, default=str)

        print(f"Saved: {b0_path}")
        print(f"Saved: {b1_path}")
        print(f"Saved: {meta_path}")

    sys.stdout.flush()
    tee.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
