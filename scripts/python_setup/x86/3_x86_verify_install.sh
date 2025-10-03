# verify PyTorch, common ML/MD/bio stack, and PyG ecosystem packages
python - << 'PY'
import sys
import traceback

def section(title):
    print(f"\n{title}")
    print("-" * len(title))

def check(pkg_name, import_name=None, version_attr="__version__", extra=None):
    """
    pkg_name: human-friendly name to show
    import_name: module to import (defaults to pkg_name)
    version_attr: attribute to read for version (None to skip)
    extra: optional callable that runs a tiny smoke test
    """
    import importlib
    name = import_name or pkg_name
    try:
        mod = importlib.import_module(name)
        try:
            ver = getattr(mod, version_attr) if version_attr else "unknown"
        except Exception:
            ver = "unknown"
        print(f"{pkg_name}: OK (version: {ver})")
        if extra:
            try:
                extra(mod)
                print(f"{pkg_name} smoke test: OK")
            except Exception as e:
                print(f"{pkg_name} smoke test: FAIL -> {e}")
    except Exception as e:
        print(f"{pkg_name}: NOT INSTALLED or error -> {e}")

# =========================================
# PyTorch + Companions
# =========================================
section("PyTorch tests")
try:
    import torch
    print("PyTorch:", torch.__version__)
    print("MPS available:", torch.backends.mps.is_available())
    print("MPS built:", torch.backends.mps.is_built())
except Exception as e:
    print(f"PyTorch import failed -> {e}")

section("PyTorch Geometric + companions")
# torch_geometric
def pyg_smoke(pyg):
    import torch
    from torch_geometric.data import Data
    x = torch.tensor([[1.0, 0.0],[0.0, 1.0]])
    edge_index = torch.tensor([[0,1],[1,0]], dtype=torch.long)
    _ = Data(x=x, edge_index=edge_index)
check("torch_geometric", "torch_geometric", extra=pyg_smoke)

# pytorch_lightning
def pl_smoke(pl):
    # Try modern Lightning namespaces first, then fall back
    try:
        from lightning import seed_everything  # Lightning 2.x preferred
    except Exception:
        try:
            from lightning_fabric.utilities.seed import seed_everything  # Fabric location
        except Exception:
            try:
                from lightning.pytorch.utilities.seed import seed_everything  # Lightning 2.x alt
            except Exception:
                seed_everything = getattr(pl, "seed_everything", None)  # legacy attr on PL
                if seed_everything is None:
                    raise ImportError(
                        "seed_everything not found in lightning/lightning_fabric/pytorch_lightning"
                    )
    seed_everything(0)

check("pytorch_lightning", "pytorch_lightning", extra=pl_smoke)
# torchmetrics
def torchmetrics_smoke(torchmetrics):
    from torchmetrics.classification import Accuracy
    _ = Accuracy(task="binary")
check("torchmetrics", "torchmetrics", extra=torchmetrics_smoke)

# scikit-learn
def sklearn_smoke(sklearn):
    from sklearn.linear_model import LogisticRegression
    _ = LogisticRegression(max_iter=10)
check("scikit-learn", "sklearn", extra=sklearn_smoke)

# torch_scatter
def tscatter_smoke(tscatter):
    import torch
    from torch_scatter import scatter_sum
    src = torch.tensor([1., 2., 3., 4.])
    index = torch.tensor([0, 0, 1, 1])
    out = scatter_sum(src, index, dim=0, dim_size=2)
    assert out.tolist() == [3.0, 7.0]
check("torch_scatter", "torch_scatter", extra=tscatter_smoke)

# torch_sparse
def tsparse_smoke(tsparse):
    import torch
    from torch_sparse import SparseTensor
    row = torch.tensor([0, 1, 1])
    col = torch.tensor([1, 0, 2])
    _ = SparseTensor(row=row, col=col, sparse_sizes=(3,3))
check("torch_sparse", "torch_sparse", extra=tsparse_smoke)

# torch_cluster
# (import only; some ops need compiled neighbors libs or CUDA)
check("torch_cluster", "torch_cluster")

# torch_spline_conv
def tspline_smoke(tspline):
    from torch_spline_conv import SplineConv
    _ = SplineConv(in_channels=3, out_channels=4, dim=2, kernel_size=3)
check("torch_spline_conv", "torch_spline_conv", extra=tspline_smoke)

# torchtriton (package name is 'triton' in Python)
check("torchtriton (triton)", "triton")

# =========================================
# Core - Math
# =========================================

section("Core array / math")
# numpy
def numpy_smoke(np):
    a = np.array([1, 2, 3])
    b = a * a
    assert b.tolist() == [1, 4, 9]
check("NumPy", "numpy", extra=numpy_smoke)

# SciPy
def scipy_smoke(scipy):
    from scipy import linalg
    import numpy as np
    A = np.array([[1.0, 2.0],[3.0, 4.0]])
    det = linalg.det(A)
    assert abs(det + 2.0) < 1e-8  # det([[1,2],[3,4]]) = -2
check("SciPy", "scipy", extra=scipy_smoke)

# =========================================
# General
# =========================================

# GUDHI
check("GUDHI", "gudhi")

# joblib
def joblib_smoke(joblib):
    from joblib import Parallel, delayed
    out = Parallel(n_jobs=1)(delayed(lambda x: x + 1)(3) for _ in range(1))
    assert out == [4]
check("joblib", "joblib", extra=joblib_smoke)

# pandas
def pandas_smoke(pd):
    df = pd.DataFrame({"a":[1,2], "b":[3,4]})
    assert df.shape == (2,2)
check("pandas", "pandas", extra=pandas_smoke)

# Biopython
def biopython_smoke(Bio):
    from Bio.Seq import Seq
    s = Seq("ATGC")
    assert str(s.reverse_complement()) == "GCAT"
check("Biopython", "Bio", extra=biopython_smoke)

# MDTraj
def mdtraj_smoke(md):
    from mdtraj.core.topology import Topology
    _ = Topology()
check("MDTraj", "mdtraj", extra=mdtraj_smoke)

PY

