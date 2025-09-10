python -c "
import importlib

# Define libraries to check
list_of_libs = [
    'numpy',
    'torch',
    'torch_geometric',
    'pytorch_lightning',
    'torch_scatter',
    'torch_sparse',
    'torch_cluster',
    'torch-spline-conv',
    'torchtriton'
]

# Define optional (not needed) libraries
optional_libs = [
    'torch_scatter',
    'torch_sparse',
    'torch_cluster',
    'torch-spline-conv',
    'torchtriton'
]

print(f'\nMac Pytorch dependent libraries tests:')

for lib in list_of_libs:
    if lib in optional_libs:
        try:
            module = importlib.import_module(lib)
            print(f'{lib}: {module.__version__} (installed, but not needed)')
        except ImportError:
            print(f'{lib}: not installed (not needed)')
    else:
        try:
            module = importlib.import_module(lib)
            print(f'{lib}: {module.__version__}')
        except ImportError:
            print(f'{lib}: not installed')
"

