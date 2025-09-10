# Python dependencies to run TopoQA Inference

# python 3.10.11

python -m pip install --upgrade pip wheel setuptools

# ARM issue - must install this first - fails if done bulk
pip install torch==2.2.2

# ARM issue - install next
pip install torch_geometric==2.5.3

# Require Mac
pip install pytorch-lightning

# Fix Numpy mess
# Downgrade to 1.26.4
pip install --upgrade 'numpy<2'  # e.g., 1.26.4

# sanity checks
# torch, geometric, lightening - required on Mac
# torchtriton, scatter, sparse, cluster, spline - not required on Mac
zsh tests/mac_test_pytorch.zsh 

# install rest of python libraries
pip install -r py_requirements/requirements_macos_mps.txt

# install mkdssp
zsh mac_requirements/install_dssp.zsh
