# verify Pytorch and MPS libraries
python - << 'PY'
import torch

print(f'\nMac Pytorch tests:')
print("PyTorch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())
PY

# verify dependent libraries
zsh tests/mac_test_pytorch.zsh
