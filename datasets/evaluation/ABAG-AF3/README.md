# ABAG-Docking Benchmark AF3 (ABAG-AF3) Dataset
## Overview

We selected proteins released after 2022 from ABAG-Docking Benchmark as targets and used AlphaFold3 to generate 25 conformations per target, running it five times with different seeds. The ABAG-AF3 dataset consists of 35 targets and 875 conformations.

## Directory Structure
```bash
dataset/
├── README.md                 # Dataset description file
├── ABAG-AF3_structures/
│   ├── protein_1.pdb         # Protein structure file (PDB format)
│   ├── protein_2.pdb         # Protein structure file
│   └── ...
├── native_structures/
│   ├── protein_1.pdb         # Protein structure file (PDB format)
│   ├── protein_2.pdb         # Protein structure file
│   └── ...
└── label.txt              # DockQ-wave and QS-score for protein conformations
```

## Citation
If you think our work is helpful, please cite our work by:

```
@article{han2024topoqa,
  title={TopoQA: a topological deep learning-based approach for protein complex structure interface quality assessment},
  author={Han, Bingqing and Zhang, Yipeng and Li, Longlong and Gong, Xinqi and Xia, Kelin},
  journal={arXiv preprint arXiv:2410.17815},
  year={2024}
}
```