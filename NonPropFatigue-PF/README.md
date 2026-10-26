# NonPropFatigue-PF

Source code for the paper

P. Zhang, P. Cui, T. Dai, K. Tang. *A phase-field fatigue model with order-dependent crack-tip displacement steering for crack paths under non-proportional mixed-mode loading.* Submitted to International Journal of Fatigue.

The package `vctdpf` contains the 2D and 3D phase-field fatigue solvers (AT2, Amor split, cumulative fatigue variable, cycle jump), the adaptive mesh refinement, and the steering by the time-reversal odd part of the incremental crack-tip displacement. The scripts in `examples` set up the computations of the paper.

## Requirements

- Python 3.10 or later
- NVIDIA GPU with CUDA 12
- packages in `requirements.txt`

```
pip install -r requirements.txt
```

## Data

The experimental crack paths and the specimen meshes of Wolf et al. and Fremy et al. are taken from the data set of Hengeveld et al. (Zenodo, record 15807377). See `data/README.md`.

## Usage

```
python examples/biaxial.py --exp {I,II,A,B,C} [--vctd-odd --vctd-Kc KC] [--ell L --h H] --out runs/NAME
python examples/two_cracks.py --phi PHI [--vctd-odd --vctd-Kc KC] --out runs/NAME
python examples/tube2d.py --case CASE [--vctd-Kc KC] --out runs/NAME
python examples/tube3d.py --case CASE [--vctd-Kc KC] --out runs/NAME
```

`biaxial.py` covers the slotted cruciform of Wolf et al. (Experiments I and II) and the cruciform of Fremy et al. (A in-phase, B square, C cross). `KC` is the steering parameter in MPa mm^0.5; without `--vctd-Kc` the isotropic fatigue phase field is run. Material and numerical parameters are listed in the paper. Each run writes `history.csv`, the tracked tip path and damage snapshots to its output folder.

## License

MIT, see `LICENSE`.
