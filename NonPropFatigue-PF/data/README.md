# Data

Download the data set of Hengeveld et al.

S.T. Hengeveld, D. Leonetti, H.H. Snijder, J. Maljaars, Zenodo, record 15807377, https://zenodo.org/records/15807377

and unpack its `supplementaryMaterial` folder to

```
data/zenodo_15807377/supplementaryMaterial
```

The cruciform outlines are read from the Abaqus input files of this data set:

```
python scripts/extract_geometry.py data/zenodo_15807377/supplementaryMaterial/Experiment_II/job-036-Wolf2019-LoadingTypeB/job-036-Wolf2019-LoadingTypeB_01.inp data/wolf_geom.npz
python scripts/extract_geometry.py data/zenodo_15807377/supplementaryMaterial/Experiment_A/job-041-FremyProportional/job-041-FremyProportional_01.inp data/fremy_geom.npz
```

The tube and two-crack models need no external data.
