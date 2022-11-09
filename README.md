# als_microct-recon

## Description

Jupyter notebooks to reconstruct and visualize microCT data from ALS beamline 8.3.2

## Getting Started
On NERSC, no setup or installation is required. Just log into https://jupyter.nersc.gov/ and use "microCT" nodes, then follow the instructions in "README_to_get_notebooks.txt" (if you don't see microCT-specific nodes, make sure the beamline scientist adds you to the NERSC microCT group).

### Installing on local machine

* Install git (eg. https://git-scm.com/downloads) and python/conda (eg. https://docs.conda.io/en/latest/miniconda.html)


* Clone repo
```
git clone git@github.com:perlmutter/battery_microct.git
```

* Create environment
```
cd als_microct-recon
conda env create -f environment.yml
```
if conda cannot solve environment, check that channel priority is flexible 
```
conda config --get channel_priority
conda config --set channel_priority flexible
```

* Create kernel
```
conda activate als832
ipython kernel install --user --name=als832
```

* You should now be able to run launch Jupyter lab and run ALS_recon.ipynb with the als832 kernel
```
jupyter lab
```


## Authors

David Perlmutter (dperl@lbl.gov)
Lipi Gupta (lipigupta@lbl.gov)
Dula Parkinson (dyparkinson@lbl.gov)

## GPL v3 License

als microct recon (microct) Copyright (c) 2022, The Regents of the
University of California, through Lawrence Berkeley National Laboratory
(subject to receipt of any required approvals from the U.S. Dept. of
Energy). All rights reserved.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

## Acknowledgments
