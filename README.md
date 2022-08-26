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

* Create kernel
```
conda activate als832
ipython kernel install --user --name=als832
```

* You should now be able to run ALS_recon.ipynb with the als832 kernel from JupyterLab

## Authors

David Perlmutter (dperl@lbl.gov)
Lipi Gupta (lipigupta@lbl.gov)
Dula Parkinson (dyparkinson@lbl.gov)

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments
