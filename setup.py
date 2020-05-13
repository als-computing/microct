from distutils.core import setup

setup(
    name='alsmicroct',
    version='0.2dev',
    author='H.S.Barnard, D.Y.Parkinson',
    packages=['alsmicroct',],
    license='LICENSE',
    long_description=open('README.md', 'r').read(),
#	requirements=['numpy','h5py','scipy'
# conda: 'numpy','h5py','scipy','numexpr','scikit-image','xlrd'
# dxchange: conda install -c dgursoy dxchange

# conda install numpy h5py scipy numexpr scikit-image xlrd setuptools
# conda install -c dgursoy dxchange
# conda install -c als832 pyf3d
)
