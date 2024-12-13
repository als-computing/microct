# Most of this is from: https://gitlab.com/NERSC/nersc-official-images/-/blob/main/nersc/python/3.9-anaconda-2021.11/Dockerfile
FROM docker.io/library/ubuntu:latest
WORKDIR /opt

RUN \
    apt-get update && apt-get install --yes \
    build-essential \
    gfortran \
    git \
    wget && \
    apt-get clean all && rm -rf /var/lib/apt/lists/*

#miniconda
# Download and install the latest Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/miniconda && \
    rm /tmp/miniconda.sh

# Update PATH environment variable to include Miniconda
ENV PATH="/opt/miniconda/bin:$PATH"

#mpich from source. python installation needs to come first.
ARG mpich=4.0
ARG mpich_prefix=mpich-$mpich
RUN \
    wget https://www.mpich.org/static/downloads/$mpich/$mpich_prefix.tar.gz && \
    tar xvzf $mpich_prefix.tar.gz                                           && \
    cd $mpich_prefix                                                        && \
    export FCFLAGS=-fallow-argument-mismatch && \
    export FFLAGS=-fallow-argument-mismatch && \
    ./configure  && \
    make -j 16                                                              && \
    make install                                                            && \
    make clean                                                              && \
    cd ..                                                                   && \
    rm -rf $mpich_prefix
RUN /sbin/ldconfig

RUN conda install --yes -c conda-forge python=3.10 mamba \
    && mamba install --yes -c conda-forge \
       mamba astropy cartopy cython cfitsio "dask[distributed]" scipy scikit-learn scikit-image numba h5py joblib pandas statsmodels _libgcc_mutex \
    && mamba install --yes -c conda-forge -c astra-toolbox -c simpleitk \
       mamba astra-toolbox tifffile tomopy dxchange svmbir itk simpleitk pyfftw natsort olefile tqdm zarr

# # install SYRIS
# COPY syris syris
# # RUN git clone git@github.com:ufo-kit/syris.git
# RUN conda install -c conda-forge -y cmake pybind11 -- the NERSC docs say something about not using cmake
# RUN cd syris && pip install -r requirements.txt && pip install .
# RUN rm -fR syris


# for NERSC jupyterhub environment
RUN pip install batchspawner
ENV NERSC_JUPYTER_IMAGE=YES

#build mpi4py on top of our mpich
# David's note: It's important to build mpi4py towards the end, after the computational packages. Initially
#I had installed mpi4py earlier and the Dockerfile build fine, but then mpi4py doesn't work properly on NERSC --
#it will run, but every node will think it's rank 0, so will do the same thing. Never figured out exactly what was
#the culprit, but somehow one of the other packages (astra? tomopy?) is overwriting the fragile way that mpi4py
#must be installed on NERSC. Anyway, this seems to work. See ticket INC0193198 for a few more details.
#
RUN python -m pip install mpi4py
#RUN conda install -c conda-forge mpi4py --no-deps

#ENV HDF5_USE_FILE_LOCKING="FALSE"

#installing h5py for mpi - from h5py parallel read documentation and NERSC's building h5py from source
ENV CC=cc
ENV HDF5_MPI="ON"
RUN pip install --no-binary=h5py h5py
RUN pip install ngff_zarr dask_image numpy==1.23.2 
# RUN pip install -v --force-reinstall --no-cache-dir --no-binary=h5py --no-build-isolation --no-deps h5py


# expected by entrypoint script
RUN mkdir -p /alsuser /alsdata

WORKDIR /alsuser
