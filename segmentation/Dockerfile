# Most of this is from: https://gitlab.com/NERSC/nersc-official-images/-/blob/main/nersc/python/3.9-anaconda-2021.11/Dockerfile
FROM docker.io/library/ubuntu:latest
WORKDIR /opt

RUN \
    apt-get update && apt-get install --yes \
    build-essential \
    gfortran \
    git \
    wget \
    libgl1 libegl1 libxext6 libsm6 libxrender1 && \
    apt-get clean all && rm -rf /var/lib/apt/lists/*

#miniforge
# Download and install the latest Miniforge (comes with Mamba and conda-forge)
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/miniconda && \
    rm /tmp/miniforge.sh

# Update PATH environment variable to include Miniforge
ENV PATH="/opt/miniconda/bin:$PATH"


# Install base packages from microct + dependencies for vtk_trial.ipynb
# Added: opencv, matplotlib, vtk, pytorch stack
RUN mamba install --yes -c conda-forge -c astra-toolbox -c simpleitk -c pytorch -c nvidia \
    python=3.10 \
    jupyterlab astropy cartopy cython cfitsio "dask[distributed]" scipy scikit-learn scikit-image numba h5py joblib pandas statsmodels _libgcc_mutex \
    astra-toolbox tifffile tomopy dxchange svmbir itk simpleitk pyfftw natsort olefile tqdm zarr \
    opencv matplotlib vtk pytorch torchvision torchaudio pytorch-cuda=12.4 \
    && mamba clean --all -f -y

# Install Detectron2 (requires pytorch)
RUN python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

# # install SYRIS
# COPY syris syris
# # RUN git clone git@github.com:ufo-kit/syris.git
# RUN conda install -c conda-forge -y cmake pybind11 -- the NERSC docs say something about not using cmake
# RUN cd syris && pip install -r requirements.txt && pip install .
# RUN rm -fR syris


# for NERSC jupyterhub environment
RUN pip install batchspawner
ENV NERSC_JUPYTER_IMAGE=YES

# VTK/EGL Environment Configuration for Headless Rendering
ENV VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN=1
ENV VTK_OPENGL_HAS_EGL=1
ENV VTK_USE_OFFSCREEN_EGL=1
ENV LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH}
ENV __GLX_VENDOR_LIBRARY_NAME=nvidia
ENV __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_nvidia.json
ENV VTK_DEFAULT_EGL_DEVICE=0



# expected by entrypoint script
RUN mkdir -p /alsuser /alsdata

WORKDIR /alsuser
