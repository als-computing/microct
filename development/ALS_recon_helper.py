import os
import numpy as np
import time
from scipy.fft import fft, ifft, fftfreq, fftshift
from skimage import transform, io
import tomopy
import astra
import svmbir
import dxchange
import pickle
import json
import base64
import requests
from authlib.integrations.requests_client import OAuth2Session
from authlib.oauth2.rfc7523 import PrivateKeyJWT
from reconstructionGPU import recon_setup, recon

def read_metadata(path):
    numslices = int(dxchange.read_hdf5(path, "/measurement/instrument/detector/dimension_y")[0])
    numrays = int(dxchange.read_hdf5(path, "/measurement/instrument/detector/dimension_x")[0])
    pxsize = dxchange.read_hdf5(path, "/measurement/instrument/detector/pixel_size")[0] / 10.0  # /10 to convert units from mm to cm
    numangles = int(dxchange.read_hdf5(path, "/process/acquisition/rotation/num_angles")[0])
    propagation_dist = dxchange.read_hdf5(path, "/measurement/instrument/camera_motor_stack/setup/camera_distance")[1]
    kev = dxchange.read_hdf5(path, "/measurement/instrument/monochromator/energy")[0] / 1000
    angularrange = dxchange.read_hdf5(path, "/process/acquisition/rotation/range")[0]
    filename = os.path.split(path)[-1]
    print(f'{filename}:')
    print(f'numslices: {numslices}, rays: {numrays}, numangles: {numangles}')
    print(f'angularrange: {angularrange}, \pxsize: {pxsize*10000} um, distance: {propagation_dist} mm. energy: {kev} keV')
    if kev>100:
        print('white light mode detected; energy is set to 30 kev for the phase retrieval function')
        
    return {'numslices': numslices,
            'numrays': numrays,
            'pxsize': pxsize,
            'numangles': numangles,
            'propagation_dist': propagation_dist,
            'kev': kev,
            'angularrange': angularrange}

def read_data(path, proj=None, sino=None, downsample_factor=None):
    tomo, flat, dark, angles = dxchange.exchange.read_aps_tomoscan_hdf5(path, proj=proj, sino=sino, dtype=np.float32)
    angles = angles[proj].squeeze()
    tomopy.normalize(tomo, flat, dark, out=tomo)
    tomopy.minus_log(tomo, out=tomo)    
    if downsample_factor and downsample_factor!=1:
        tomo = np.asarray([transform.downscale_local_mean(proj, (downsample_factor,downsample_factor), cval=0).astype(proj.dtype) for proj in tomo])

    return tomo, angles

def preprocess_tomo(tomo,
                    outlier_diff1D=None,outlier_size=3, # forremove_outlier1d, hardcoded from reconstruction.py
                    minimum_transmission = 0.01, # hardcoded from reconstruction.py
                    ringSigma=3,  # for remove stripe fw, hardcoded from reconstruction.py
                    ringLevel=8,  # for remove stripe fw, hardcoded from reconstruction.py
                    ringWavelet='db5'): # for remove stripe fw, hardcoded from reconstruction.py
    # remove_outlier1d 
    outlier_size3D=[outlier_size,1,1]
    tomo = tomo.astype(np.float32,copy=False)
    med = ndi.median_filter(tomo, size=outlier_size3D, mode='reflect')
    outliers = np.abs(tomo - med) > outlier_diff1D
    tomo[outliers] = med[outliers]
    # normalize
    tomopy.normalize(tomo, flat, dark, out=tomo)
    # minus_log 
    tomo[tomo < minimum_transmission] = minimum_transmission
    tomopy.minus_log(tomo, out=tomo)
    # remove_dtripe_fw
    tomo = tomopy.remove_stripe_fw(tomo, sigma=ringSigma, level=ringLevel, pad=True, wname=ringWavelet)
    return tomo

# def astra_recon_slice(tomo,angles,cor=None,algorithm='FBP'):
#     tomo = tomo.squeeze()
#     numrays = int(tomo.shape[1])
#     if cor is None: cor = 0
#     proj_geom = astra.create_proj_geom('parallel', 1, numrays, angles)
#     proj_geom = astra.functions.geom_postalignment(proj_geom, cor)
#     sino_id = astra.data2d.create('-sino', proj_geom, tomo)
#     # min_x, max_x, min_y, max_y = (-numrays/2+cor, numrays/2+cor, -numrays/2, numrays/2)
#     # print(min_x, max_x, min_y, max_y)
#     # vol_geom = astra.create_vol_geom(numrays, numrays, min_x, max_x, min_y, max_y)
#     vol_geom = astra.create_vol_geom(numrays, numrays)
#     rec_id = astra.data2d.create('-vol', vol_geom)
#     proj_id = astra.create_projector('strip', proj_geom, vol_geom) # can choose line (fastest?), or strip (most accurate?) or linear (??)

#     cfg = astra.astra_dict(algorithm) # FBP is the fastest algorithm (use FBP_CUDA if on GPU machine) 
#     cfg['ReconstructionDataId'] = rec_id
#     cfg['ProjectionDataId'] = sino_id
#     cfg['ProjectorId'] = proj_id

#     alg_id = astra.algorithm.create(cfg)
#     astra.algorithm.run(alg_id)
#     rec = astra.data2d.get(rec_id)
#     return rec

def svmbir_fbp(tomo,angles,cor=None):   
    fourier_filter = transform.radon_transform._get_fourier_filter(tomo.shape[2],'hann').squeeze()
    filtered_tomo = np.real(ifft( fft(tomo, axis=2) * fourier_filter, axis=2))
    if cor is None: cor = 0
    rec = svmbir.backproject(filtered_tomo, angles, geometry='parallel', center_offset=cor, num_threads=4, verbose=False)
    return rec

def shift_prjections(projs, COR):
    translateFunction = transform.SimilarityTransform(translation=(COR, 0))
    shifted = np.asarray([transform.warp(proj, translateFunction) for proj in projs]) # Apply translation with interpolation to projection[n]
    return shifted


def cache_svmbir_projector(img_size,num_angles,num_threads=None):

    for i,(sz,nang) in enumerate(zip(img_size,num_angles)):
        print(f"Starting size={sz[0]}x{sz[1]}")
        img = svmbir.phantom.gen_shepp_logan(sz[0],sz[1])[np.newaxis]
        angles = np.linspace(0,np.pi,nang)
        t0 = time.time()
        tomo = svmbir.project(img, angles, img.shape[2],
                              num_threads=num_threads,
                              verbose=0,
                              svmbir_lib_path='/global/cscratch1/sd/dperl/svmbir_cache')
        t = time.time() - t0
        print(f"Finisehd: time={t}")    
        
def normalize(x,lower=0,upper=100):
    # normalize image by grayscale percentiles
    x = x - np.percentile(x,lower)
    x = x / np.percentile(x,upper)
    x[x<0] = 0
    x[x>1] = 1
    return x

    
    