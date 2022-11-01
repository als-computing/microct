"""
ALS_recon_functions.py
Modularized functions used throughout the various notebooks. Mostly for processing, though some are for plotting
Functions are in rough order of when they are called in ALS_recon.ipynb
"""

import sys
import os
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import scipy.signal as signal
from scipy.fft import fft, ifft, fftfreq, fftshift
from skimage import transform, io
import tomopy
import astra
import dxchange
import importlib
# checks if svmbir is installed before importing (so users who install locally aren't required to install svmbir if they won't use it)
svmbir_spec = importlib.util.find_spec("svmbir")
if svmbir_spec is not None: # this 
    import svmbir

def check_for_gpu(verbose = False):
    """ Checks if GPU can be used for reconstruction """
    try:
        subprocess.check_output('nvidia-smi')
        if verbose:
            print('Nvidia GPU detected, will use to reconstruct!')
        return True
    except Exception: # if command not found, cant talk to GPU (or doesnt exists)
        print('No Nvidia GPU in system, will use CPU')
        return False

def get_directory_filelist(path,max_num=10000, verbose = False):
    """ Copied from Dula's legacy notebook. Prints files in directory, no fancy widget. Not currently used """
    filenamelist = os.listdir(path)
    filenamelist = [x for x in filenamelist if os.path.isfile(os.path.join(path,x))]
    filenamelist.sort()
    sorted_file_names = []
    for i in range(len(filenamelist)-1,np.maximum(len(filenamelist)-max_num,-1),-1):
        if verbose:
            print(f'{i}: {filenamelist[i]}')
        sorted_file_names.append(f'{i}: {filenamelist[i]}')
    return filenamelist, sorted_file_names

def read_metadata(path,print_flag=True):
    """ Reads metadata (slices, rays, etc) from APS tomoscan hdf5 format and returns in dictionary
        path: full path to .h5 file
        print_flag: whether to print metadata to screen
    """
    numslices = int(dxchange.read_hdf5(path, "/measurement/instrument/detector/dimension_y")[0])
    numrays = int(dxchange.read_hdf5(path, "/measurement/instrument/detector/dimension_x")[0])
    pxsize = dxchange.read_hdf5(path, "/measurement/instrument/detector/pixel_size")[0] / 10.0  # /10 to convert units from mm to cm
    numangles = int(dxchange.read_hdf5(path, "/process/acquisition/rotation/num_angles")[0])
    propagation_dist = dxchange.read_hdf5(path, "/measurement/instrument/camera_motor_stack/setup/camera_distance")[1]
    kev = dxchange.read_hdf5(path, "/measurement/instrument/monochromator/energy")[0] / 1000
    angularrange = dxchange.read_hdf5(path, "/process/acquisition/rotation/range")[0]
    filename = os.path.split(path)[-1]
    
    if print_flag:
        print(f'{filename}:')
        print(f'numslices: {numslices}, rays: {numrays}, numangles: {numangles}')
        print(f'angularrange: {angularrange}, pxsize: {pxsize*10000} um, distance: {propagation_dist} mm. energy: {kev} keV')
        if kev>100:
            print('white light mode detected; energy is set to 30 kev for the phase retrieval function')
          
    return {'numslices': numslices,
            'numrays': numrays,
            'pxsize': pxsize,
            'numangles': numangles,
            'propagation_dist': propagation_dist,
            'kev': kev,
            'angularrange': angularrange}

def read_data(path, proj=None, sino=None, downsample_factor=None, prelog=False,
              preprocess_settings={'minimum_transmission':0.01}, postprocess_settings=None, **kwargs):
    """ Reads projetion data gets prepares for reconstruction (ie normalizes, takes log, filters, etc).
        Assumes APS tomoscan hdf5 format (see here: https://dxchange.readthedocs.io/en/latest/source/api/dxchange.exchange.html#)
        path: full path to .h5 file
        proj: which projections to read (first,last,step). None means all projections.
        sino: which slices to read (first,last,step). None means all slices.
        downsample_factor: Integer downsampling of projection images using local pixel averaging. None (or 1) means no downsampling 
        prelog: if True, don't take log
        preprocess_settings: dictionary of parameters used to process projections BEFORE log (see prelog_process_tomo)
        postprocess_settings: dictionary of parameters used to process projections AFTER log (see postlog_process_tomo)
    """
    metadata = read_metadata(path,print_flag=False)

    tomo, flat, dark, angles = dxchange.exchange.read_aps_tomoscan_hdf5(path, proj=proj, sino=sino, dtype=np.float32)
    angles = angles[proj].squeeze()
    tomopy.normalize(tomo, flat, dark, out=tomo)
        
    if preprocess_settings:
        tomo = prelog_process_tomo(tomo, preprocess_settings)
    if prelog:
        # downsampling pre-log can lead to bright halo in recon with radius = nrays -- may need to mask recon
        if downsample_factor and downsample_factor!=1:
            tomo = np.asarray([transform.downscale_local_mean(proj, (downsample_factor,downsample_factor), cval=0).astype(proj.dtype) for proj in tomo])
        return tomo, angles
    # take log
    tomopy.minus_log(tomo, out=tomo)
    # To Do: safety check for Inf/NaN pixels after log?
    # downsampling post-log is better
    if downsample_factor and downsample_factor!=1:
        tomo = np.asarray([transform.downscale_local_mean(proj, (downsample_factor,downsample_factor), cval=0).astype(proj.dtype) for proj in tomo])
    if postprocess_settings: # putting after downsample for efficiency, but could put before too 
        tomo = postlog_process_tomo(tomo, postprocess_settings)
    return tomo, angles

def prelog_process_tomo(tomo, args):
    """ Apply processing steps to PROJECTIONS (not sinograms) before log. Can make this list as long as you want. """
    # sarepy ring removal (combo of 3 methods, see: https://sarepy.readthedocs.io/toc/section3_1/section3_1_6.html)
    # "small stripe" method relies on median filter along angle dimension (after sorting) 
    if 'sm_size' in args and args['sm_size']:
        tomo = tomopy.remove_all_stripe(tomo,snr=args['snr'], la_size=args['la_size'], sm_size=args['sm_size'])

    # 1D median filter along angle dimension, to remove outliers 
    if 'outlier_diff_1D' in args and args['outlier_diff_1D']:
        # currently hardcoded to filter along angle dimension
        tomopy.misc.corr.remove_outlier1d(tomo, args['outlier_diff_1D'], size=args['outlier_size_1D'], axis=0, out=tomo)
        
    # 2D median filter on each projection (ie, perpendicular to angle), to remove outliers 
    if 'outlier_diff_2D' in args and args['outlier_diff_2D']:
        # currently hardcoded to filter along
        tomopy.misc.corr.remove_outlier(tomo, args['outlier_diff_2D'], size=args['outlier_size_2D'], axis=0, out=tomo)

    # threshold low measurements
    if 'minimum_transmission' in args and args['minimum_transmission']:
        tomo[tomo < args['minimum_transmission'] ] = args['minimum_transmission']
    return tomo

def postlog_process_tomo(tomo, args):
    """ Apply processing steps to PROJECTIONS (not sinograms) after log. Can make this list as long as you want. """
    # wavelet filter to remove rings (stripes in sinogram)
    if 'ringSigma' in args and args['ringSigma']:
        tomo = tomopy.remove_stripe_fw(tomo, sigma=args['ringSigma'], level=args['ringLevel'], pad=False, wname='db5')
    
    return tomo

# this is the default processing done in Dula's reconstruction.py
def preprocess_tomo_orig(tomo, flat, dark,
                    outlier_diff1D=750, # difference between good data and outlier data (outlier removal) 
                    outlier_size1D=3, # for remove_outlier1d, hardcoded from reconstruction.py
                    minimum_transmission = 0.01, # from reconstruction.py
                    ringSigma=3,  # for remove stripe fw, from reconstruction.py
                    ringLevel=8,  # for remove stripe fw, from reconstruction.py
                    ringWavelet='db5', # for remove stripe fw, from reconstruction.py
                    **kwargs):
    """ Copied from Dula's legacy notebook. Processes projections AND takes log. Not currently used """

    # median filter -- WHY ACROSS ROTATION AXIS?
    tomopy.misc.corr.remove_outlier(tomo, outlier_diff, size=outlier_size, axis=0, ncore=None, out=tomo)
    # normalize with flat/dark, threshold transmission, and take negative log
    if minimum_transmission:
        tomo[tomo < minimum_transmission] = minimum_transmission
    tomopy.minus_log(tomo, out=tomo)
    # post-log, wavelet-based ring removal
    tomo = tomopy.remove_stripe_fw(tomo, sigma=ringSigma, level=ringLevel, pad=True, wname=ringWavelet)
    return tomo

def mask_recon(recon,r=None):
    """ Applies circular mask to image - all pixels outside radius set to zero.
        r: mask radius. None defaults to half of image width or height (whichever is larger)
    """
    assert recon.ndim in [2,3], f"Image dimensions must be 2 or 3, but got: {recon.ndim}"
    stack_flag = True if recon.ndim == 3 else False
    if not stack_flag:
        recon = np.expand_dims(recon,0)

    # Need to add this to remove bright halo
    x, y = np.arange(recon.shape[1]), np.arange(recon.shape[2])
    X,Y = np.meshgrid(x-x.mean(),y-y.mean(),indexing='ij')
    if r is None:
        r = np.maximum(recon.shape[1],recon.shape[2])/2
    recon[:,(X**2 + Y**2 > (r)**2)] = 0
    
    if not stack_flag:
        recon = recon.squeeze()
        
    return recon

def auto_find_cor(path):
    """ Reads first and last projection image and uses cross-correlation to estimate COR. Uses tomopy implementation.
        Note: COR converted to units of pixels FROM CENTER (ie perfectly centered projections have a COR=0). Tomopy uses pixels from edge. 
        path: full path to .h5 file    
    """
    metadata = read_metadata(path,print_flag=False)
    if metadata['angularrange'] > 300:
        lastcor = int(np.ceil(metadata['numangles'] / 2) )
    else:
        lastcor = metadata['numangles'] # This would take last projection

    tomo, _ = read_data(path, proj=slice(0,lastcor,lastcor-1),downsample_factor=None)
    cor = tomopy.find_center_pc(tomo[0], tomo[-1], tol=0.25)
    cor = cor - tomo.shape[2]/2
    return cor, tomo

def plot_0_and_180_proj_diff(first_proj,last_proj_flipped,init_cor=0,fignum=1,yshift=False,continuous_update=True):
    """ Creates projection overlay and shift sliders for manual COR selection
        first_proj: first projection image
        last_proj_flipped: flipped 180 degree projection
        init_cor: initial COR of plot. Usually chosen with auto_find_cor
        fignum: matplotlib figure number. Kind of irrelevant
        yshift: whether to allow up/down shift or not
        continuous_update: If True, slider will update on any movement (more responsive but potentially laggy). If false, will only update when slider is released.
        
        Returns:
        axs: matplotib axis handle to plot
        img: matplotlib image handle to image
        ui: ipywidgets handle to slider functionality
        sliders: ipywidgets handle to sliders
    """
    
    if plt.fignum_exists(num=fignum): plt.close(fignum)
    fig, axs = plt.subplots(num=fignum)
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    shifted_last_proj = shift_projections(last_proj_flipped, 2*init_cor, yshift=0)
    img = axs.imshow(first_proj - shifted_last_proj, cmap='gray',vmin=-.1,vmax=.1)
    plt.tight_layout()

    slider_dx = widgets.FloatSlider(description='Shift X', readout=True, min=-800, max=800, step=0.25, value=init_cor, layout=widgets.Layout(width='50%'),continuous_update=continuous_update)
    slider_dy = widgets.FloatSlider(description='Shift Y', readout=True, min=-800, max=800, step=0.25, value=0, layout=widgets.Layout(width='50%'),continuous_update=continuous_update)
    # only show yshift slider if flag is True
    if yshift:
        ui = widgets.VBox([slider_dx, slider_dy])
        axs.set_title(f"COR: 0, y_shift: 0")
    else:
        ui = widgets.VBox([slider_dx])
        axs.set_title(f"COR: 0")

    sliders = widgets.interactive_output(shift_proj_difference,{'dx':slider_dx,'dy':slider_dy,
                                         'img':widgets.fixed(img),'axs':widgets.fixed(axs),
                                         'first_proj':widgets.fixed(first_proj),'last_proj_flipped':widgets.fixed(last_proj_flipped)})
    
    return axs, img, ui, sliders

def shift_projections(projs, COR, yshift=0):
    """ Applies tranlation to image using scikit-image. Used for manual COR finding.
        projs: 2D projection images to translate (can be one or multiple in a stack)
        COR: x shift to apply
        y_shift: y shift to apply
    """
    translateFunction = transform.SimilarityTransform(translation=(-COR, yshift))
    if projs.ndim == 2:
        shifted = transform.warp(projs, translateFunction)
    elif projs.ndim == 3:
        # Apply translation with interpolation to projection[n]
        shifted = np.asarray([transform.warp(proj, translateFunction) for proj in projs])
    else:
        print('projs must be 2D or 3D')
        return
    return shifted

def astra_fbp_recon(tomo,angles,COR=0,fc=1,gpu=False,**kwargs):
    """ Filtred backprojection reconstruction using 2D Astra backprojection operator (ie slice by slice).
        tomo: sinogram(s) to reconstuct. 3D numpy array (angles,slices,rays)
        angles: projection angles, in radians 
        COR: center of rotation, in pixels from center of image
        fc: normalized LP filter cutoff (1 = no LP filter, 0 = filter everything)
        gpu: whether to use Astra GPU or CPU implementation
    """
    if fc != 1:
        # Apply 1D LP filter to sinogram along ray dimension
        N = np.minimum(100,tomo.shape[2])
        lpf = signal.firwin(N,fc) # time domain filter taps
        _, LPF = np.abs(signal.freqz(lpf,a=1,worN=tomo.shape[2],whole=True)) # freq domain filter, part 1 (abs keeps filter zero phase -- no pixel shift)
        tomo = np.real(ifft( fft(tomo, axis=2) * LPF, axis=2)) # apply filter in freq domain, part 2
        # tomo = signal.filtfilt(b,1,tomo,axis=2) # apply filter in time domain. Note: filtfilt ensures no pixel shift, but overfilters a little (ie fc is not technically accurate)
    
    if gpu:
        rec = tomopy.recon(tomo, angles,
                           center=COR + tomo.shape[2]/2,
                           algorithm=tomopy.astra,
                           options={'method':"FBP_CUDA", 'proj_type':'cuda'})
    else:
        rec = tomopy.recon(tomo, angles,
                           center=COR + tomo.shape[2]/2,
                           algorithm=tomopy.astra,
                           options={'method':"FBP", 'proj_type':'linear'})
    return rec

def astra_cgls_recon(tomo,angles,COR=0,num_iter=20,gpu=False,**kwargs):
    """ Conjugate gradient least squares reconstruction using 2D Astra backprojection operator (ie slice by slice).
        tomo: sinogram(s) to reconstuct. 3D numpy array (angles,slices,rays)
        angles: projection angles, in radians 
        COR: center of rotation, in pixels from center of image
        num_iter: how many iterations to perform
        gpu: whether to use Astra GPU or CPU implementation
    """    
    if gpu:
        rec = tomopy.recon(tomo, angles,
                           center=COR + tomo.shape[2]/2,
                           algorithm=tomopy.astra,
                           options={'method':"CGLS_CUDA", 'proj_type':'cuda', 'num_iter': num_iter})
    else:
        rec = tomopy.recon(tomo, angles,
                           center=COR + tomo.shape[2]/2,
                           algorithm=tomopy.astra,
                           options={'method':"CGLS", 'proj_type':'linear', 'num_iter': num_iter})
    return rec

def astra_fbp_recon_3d(tomo,angles_or_vectors,vectors=False,COR=0,fc=1):
    """ Filtered back projection reconstruction using 3D Astra backprojection operator.
        Useful if you want to define custom projection geometry in case slices are not truly separable (eg. misalignment).
        Requires more memory than slice-by-slice reconstruction (ie whole sinogram stack) so may need to downsample data. Not sure how it affects speed. 
        Note: Astra 3D projectors only work on GPU, so this requires GPU 
        
        tomo: sinogram(s) to reconstuct. 3D numpy array (angles,slices,rays)
        angles_or_vectors: EITHER an array of projection angles (same as normal) or a list of custom projection vectors produced by astra_projection_wrapper
        vectors: If True, vector list was given. If False, a standard array of angles were given.
        COR: center of rotation, in pixels from center of image
        fc: normalized LP filter cutoff (1 = no LP filter, 0 = filter everything)
    """        
    numslices = tomo.shape[1]
    numrays = tomo.shape[2]
    if vectors: # vectors created with astra_projection_wrapper
        vectors = angles_or_vectors
        proj_geom = astra.create_proj_geom('parallel3d_vec', numslices, numrays, vectors)
    else: # just angles, not vectors
        angles = -angles_or_vectors # need to be negative to match tomopy
        proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, numslices, numrays, angles)
        proj_geom = astra.geom_postalignment(proj_geom, [-COR])
    
    # filtered
    ramp_filter_freq_domain = transform.radon_transform._get_fourier_filter(tomo.shape[2],'None').squeeze()
    if fc != 1:
        N = np.minimum(100,tomo.shape[2])
        lpf = signal.firwin(N,fc) # time domain filter taps
        _, LPF = np.abs(signal.freqz(lpf,a=1,worN=tomo.shape[2],whole=True)) # zero-phase freq domain filter
        ramp_filter_freq_domain *= LPF
    tomo = np.real(ifft( fft(tomo, axis=2) * ramp_filter_freq_domain, axis=2))

    # backprojection
    cfg = astra.astra_dict('BP3D_CUDA')    
    vol_geom = astra.create_vol_geom(numrays,numrays,numslices)
    rec_id = astra.data3d.create('-vol', vol_geom)
    proj_id = astra.data3d.create('-proj3d', proj_geom, tomo.transpose(1,0,2))
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id    
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    rec = astra.data3d.get(rec_id)
    return rec

def astra_cgls_recon_3d(tomo,angles_or_vectors,vectors=False,COR=0,num_iter=20):
    """ Conjugate gradient least squares reconstruction using 3D Astra backprojection operator.
        Useful if you want to define custom projection geometry in case slices are not truly separable (eg. misalignment).
        Requires more memory than slice-by-slice reconstruction (ie whole sinogram stack) so may need to downsample data. Not sure how it affects speed. 
        Note: Astra 3D projectors only work on GPU, so this requires GPU 
        
        tomo: sinogram(s) to reconstuct. 3D numpy array (angles,slices,rays)
        angles_or_vectors: EITHER an array of projection angles (same as normal) or a list of custom projection vectors produced by astra_projection_wrapper
        vectors: If True, vector list was given. If False, a standard array of angles were given.
        COR: center of rotation, in pixels from center of image
        num_iter: how many iterations to perform
    """            
    numslices = tomo.shape[1]
    numrays = tomo.shape[2]
    if vectors: # vectors created with astra_projection_wrapper
        vectors = angles_or_vectors
        proj_geom = astra.create_proj_geom('parallel3d_vec', numslices, numrays, vectors)
    else: # just angles, not vectors
        angles = -angles_or_vectors # need to be negative to match tomopy
        proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, numslices, numrays, angles)
        proj_geom = astra.geom_postalignment(proj_geom, [-COR])
    
    cfg = astra.astra_dict('CGLS3D_CUDA')        
    vol_geom = astra.create_vol_geom(numrays,numrays,numslices)
    rec_id = astra.data3d.create('-vol', vol_geom)
    proj_id = astra.data3d.create('-proj3d', proj_geom, tomo.transpose(1,0,2))
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id    
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id,num_iter)
    rec = astra.data3d.get(rec_id)
    return rec

def svmbir_recon(tomo,angles,COR=0,proj_downsample=1,p=1.2,q=2,T=0.1,sharpness=0,snr_dB=40.0,max_iter=100,init_image=None,num_threads=None):
    """ Super Voxel Model Based Image Reconstruction.       
        tomo: sinogram(s) to reconstuct. 3D numpy array (angles,slices,rays)
        angles: projection angles, in radians 
        COR: center of rotation, in pixels from center of image
        proj_downsample: Same as downsample_factor -- integer downsampling of projection images using local pixel averaging. None (or 1) means no downsampling 
        For other parameters, see SVMBIR documetnation (https://svmbir.readthedocs.io/en/latest/index.html)
    """
    if not proj_downsample: proj_downsample = 1
    if init_image is None:
        init_image = astra_fbp_recon(tomo,angles,COR=COR/proj_downsample,fc=0.5,gpu=check_for_gpu())
    tomo = shift_projections(tomo,-COR/proj_downsample) # must manually shift COR. Shifting SVMBIR projector requires recomputing system matrix
    recon = svmbir.recon(tomo,angles,
                              center_offset=0.0, # MUST BE ZERO TO AVOID VERY LONG COMPUTATION OF SYSTEM MATRIX
                              init_image=init_image, # init with fbp for faster convergence
                              T=T, q=q, p=p, sharpness=sharpness, snr_db=snr_dB,
                              positivity=False, # must be False due to phase contrast in ALS data
                              num_threads=num_threads,  
                              max_iterations=max_iter,
                              svmbir_lib_path=get_svmbir_cache_dir(), # must have access to this directory
                              verbose=0) # 0, 1 or 2
    recon = recon.transpose(0,2,1) # to match tomopy format
    return recon
      
def svmbir_fbp(tomo,angles,cor=0,num_threads=None):
    """ Use SVMBIR projector to do regular FBP reconstruction. Only really useful as a faster test of SVMBIR projectors.
        tomo: sinogram(s) to reconstuct. 3D numpy array (angles,slices,rays)
        angles: projection angles, in radians 
        cor: center of rotation, in pixels from center of image
        num_threads: How many CPU threads to use. None defaults to all available threads
    """
    # Using scikit-image implementation of ramp filter, but I've noticed it only works for an even number of rays? I'm sure there's better implementations out there.
    fourier_filter = transform.radon_transform._get_fourier_filter(tomo.shape[2],'ramp').squeeze()
    filtered_tomo = np.real(ifft( fft(tomo, axis=2) * fourier_filter, axis=2))
    rec = svmbir.backproject(filtered_tomo, angles,
                             geometry='parallel',
                             center_offset=cor,
                             num_threads=num_threads,
                             svmbir_lib_path=get_svmbir_cache_dir(),
                             verbose=False)
    return rec


def tomopy_gridrec_recon(tomo,angles,COR=0,fc=1,butterworth_order=2,**kwargs):
    """ Tomopy's gridrec reconstruction. Worst image quality but very fast. Useful for fast feedback on Cori shared CPU node
        tomo: sinogram(s) to reconstuct. 3D numpy array (angles,slices,rays)
        angles: projection angles, in radians 
        center: center of rotation, in pixels from center of image
        For other parameters see Tomopy documentation (https://tomopy.readthedocs.io/en/latest/ipynb/tomopy.html)
    """
    rec = tomopy.recon(tomo, angles,
                       center=COR + tomo.shape[2]/2,
                       algorithm='gridrec',
                       filter_name='butterworth',
                       filter_par=[fc, butterworth_order])                       
    return rec


def cache_svmbir_projector(num_rays,num_angles,num_threads=None):
    """ Creates a SVMBIR system matrix cache as quickly as possible by and automatically saves in location set by get_svmbir_cache_dir.
        Cache depends on both image size, projections angles (assumed evenly distributed from 0 to 180), and COR (assumed 0).
        
        num_rays: number of rays in a projection. Same as image size in each dimension.
        num_angles: Number of projection angles.
    """
    if os.environ.get('CLIB') =='CMD_LINE':
        import svmbir.interface_py_c as ci
    else:
        import svmbir.interface_cy_c as ci
    
    num_slices = 1
    num_rows = num_rays
    num_cols = num_rays

    angles = np.linspace(0,np.pi,num_angles)
    num_views = len(angles)
    num_channels = num_rays
    roi_radius = float(1.0 * max(num_rows, num_cols))/2.0

    print(f"Starting size: {num_rays}, num_angles: {num_angles}")
    t0 = time.time()
    paths, sinoparams, imgparams = ci._init_geometry(angles, center_offset=0.0,
                                                 geometry='parallel', dist_source_detector=0.0,
                                                 magnification=1.0,
                                                 num_channels=num_channels, num_views=num_views, num_slices=num_slices,
                                                 num_rows=num_rows, num_cols=num_cols,
                                                 delta_channel=1.0, delta_pixel=1.0,
                                                 roi_radius=roi_radius,
                                                 object_name='object',
                                                 svmbir_lib_path=get_svmbir_cache_dir(),
                                                 verbose=2)
    t = time.time() - t0
    print(f"Finisehd: time={t}")    
        
def get_svmbir_cache_dir():
    """ Sets location of SVMBIR system matrix cache. Must be accessible by all users, otherwise SVMBIR will take prohibitively long """
    return '//global/cfs/cdirs/als/users/tomography_notebooks/svmbir_cache'

def get_scratch_path():
    """ Gets path to user's scratch if on NERSC, otherwise returns current directory """
    scratch_echo = subprocess.check_output('echo $SCRATCH',shell=True).decode("utf-8")
    if "scratch" in scratch_echo: # on NERSC
        return scratch_echo[:-1]
    else: # not on NERSC
        return os.getcwd()
    
def sino_360_to_180(data, overlap=0, rotation='left'):
    """ Taken directly from Dula's legacy "reconstruction.py"
    Converts 0-360 degrees sinogram to a 0-180 sinogram.
    
    Parameters
    ----------
    data : ndarray
        Input 3D data.

    overlap : scalar, optional
        Overlapping number of pixels.

    rotation : string, optional
        Left if rotation center is close to the left of the
        field-of-view, right otherwise.

    Returns
    -------
    ndarray
    Output 3D data.
    """
    dx, dy, dz = data.shape
    lo = overlap//2
    ro = overlap - lo
    n = dx//2
    out = np.zeros((n, dy, 2*dz-overlap), dtype=data.dtype)
    if rotation == 'left':
        weights = (np.arange(overlap)+0.5)/overlap
        out[:, :, -dz+overlap:] = data[:n, :, overlap:]
        out[:, :, :dz-overlap] = data[n:2*n, :, overlap:][:, :, ::-1]
        out[:, :, dz-overlap:dz] = weights*data[:n, :, :overlap] + (weights*data[n:2*n, :, :overlap])[:, :, ::-1]
    elif rotation == 'right':
        weights = (np.arange(overlap)[::-1]+0.5)/overlap
        out[:, :, :dz-overlap] = data[:n, :, :-overlap]
        out[:, :, -dz+overlap:] = data[n:2*n, :, :-overlap][:, :, ::-1]
        out[:, :, dz-overlap:dz] = weights*data[:n, :, -overlap:] + (weights*data[n:2*n, :, -overlap:])[:, :, ::-1]
    return out

######### The functions below are used for ipywidgets calls #########
        
def plot_recon(recon,fignum=1,figsize=4,clims=None):
    """ Creates a plot and colorbar slider for 2D reconstruction. Not currently used """

    if clims is None:
        clims = [np.percentile(recon,1),np.percentile(recon,99)]
    if plt.fignum_exists(fignum): plt.close(fignum)
    fig = plt.figure(num=fignum,figsize=(figsize, figsize))
    axs = plt.gca()
    img = axs.imshow(recon[0],cmap='gray')    
    clim_slider = widgets.interactive(set_clim, img=widgets.fixed(img),
                                      clims=widgets.FloatRangeSlider(description='Color Scale', layout=widgets.Layout(width='50%'),
                                                                           min=recon.min(), max=recon.max(),
                                                                           step=(recon.max()-recon.min())/500, value=clims))

    return img, axs, clim_slider

def plot_recon_comparison(recon1,recon2,titles=['',''],fignum=1,figsize=4):
    """ Creates a side-by-side plot of two 2D reconstructions with a shared and colorbar slider. Only used in Astra/SVMIBR comparison """
    
    if plt.fignum_exists(fignum): plt.close(fignum)
    fig, axs = plt.subplots(1,2,num=fignum,figsize=(2*figsize,figsize),sharex=True,sharey=True)
    img = [None, None]
    img[0] = axs[0].imshow(recon1[0],cmap='gray')
    axs[0].set_title(titles[0])
    img[1] = axs[1].imshow(recon2[0],cmap='gray')
    axs[1].set_title(titles[1])
    plt.tight_layout()
   
    recon = np.concatenate((recon1,recon2))
    clims = [np.percentile(recon[0],1), np.percentile(recon[0],99)]
    clim_slider = widgets.interactive(set_clim, img=widgets.fixed(img),
                                  clims=widgets.FloatRangeSlider(description='Color Scale', layout=widgets.Layout(width='50%'),
                                                                       min=recon.min(), max=recon.max(),
                                                                       step=(recon.max()-recon.min())/500, value=clims))
    return axs, img, clim_slider

def set_proj(img,path,proj_num,hline_handles=None):
    """ Sets projection image to display. Used by projection plotting sliders
        img: matplotlib image handle(s)
        path: full path to .h5 file
        proj_num: projection number to display
        hline_handles: horizontal line handles to update (eg, on recon image). Not currently used
    """
    
    tomo, _ = read_data(path, proj=slice(proj_num,proj_num+1,1), downsample_factor=None, prelog=True)
    if not isinstance(img, list):
        img = [img]
    for im in img:
        im.set_data(tomo.squeeze()) 
    if hline_handles:
        if not isinstance(hline_handles, list):
            hline_handles = [hline_handles]
        for h in hline_handles:
            h.set_ydata([proj_num,proj_num])

def set_sino(img,path,sino_num,hline_handles=None):
    """ Sets sinogram slice to display. Used by sinogram plotting sliders
        img: matplotlib image handle(s)
        path: full path to .h5 file
        sino_num: slice number to display
        hline_handles: horizontal line handles to update (eg, on projection image). Not currently used
    """

    tomo, _ = read_data(path, sino=slice(sino_num,sino_num+1,1), downsample_factor=None, prelog=True)
    if not isinstance(img, list):
        img = [img]
    for im in img:
        im.set_data(tomo.squeeze())         
    if hline_handles:
        if not isinstance(hline_handles, list):
            hline_handles = [hline_handles]
        for h in hline_handles:        
            h.set_ydata([sino_num,sino_num])

def set_slice(img,recon,slice_num):
    """ Sets sinogram slice to display. Used by sinogram plotting sliders
        img: matplotlib image handle(s)
        recon: full 3D recon
        slice_num: slice number to display
    """
    if not isinstance(img, list):
        img = [img]
    for im in img:
        im.set_data(recon[slice_num])
            
def set_clim(img,clims):
    """ Sets grayscale limits on image. Used by colorscale sliders
        img: matplotlib image handle(s)
        clim: list of grayscale values: [min, max]
    """

    if not isinstance(img, list):
        img = [img]
    for im in img:
        im.set_clim(vmin=clims[0],vmax=clims[1])        
        

def shift_proj_difference(dx,dy,img,axs,first_proj,last_proj_flipped,downsample_factor=1):
    """ Updates overalyed 0/180 projection difference. Used by plot_0_and_180_proj_diff
        dx: x shift (COR)
        dy: y shift
        img: matplotlib image handle
        axs: matplotlib axes handle
        first_proj: first projection image
        last_proj_flipped: flipped 180 degree projection        
        downsample_factor: Integer downsampling of projection images using local pixel averaging. None (or 1) means no downsampling 
    """
    shifted_last_proj = shift_projections(last_proj_flipped, 2*dx, yshift=dy)
    img.set_data(first_proj - shifted_last_proj)
    axs.set_title(f"COR: {downsample_factor*dx}, y_shift: {downsample_factor*dy/2}")