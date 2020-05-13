from __future__ import print_function
import time
import h5py
import numpy as np
import numexpr as ne
import skimage.transform as st
import os
import sys
import scipy.ndimage.filters as snf
import concurrent.futures as cf
import warnings
import importlib
import xlrd # for importing excel spreadsheets
from ast import literal_eval # For converting string to tuple
import glob

try:
    import tomopy
    from tomopy.util import mproc
except:
    print("warning: tomopy is not available")

try:
    import dxchange
except:
    print("warning: dxchange is not available")
    
try:
    importlib.import_module('pyF3D')
    import pyF3D
except ImportError:
    print("Warning: pyF3D not available")

#run this from the command line:
#python tomopy832.py
#it requires a separate file, which contains at minimum a list of filenames
#on separate lines. Default name of this file is input832.txt, but you can use any
#filename and run from the commandline as
#python tomopy832.py yourinputfile.txt
#If desired, on each line (separated by spaces) you can
#include parameters to override the defaults. 
#to do this you need pairs, first the name of the variable, then the desired value
#For True/False, use 1/0.
#You can generate these input files in excel, in which case use tab-separated
#(or space separated). Some input overrides require multiple values,
#these should be comma-separated (with no spaces). Example is sinoused
#which would be e.g. 500,510,1 to get slices 500 through 509. For sinoused,
#you can use first value -1 and second value number of slices to get that number
#of slices from the middle of the stack. 
#an example of the contents of the input file look like this:

#20150820_162025_Mvesparium_20948-131_pieceA_10x_x00y00.h5     cor    1196    sinoused    "-1,10,1"    doPhaseRetrieval    0 outputFilename c1196.0
#20150820_162025_Mvesparium_20948-131_pieceA_10x_x00y00.h5     cor    1196.5    sinoused    "-1,10,1"    doPhaseRetrieval    0 outputFilename c1196.5

#this was generated in excel and saved as txt tab separated, so the quotes were
#added automatically by excel. Note also that for parameters expecting strings as
#input (outputFilename for example), the program will choke if you put in a number.

#if cor is not defined in the parameters file, automated cor detection will happen

#chunk_proj and chunk_sino handle memory management. If you are running out of memory, make one or both of those smaller.

slice_dir = {
'remove_outlier1d': 'sino',
'remove_outlier2d': 'proj',
'normalize_nf': 'sino',
'normalize': 'both',
'minus_log': 'both',
'beam_hardening': 'both',
'remove_stripe_fw': 'sino',
'remove_stripe_ti': 'sino',
'remove_stripe_sf': 'sino',
'do_360_to_180': 'sino',
'correcttilt': 'proj',
'phase_retrieval': 'proj',
'recon_mask': 'sino',
'polar_ring': 'sino',
'bilateral_filter': 'both',
'castTo8bit': 'both',
'write_output': 'both'
}

#to profile memory, uncomment the following line
#and then run program from command line as
#python -m memory_profiler tomopy832.py
#(you have to have memory_profiler installed)
#@profile
def recon(
    filename,
    inputPath = './',
    outputPath = None,
    outputFilename = None,
    doOutliers1D = False, # outlier removal in 1d (along sinogram columns)
    outlier_diff1D = 750, # difference between good data and outlier data (outlier removal)
    outlier_size1D = 3, # radius around each pixel to look for outliers (outlier removal)
    doOutliers2D = False, # outlier removal, standard 2d on each projection
    outlier_diff2D = 750, # difference between good data and outlier data (outlier removal)
    outlier_size2D = 3, # radius around each pixel to look for outliers (outlier removal)
    doFWringremoval = True,  # Fourier-wavelet ring removal
    doTIringremoval = False, # Titarenko ring removal
    doSFringremoval = False, # Smoothing filter ring removal
    ringSigma = 3, # damping parameter in Fourier space (Fourier-wavelet ring removal)
    ringLevel = 8, # number of wavelet transform levels (Fourier-wavelet ring removal)
    ringWavelet = 'db5', # type of wavelet filter (Fourier-wavelet ring removal)
    ringNBlock = 0, # used in Titarenko ring removal (doTIringremoval)
    ringAlpha = 1.5, # used in Titarenko ring removal (doTIringremoval)
    ringSize = 5, # used in smoothing filter ring removal (doSFringremoval)
    doPhaseRetrieval = False, # phase retrieval
    alphaReg = 0.0002, # smaller = smoother (used for phase retrieval)
    propagation_dist = 75, # sample-to-scintillator distance (phase retrieval)
    kev = 24, # energy level (phase retrieval)
    butterworth_cutoff = 0.25, #0.1 would be very smooth, 0.4 would be very grainy (reconstruction)
    butterworth_order = 2, # for reconstruction
    doTranslationCorrection = False, # correct for linear drift during scan
    xshift = 0, # undesired dx transation correction (from 0 degree to 180 degree proj)
    yshift = 0, # undesired dy transation correction (from 0 degree to 180 degree proj)
    doPolarRing = False, # ring removal
    Rarc=30, # min angle needed to be considered ring artifact (ring removal)
    Rmaxwidth=100, # max width of rings to be filtered (ring removal)
    Rtmax=3000.0, # max portion of image to filter (ring removal)
    Rthr=3000.0, # max value of offset due to ring artifact (ring removal)
    Rtmin=-3000.0, # min value of image to filter (ring removal)
    cor=None, # center of rotation (float). If not used then cor will be detected automatically
    corFunction = 'pc', # center of rotation function to use - can be 'pc', 'vo', or 'nm'
    voInd = None, # index of slice to use for cor search (vo)
    voSMin = -40, # min radius for searching in sinogram (vo)
    voSMax = 40, # max radius for searching in sinogram (vo)
    voSRad = 10, # search radius (vo)
    voStep = 0.5, # search step (vo)
    voRatio = 2.0, # ratio of field-of-view and object size (vo)
    voDrop = 20, # drop lines around vertical center of mask (vo)
    nmInd = None, # index of slice to use for cor search (nm)
    nmInit = None, # initial guess for center (nm)
    nmTol = 0.5, # desired sub-pixel accuracy (nm)
    nmMask = True, # if True, limits analysis to circular region (nm)
    nmRatio = 1.0, # ratio of radius of circular mask to edge of reconstructed image (nm)
    nmSinoOrder = False, # if True, analyzes in sinogram space. If False, analyzes in radiograph space
    use360to180 = False, # use 360 to 180 conversion
    doBilateralFilter = False, # if True, uses bilateral filter on image just before write step # NOTE: image will be converted to 8bit if it is not already
    bilateral_srad = 3, # spatial radius for bilateral filter (image will be converted to 8bit if not already)
    bilateral_rrad = 30, # range radius for bilateral filter (image will be converted to 8bit if not already)
    castTo8bit = False, # convert data to 8bit before writing
    cast8bit_min=-10, # min value if converting to 8bit
    cast8bit_max=30, # max value if converting to 8bit
    useNormalize_nf = False, # normalize based on background intensity (nf)
    chunk_proj = 100, # chunk size in projection direction
    chunk_sino = 100, # chunk size in sinogram direction
    npad = None, # amount to pad data before reconstruction
    projused = None, #should be slicing in projection dimension (start,end,step)
    sinoused = None, #should be sliceing in sinogram dimension (start,end,step). If first value is negative, it takes the number of slices from the second value in the middle of the stack.
    correcttilt = 0, #tilt dataset
    tiltcenter_slice = None, # tilt center (x direction)
    tiltcenter_det = None, # tilt center (y direction)
    angle_offset = 0, #this is the angle offset from our default (270) so that tomopy yields output in the same orientation as previous software (Octopus)
    anglelist = None, #if not set, will assume evenly spaced angles which will be calculated by the angular range and number of angles found in the file. if set to -1, will read individual angles from each image. alternatively, a list of angles can be passed.
    doBeamHardening = False, #turn on beam hardening correction, based on "Correction for beam hardening in computed tomography", Gabor Herman, 1979 Phys. Med. Biol. 24 81
    BeamHardeningCoefficients = None, #6 values, tomo = a0 + a1*tomo + a2*tomo^2 + a3*tomo^3 + a4*tomo^4 + a5*tomo^5
    projIgnoreList = None, #projections to be ignored in the reconstruction (for simplicity in the code, they will not be removed and will be processed as all other projections but will be set to zero absorption right before reconstruction.
    *args, **kwargs):
    
    start_time = time.time()
    print("Start {} at:".format(filename)+time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime()))
    
    outputPath = inputPath if outputPath is None else outputPath

    outputFilename = filename if outputFilename is None else outputFilename
    outputFilename = outputFilename.replace('.h5','')
    tempfilenames = [outputPath+'tmp0.h5',outputPath+'tmp1.h5']
    filenametowrite = outputPath+'/rec'+filename.strip(".h5")+'/'+outputFilename
    #filenametowrite = outputPath+'/rec'+filename+'/'+outputFilename
    
    print("cleaning up previous temp files", end="")
    for tmpfile in tempfilenames:
        try:
            os.remove(tmpfile)
        except OSError:
            pass
    
    print(", reading metadata")
    
    datafile = h5py.File(inputPath+filename, 'r')
    gdata = dict(dxchange.reader._find_dataset_group(datafile).attrs) 
    pxsize = float(gdata['pxsize'])/10 # /10 to convert units from mm to cm
    numslices = int(gdata['nslices'])
    numangles = int(gdata['nangles'])
    angularrange = float(gdata['arange'])
    numrays = int(gdata['nrays'])
    npad = int(np.ceil(numrays * np.sqrt(2)) - numrays)//2 if npad is None else npad
    projused = (0,numangles-1,1) if projused is None else projused

#    ndark = int(gdata['num_dark_fields'])
#    ind_dark = list(range(0, ndark))
#    group_dark = [numangles - 1]
    inter_bright = int(gdata['i0cycle'])
    nflat = int(gdata['num_bright_field'])
    ind_flat = list(range(0, nflat))
    if inter_bright > 0:
        group_flat = list(range(0, numangles, inter_bright))
        if group_flat[-1] != numangles - 1:
            group_flat.append(numangles - 1)
    elif inter_bright == 0:
        group_flat = [0, numangles - 1]
    else:
        group_flat = None
    ind_tomo = list(range(0, numangles))
    floc_independent = dxchange.reader._map_loc(ind_tomo, group_flat)        

    #figure out the angle list (a list of angles, one per projection image)
    dtemp = datafile[list(datafile.keys())[0]]
    fltemp = list(dtemp.keys())
    firstangle = float(dtemp[fltemp[0]].attrs.get('rot_angle',0))
    if anglelist is None:
        #the offset angle should offset from the angle of the first image, which is usually 0, but in the case of timbir data may not be.
        #we add the 270 to be inte same orientation as previous software used at bl832
        angle_offset = 270 + angle_offset - firstangle
        anglelist = tomopy.angles(numangles, angle_offset, angle_offset-angularrange)
    elif anglelist==-1:
        anglelist = np.zeros(shape=numangles)
        for icount in range(0,numangles):
            anglelist[icount] = np.pi/180*(270 + angle_offset - float(dtemp[fltemp[icount]].attrs['rot_angle']))
            
    #if projused is different than default, need to chnage numangles and angularrange
    
    #can't do useNormalize_nf and doOutliers2D at the same time, or doOutliers2D and doOutliers1D at the same time, b/c of the way we chunk, for now just disable that
    if useNormalize_nf==True and doOutliers2D==True:
        useNormalize_nf = False
        print("we cannot currently do useNormalize_nf and doOutliers2D at the same time, turning off useNormalize_nf")
    if doOutliers2D==True and doOutliers1D==True:
        doOutliers1D = False
        print("we cannot currently do doOutliers1D and doOutliers2D at the same time, turning off doOutliers1D")
    
    #figure out how user can pass to do central x number of slices, or set of slices dispersed throughout (without knowing a priori the value of numslices)
    if sinoused is None:
        sinoused = (0,numslices,1)
    elif sinoused[0]<0:
        sinoused=(int(np.floor(numslices/2.0)-np.ceil(sinoused[1]/2.0)),int(np.floor(numslices/2.0)+np.floor(sinoused[1]/2.0)),1)
    
    num_proj_per_chunk = np.minimum(chunk_proj,projused[1]-projused[0])
    numprojchunks = (projused[1]-projused[0]-1)//num_proj_per_chunk+1
    num_sino_per_chunk = np.minimum(chunk_sino,sinoused[1]-sinoused[0])
    numsinochunks = (sinoused[1]-sinoused[0]-1)//num_sino_per_chunk+1
    numprojused = (projused[1]-projused[0])//projused[2]
    numsinoused = (sinoused[1]-sinoused[0])//sinoused[2]
    
    BeamHardeningCoefficients = (0, 1, 0, 0, 0, .1) if BeamHardeningCoefficients is None else BeamHardeningCoefficients

    if cor is None:
        print("Detecting center of rotation", end="") 
        if angularrange>300:
            lastcor = int(np.floor(numangles/2)-1)
        else:
            lastcor = numangles-1
        #I don't want to see the warnings about the reader using a deprecated variable in dxchange
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tomo, flat, dark, floc = dxchange.read_als_832h5(inputPath+filename,ind_tomo=(0,lastcor))
        tomo = tomo.astype(np.float32)
        if useNormalize_nf:
            tomopy.normalize_nf(tomo, flat, dark, floc, out=tomo)
        else:
            tomopy.normalize(tomo, flat, dark, out=tomo)

        if corFunction == 'vo':
            # same reason for catching warnings as above
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cor = tomopy.find_center_vo(tomo, ind=voInd, smin=voSMin, smax=voSMax, srad=voSRad, step=voStep,
                                        ratio=voRatio, drop=voDrop)
        elif corFunction == 'nm':
            cor = tomopy.find_center(tomo, tomopy.angles(numangles, angle_offset, angle_offset-angularrange),
                                     ind=nmInd, init=nmInit, tol=nmTol, mask=nmMask, ratio=nmRatio,
                                     sinogram_order=nmSinoOrder)
        elif corFunction == 'pc':
            cor = tomopy.find_center_pc(tomo[0], tomo[1], tol=0.25)
        else:
            raise ValueError("\'corFunction\' must be one of: [ pc, vo, nm ].")
        print(", {}".format(cor))
    else:
        print("using user input center of {}".format(cor))
        
    
    function_list = []

    if doOutliers1D:
        function_list.append('remove_outlier1d')
    if doOutliers2D:
        function_list.append('remove_outlier2d')
    if useNormalize_nf:
        function_list.append('normalize_nf')
    else:
        function_list.append('normalize')
    function_list.append('minus_log')
    if doBeamHardening:
        function_list.append('beam_hardening')
    if doFWringremoval:
        function_list.append('remove_stripe_fw')
    if doTIringremoval:
        function_list.append('remove_stripe_ti')
    if doSFringremoval:
        function_list.append('remove_stripe_sf')
    if correcttilt:
        function_list.append('correcttilt')
    if use360to180:
        function_list.append('do_360_to_180')
    if doPhaseRetrieval:
        function_list.append('phase_retrieval')
    function_list.append('recon_mask')
    if doPolarRing:
        function_list.append('polar_ring')
    if castTo8bit:
        function_list.append('castTo8bit')
    if doBilateralFilter:
        function_list.append('bilateral_filter')
    function_list.append('write_output')
        
    
    # Figure out first direction to slice
    for func in function_list:
        if slice_dir[func] != 'both':
            axis = slice_dir[func]
            break
    
    done = False
    curfunc = 0
    curtemp = 0
    while True: # Loop over reading data in certain chunking direction
        if axis=='proj':
            niter = numprojchunks
        else:
            niter = numsinochunks
        for y in range(niter): # Loop over chunks
            print("{} chunk {} of {}".format(axis, y+1, niter))
            if curfunc==0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if axis=='proj':
                        tomo, flat, dark, floc = dxchange.read_als_832h5(inputPath+filename,ind_tomo=range(y*num_proj_per_chunk+projused[0],np.minimum((y + 1)*num_proj_per_chunk+projused[0],numangles)),sino=(sinoused[0],sinoused[1], sinoused[2]) )
                    else:
                        tomo, flat, dark, floc = dxchange.read_als_832h5(inputPath+filename,ind_tomo=range(projused[0],projused[1],projused[2]),sino=(y*num_sino_per_chunk+sinoused[0],np.minimum((y + 1)*num_sino_per_chunk+sinoused[0],numslices),1) )
            else:
                if axis=='proj':
                    start, end = y * num_proj_per_chunk, np.minimum((y + 1) * num_proj_per_chunk,numprojused)
                    tomo = dxchange.reader.read_hdf5(tempfilenames[curtemp],'/tmp/tmp',slc=((start,end,1),(0,numslices,1),(0,numrays,1))) #read in intermediate file
                else:
                    start, end = y * num_sino_per_chunk, np.minimum((y + 1) * num_sino_per_chunk,numsinoused)
                    tomo = dxchange.reader.read_hdf5(tempfilenames[curtemp],'/tmp/tmp',slc=((0,numangles,1),(start,end,1),(0,numrays,1)))
            dofunc = curfunc
            keepvalues = None
            while True: # Loop over operations to do in current chunking direction
                func_name = function_list[dofunc]
                newaxis = slice_dir[func_name]
                if newaxis != 'both' and newaxis != axis:
                    # We have to switch axis, so flush to disk
                    if y==0:
                        try:
                            os.remove(tempfilenames[1-curtemp])
                        except OSError:
                            pass
                    appendaxis = 1 if axis=='sino' else 0
                    dxchange.writer.write_hdf5(tomo,fname=tempfilenames[1-curtemp],gname='tmp',dname='tmp',overwrite=False,appendaxis=appendaxis) #writing intermediate file...
                    break
                print(func_name, end=" ")
                curtime = time.time()
                if func_name == 'remove_outlier1d':
                    tomo = tomo.astype(np.float32,copy=False)
                    remove_outlier1d(tomo, outlier_diff1D, size=outlier_size1D, out=tomo)
                if func_name == 'remove_outlier2d':
                    tomo = tomo.astype(np.float32,copy=False)
                    tomopy.remove_outlier(tomo, outlier_diff2D, size=outlier_size2D, axis=0, out=tomo)
                elif func_name == 'normalize_nf':
                    tomo = tomo.astype(np.float32,copy=False)
                    tomopy.normalize_nf(tomo, flat, dark, floc_independent, out=tomo) #use floc_independent b/c when you read file in proj chunks, you don't get the correct floc returned right now to use here.
                elif func_name == 'normalize':
                    tomo = tomo.astype(np.float32,copy=False)
                    tomopy.normalize(tomo, flat, dark, out=tomo)
                elif func_name == 'minus_log':
                    mx = np.float32(0.00000000000000000001)
                    ne.evaluate('where(tomo>mx, tomo, mx)', out=tomo)
                    tomopy.minus_log(tomo, out=tomo)
                elif func_name == 'beam_hardening':
                    loc_dict = {'a{}'.format(i):np.float32(val) for i,val in enumerate(BeamHardeningCoefficients)}
                    tomo = ne.evaluate('a0 + a1*tomo + a2*tomo**2 + a3*tomo**3 + a4*tomo**4 + a5*tomo**5', local_dict=loc_dict, out=tomo)
                elif func_name == 'remove_stripe_fw':
                    tomo = tomopy.remove_stripe_fw(tomo, sigma=ringSigma, level=ringLevel, pad=True, wname=ringWavelet)
                elif func_name == 'remove_stripe_ti':
                    tomo = tomopy.remove_stripe_ti(tomo, nblock=ringNBlock, alpha=ringAlpha)
                elif func_name == 'remove_stripe_sf':
                    tomo = tomopy.remove_stripe_sf(tomo, size=ringSize)
                elif func_name == 'correcttilt':
                    if tiltcenter_slice is None:
                        tiltcenter_slice = numslices/2.
                    if tiltcenter_det is None:
                        tiltcenter_det = tomo.shape[2]/2
                    new_center = tiltcenter_slice - 0.5 - sinoused[0]
                    center_det = tiltcenter_det - 0.5
                    
                    #add padding of 10 pixels, to be unpadded right after tilt correction. This makes the tilted image not have zeros at certain edges, which matters in cases where sample is bigger than the field of view. For the small amounts we are generally tilting the images, 10 pixels is sufficient.
#                    tomo = tomopy.pad(tomo, 2, npad=10, mode='edge')
#                    center_det = center_det + 10
                    
                    cntr = (center_det, new_center)
                    for b in range(tomo.shape[0]):
                        tomo[b] = st.rotate(tomo[b], correcttilt, center=cntr, preserve_range=True, order=1, mode='edge', clip=True) #center=None means image is rotated around its center; order=1 is default, order of spline interpolation
#                    tomo = tomo[:, :, 10:-10]    
                        
                elif func_name == 'do_360_to_180':
                    
                    # Keep values around for processing the next chunk in the list
                    keepvalues = [angularrange, numangles, projused, num_proj_per_chunk, numprojchunks, numprojused, numrays, anglelist]
                    
                    #why -.5 on one and not on the other?
                    if tomo.shape[0]%2>0:
                        tomo = sino_360_to_180(tomo[0:-1,:,:], overlap=int(np.round((tomo.shape[2]-cor-.5))*2), rotation='right')
                        angularrange = angularrange/2 - angularrange/(tomo.shape[0]-1)
                    else:
                        tomo = sino_360_to_180(tomo[:,:,:], overlap=int(np.round((tomo.shape[2]-cor))*2), rotation='right')
                        angularrange = angularrange/2
                    numangles = int(numangles/2)
                    projused = (0,numangles-1,1)
                    num_proj_per_chunk = np.minimum(chunk_proj,projused[1]-projused[0])
                    numprojchunks = (projused[1]-projused[0]-1)//num_proj_per_chunk+1
                    numprojused = (projused[1]-projused[0])//projused[2]
                    numrays = tomo.shape[2]
                    
                    anglelist = anglelist[:numangles]
                
                elif func_name == 'phase_retrieval':
                    tomo = tomopy.retrieve_phase(tomo, pixel_size=pxsize, dist=propagation_dist, energy=kev, alpha=alphaReg, pad=True)
                
                elif func_name == 'translation_correction':
                    tomo = linear_translation_correction(tomo,dx=xshift,dy=yshift,interpolation=False):
                    
                elif func_name == 'recon_mask':
                    tomo = tomopy.pad(tomo, 2, npad=npad, mode='edge')

                    if projIgnoreList is not None:
                        for badproj in projIgnoreList:
                            tomo[badproj] = 0

                    rec = tomopy.recon(tomo, anglelist, center=cor+npad, algorithm='gridrec', filter_name='butterworth', filter_par=[butterworth_cutoff, butterworth_order])
                    rec = rec[:, npad:-npad, npad:-npad]
                    rec /= pxsize  # convert reconstructed voxel values from 1/pixel to 1/cm
                    rec = tomopy.circ_mask(rec, 0)
                elif func_name == 'polar_ring':
                    rec = np.ascontiguousarray(rec, dtype=np.float32)
                    rec = tomopy.remove_ring(rec, theta_min=Rarc, rwidth=Rmaxwidth, thresh_max=Rtmax, thresh=Rthr, thresh_min=Rtmin,out=rec)
                elif func_name == 'castTo8bit':
                    rec = convert8bit(rec, cast8bit_min, cast8bit_max)
                elif func_name == 'bilateral_filter':
                    rec = pyF3D.run_BilateralFilter(rec, spatialRadius=bilateral_srad, rangeRadius=bilateral_rrad)
                elif func_name == 'write_output':
                    dxchange.write_tiff_stack(rec, fname=filenametowrite, start=y*num_sino_per_chunk + sinoused[0])
                print('(took {:.2f} seconds)'.format(time.time()-curtime))
                dofunc+=1
                if dofunc==len(function_list):
                    break
            if y<niter-1 and keepvalues: # Reset original values for next chunk
                angularrange, numangles, projused, num_proj_per_chunk, numprojchunks, numprojused, numrays, anglelist = keepvalues
                
        curtemp = 1 - curtemp
        curfunc = dofunc
        if curfunc==len(function_list):
            break
        axis = slice_dir[function_list[curfunc]]
    print("cleaning up temp files")
    for tmpfile in tempfilenames:
        try:
            os.remove(tmpfile)
        except OSError:
            pass
    print("End Time: "+time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime()))
    print('It took {:.3f} s to process {}'.format(time.time()-start_time,inputPath+filename))

def recon_from_spreadsheet(filePath):
    """
    Runs recon() function using data read from spreadsheet
    """
    parameterList = spreadsheet(filePath)

    for i in range(len(parameterlist)):
        recon(parameterList[i])
    


def convert8bit(rec,data_min,data_max):
    rec = rec.astype(np.float32,copy=False)
    df = np.float32(data_max-data_min)
    mn = np.float32(data_min)
    scl = ne.evaluate('0.5+255*(rec-mn)/df',truediv=True)
    ne.evaluate('where(scl<0,0,scl)',out=scl)
    ne.evaluate('where(scl>255,255,scl)',out=scl)
    return scl.astype(np.uint8)
    

def sino_360_to_180(data, overlap=0, rotation='left'):
    """
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




def remove_outlier1d(arr, dif, size=3, axis=0, ncore=None, out=None):
    """
    Remove high intensity bright spots from an array, using a one-dimensional
    median filter along the specified axis.
    
    Dula: also removes dark spots
    
    Parameters
    ----------
    arr : ndarray
        Input array.
    dif : float
        Expected difference value between outlier value and
        the median value of the array.
    size : int
        Size of the median filter.
    axis : int, optional
        Axis along which median filtering is performed.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    out : ndarray, optional
        Output array for result.  If same as arr, process will be done in-place.
    Returns
    -------
    ndarray
       Corrected array.
    """
    arr = arr.astype(np.float32,copy=False)
    dif = np.float32(dif)

    tmp = np.empty_like(arr)

    other_axes = [i for i in range(arr.ndim) if i != axis]
    largest = np.argmax([arr.shape[i] for i in other_axes])
    lar_axis = other_axes[largest]
    ncore, chnk_slices = mproc.get_ncore_slices(arr.shape[lar_axis],ncore=ncore)
    filt_size = [1]*arr.ndim
    filt_size[axis] = size

    with cf.ThreadPoolExecutor(ncore) as e:
        slc = [slice(None)]*arr.ndim
        for i in range(ncore):
            slc[lar_axis] = chnk_slices[i]
            e.submit(snf.median_filter, arr[slc], size=filt_size,output=tmp[slc], mode='mirror')

    with mproc.set_numexpr_threads(ncore):
        out = ne.evaluate('where(abs(arr-tmp)>=dif,tmp,arr)', out=out)

    return out

    
def translate(data,dx=0,dy=0,interpolation=True):
    """
    Shifts all projections in an image stack by dx (horizontal) and dy (vertical) pixels. Translation with subpixel resolution is possible with interpolation==True
    
    Parameters
    ----------
    data: ndarray 
        Input array, stack of 2D (x,y) images, angle in z
        
    dx: int or float
        desored horizontal pixel shift
        
    dy: int or float
        desired vertical pixel shift
    
    interpolation: boolean
        True calls funtion from sckimage to interpolate image when subpixel shifts are applied
    
    Returns
    -------
    ndarray
       Corrected array.
    """

    Nproj, Nrow, Ncol = data.shape
    dataOut = np.zeros(data.shape)
    
    if interpolation == True:
        #translateFunction = st.SimilarityTransform(translation=(-dx,dy))
        M=np.matrix([[1,0,-dx],[0,1,dy],[0,0,1]])
        translateFunction = st.SimilarityTransform(matrix=M)
        for n in range(Nproj):
            dataOut[n,:,:] = st.warp(data[n,:,:], translateFunction)
            
    if interpolation == False:
        Npad = max(dx,dy)        
        drow = int(-dy) # negative matrix row increments = dy
        dcol = int(dx)  # matrix column increments = dx
        for n in range(Nproj):
            PaddedImage = np.pad(data[n,:,:],Npad,'constant')
            dataOut[n,:,:] = PaddedImage[Npad-drow:Npad+Nrow-drow,Npad-dcol:Npad+Ncol-dcol]  # shift image by dx and dy, replace original without padding
            
 return dataOut

    
def linear_translation_correction(data,dx=100.5,dy=700.1,interpolation=True):

    """
    Corrects for a linear drift in field of view (horizontal dx, vertical dy) over time. The first index indicaties time data[time,:,:] in the time series of projections. dx and dy are the final shifts in FOV position.
    
    Parameters
    ----------
    data: ndarray 
        Input array, stack of 2D (x,y) images, angle in z
        
    dx: int or float
        total horizontal pixel offset from first (0 deg) to last (180 deg) projection 

    dy: int or float
        total horizontal pixel offset from first (0 deg) to last (180 deg) projection 
    
    interpolation: boolean
        True calls funtion from sckimage to interpolate image when subpixel shifts are applied
    
    Returns
    -------
    ndarray
       Corrected array.
    """

    Nproj, Nrow, Ncol = data.shape
    Nproj=10
    
    dataOut = np.zeros(data0.shape)
        
    dx_n = np.linspace(0,dx,Nproj) # generate array dx[n] of pixel shift for projection n = 0, 1, ... Nproj

    dy_n = np.linspace(0,dy,Nproj) # generate array dy[n] of pixel shift for projection n = 0, 1, ... Nproj
    
    if interpolation==True:
        for n in range(Nproj):
            #translateFunction = st.SimilarityTransform(translation=(-dx_n[n],dy_n[n])) # Generate Translation Function based on dy[n] and dx[n]
            M=np.matrix([[1,0,-dx_n[n]],[0,1,dy_n[n]],[0,0,1]])
            translateFunction = st.SimilarityTransform(matrix=M)
            image_n = data[n,:,:]
            dataOut[n,:,:] = st.warp(image_n, translateFunction) # Apply translation with interpolation to projection[n]
            #print(n)

    if interpolation==False:
        Npad = max(dx,dy)
        for n in range(Nproj):
            PaddedImage = np.pad(data[n,:,:],Npad,'constant') # copy single projection and pad with maximum of dx,dy
            drow = int(round(-dy_n[n])) # round shift to nearest pixel step, negative matrix row increments = dy
            dcol = int(round(dx_n[n]))  # round shift to nearest pixel step, matrix column increments = dx
            dataOut[n,:,:] = PaddedImage[Npad-drow:Npad+Nrow-drow,Npad-dcol:Npad+Ncol-dcol] # shift image by dx and dy, replace original without padding
            #print(n)

    return dataOut

    
    """
    Parameters
    ----------
    data: ndarray 
        Input array, stack of 2D (x,y) images, angle in z 
    pixelshift: float
        total pixel offset from first (0 deg) to last (180 deg) projection 
    
    Returns
    -------
    ndarray
       Corrected array.
    """
    
    
"""Hi Dula,
This is roughly what I am doing in the script to 'unspiral' the superweave data:
spd = float(int(sys.argv[2])/2048)
x = np.zeros((2049,200,2560), dtype=np.float32)
blks = np.round(np.linspace(0,2049,21)).astype(np.int)
for i in range(0,20):
    dat = dxchange.read_als_832h5(fn, ind_tomo=range(blks[i],blks[i+1]))
    prj = tomopy.normalize_nf(dat[0],dat[1],dat[2],dat[3])
    for ik,j in enumerate(range(blks[i],blks[i+1])):
        l = prj.shape[1]//2-j*spd
        li = int(l)
        ri = li+200
        fc = l-li
        x[j] = (1-fc)*prj[ik,li:ri]
        x[j] += fc*prj[ik,li+1:ri+1]
dxchange.writer.write_hdf5(x, fname=fn[:-3]+'_unspiral.h5', overwrite=True, gname='tmp', dname='tmp', appendaxis=0)

This processes the (roughly) central 200 slices, and saves it to a new file. The vertical speed is one of the input arguments, and I simply estimate it manually by looking at the first and last projection, shifting them by 'np.roll'. The input argument is the total number of pixels that are shifted over the whole scan (which is then converted to pixels per projection by dividing by the number of projections-1).
I don't really remember why I wrote my own code, but maybe I was running into problems using scikit-image as well. The current code uses linear interpolation, and gives pretty good results for the data I tested.

Best,

Daniel"""
    
    
def convertthetype(val):
    constructors = [int, float, str]
    for c in constructors:
        try:
            return c(val)
        except ValueError:
            pass

#Converts spreadsheet.xlsx file with headers into dictionaries
def read_spreadsheet(filepath):
    workbook=xlrd.open_workbook(filepath)
    worksheet = workbook.sheet_by_index(0)

    # imports first row and converts to a list of header strings
    headerList = []
    for col_index in range(worksheet.ncols):
        headerList.append(str(worksheet.cell_value(0,col_index)))

    dataList = []
    # For each row, create a dictionary and like header name to data 
    # converts each row to following format rowDictionary1 ={'header1':colvalue1,'header2':colvalue2,... }
    # compiles rowDictinaries into a list: dataList = [rowDictionary1, rowDictionary2,...]
    for row_index in range(1,worksheet.nrows):
        rowDictionary = {}
        for col_index in range(worksheet.ncols):
            cellValue = worksheet.cell_value(row_index,col_index)

            if type(cellValue)==unicode:
                cellValue = str(cellValue)
            
            # if cell contains string that looks like a tuple, convert to tuple
            if '(' in str(cellValue):
                cellValue = literal_eval(cellValue)

            # if cell contains string or int that looks like 'True', convert to boolean True
            if str(cellValue).lower() =='true' or (type(cellValue)==int and cellValue==1):
                cellValue = True

            # if cell contains string or int that looks like 'False', convert to boolean False
            if str(cellValue).lower() =='false' or (type(cellValue)==int and cellValue==0):
                cellValue = False

            if cellValue != '': # create dictionary element if cell value is not empty
                rowDictionary[headerList[col_index]] = cellValue
        dataList.append(rowDictionary)

    return(dataList)


# D.Y.Parkinson's interpreter for text input files
def main():
    parametersfile = 'input832.txt' if (len(sys.argv)<2) else sys.argv[1]

    if parametersfile.split('.')[-1] == 'txt':
        with open(parametersfile,'r') as theinputfile:
            theinput = theinputfile.read()
            inputlist = theinput.splitlines()
            for reconcounter in range(0,len(inputlist)):
                inputlisttabsplit = inputlist[reconcounter].split()
                functioninput = {'filename': inputlisttabsplit[0]}
                for inputcounter in range(0,(len(inputlisttabsplit)-1)//2):
                    inputlisttabsplit[inputcounter*2+2] = inputlisttabsplit[inputcounter*2+2].replace('\"','')
                    inputcommasplit = inputlisttabsplit[inputcounter*2+2].split(',')
                    if len(inputcommasplit)>1:
                        inputcommasplitconverted = []
                        for jk in range(0,len(inputcommasplit)):
                            inputcommasplitconverted.append(convertthetype(inputcommasplit[jk]))
                    else:
                        inputcommasplitconverted = convertthetype(inputlisttabsplit[inputcounter*2+2])
                    functioninput[inputlisttabsplit[inputcounter*2+1]] = inputcommasplitconverted
                print("Read user input:")
                print(functioninput)
                recon(**functioninput)

# H.S.Barnard Spreadsheet interpreter
    if parametersfile.split('.')[-1]=='xlsx':
        functioninput = read_spreadsheet(parametersfile)
        for i in range(len(functioninput)):
            recon(**functioninput[i])

if __name__ == '__main__':
    main()
        
