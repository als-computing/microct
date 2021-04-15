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
#import xlrd # for importing excel spreadsheets
#from ast import literal_eval # For converting string to tuple

try:
    import tomopy
    from tomopy.util import mproc
except:
    print("warning: tomopy is not available")

try:
    import dxchange
except:
    print("warning: dxchange is not available")

# run this from the command line:
# python tomopy832.py
# it requires a separate file, which contains at minimum a list of filenames
# on separate lines. Default name of this file is input832.txt, but you can use any
# filename and run from the commandline as
# python tomopy832.py yourinputfile.txt
# If desired, on each line (separated by spaces) you can
# include parameters to override the defaults.
# to do this you need pairs, first the name of the variable, then the desired value
# For True/False, use 1/0.
# You can generate these input files in excel, in which case use tab-separated
# (or space separated). Some input overrides require multiple values,
# these should be comma-separated (with no spaces). Example is sinoused
# which would be e.g. 500,510,1 to get slices 500 through 509. For sinoused,
# you can use first value -1 and second value number of slices to get that number
# of slices from the middle of the stack.
# an example of the contents of the input file look like this:

# filename.h5 cor 1196 sinoused "-1,10,1" doPhaseRetrieval 0 outputFilename c1196.0
# filename.h5 cor 1196.5 sinoused "-1,10,1" doPhaseRetrieval 0 outputFilename c1196.5

# this was generated in excel and saved as txt tab separated, so the quotes were
# added automatically by excel. Note also that for parameters expecting strings as
# input (outputFilename for example), the program will choke if you put in a number.

# if cor is not defined in the parameters file, automated cor detection will happen

# chunk_proj and chunk_sino handle memory management.
# If you are running out of memory, make one or both of those smaller.

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
    'lensdistortion': 'proj',
    'phase_retrieval': 'proj',
    'recon_mask': 'sino',
    'polar_ring': 'sino',
    'polar_ring2': 'sino',
    'castTo8bit': 'both',
    'write_reconstruction': 'both',
    'write_normalized': 'proj',
}


def recon_setup(
    filename,
    filetype = 'dxfile',
    timepoint = 0,
    bffilename = None,
    inputPath = './',  # input path, location of the data set to reconstruct
    outputPath=None,
    # define an output path (default is inputPath), a sub-folder will be created based on file name
    outputFilename=None,
    # file name for output tif files (a number and .tiff will be added). default is based on input filename
    fulloutputPath=None,  # definte the full output path, no automatic sub-folder will be created
    doOutliers1D=False,  # outlier removal in 1d (along sinogram columns)
    outlier_diff1D=750,  # difference between good data and outlier data (outlier removal)
    outlier_size1D=3,  # radius around each pixel to look for outliers (outlier removal)
    doOutliers2D=False,  # outlier removal, standard 2d on each projection
    outlier_diff2D=750,  # difference between good data and outlier data (outlier removal)
    outlier_size2D=3,  # radius around each pixel to look for outliers (outlier removal)
    doFWringremoval=True,  # Fourier-wavelet ring removal
    doTIringremoval=False,  # Titarenko ring removal
    doSFringremoval=False,  # Smoothing filter ring removal
    ringSigma=3,  # damping parameter in Fourier space (Fourier-wavelet ring removal)
    ringLevel=8,  # number of wavelet transform levels (Fourier-wavelet ring removal)
    ringWavelet='db5',  # type of wavelet filter (Fourier-wavelet ring removal)
    ringNBlock=0,  # used in Titarenko ring removal (doTIringremoval)
    ringAlpha=1.5,  # used in Titarenko ring removal (doTIringremoval)
    ringSize=5,  # used in smoothing filter ring removal (doSFringremoval)
    doPhaseRetrieval=False,  # phase retrieval
    alphaReg=0.0002,  # smaller = smoother (used for phase retrieval)
    propagation_dist=75,  # sample-to-scintillator distance (phase retrieval)
    kev=24,  # energy level (phase retrieval)
    butterworth_cutoff=0.25,  # 0.1 would be very smooth, 0.4 would be very grainy (reconstruction)
    butterworth_order=2,  # for reconstruction
    doTranslationCorrection=False,  # correct for linear drift during scan
    xshift=0,  # undesired dx transation correction (from 0 degree to 180 degree proj)
    yshift=0,  # undesired dy transation correction (from 0 degree to 180 degree proj)
    doPolarRing=False,  # ring removal
    Rarc=30,  # min angle needed to be considered ring artifact (ring removal)
    Rmaxwidth=100,  # max width of rings to be filtered (ring removal)
    Rtmax=3000.0,  # max portion of image to filter (ring removal)
    Rthr=3000.0,  # max value of offset due to ring artifact (ring removal)
    Rtmin=-3000.0,  # min value of image to filter (ring removal)
    doPolarRing2=False,  # ring removal
    Rarc2=30,  # min angle needed to be considered ring artifact (ring removal)
    Rmaxwidth2=100,  # max width of rings to be filtered (ring removal)
    Rtmax2=3000.0,  # max portion of image to filter (ring removal)
    Rthr2=3000.0,  # max value of offset due to ring artifact (ring removal)
    Rtmin2=-3000.0,  # min value of image to filter (ring removal)
    cor=None,  # center of rotation (float). If not used then cor will be detected automatically
    corFunction='pc',  # center of rotation function to use - can be 'pc', 'vo', or 'nm', or use 'skip' to return tomo variable without having to do a calc.
    corLoadMinimalBakDrk=True, #during cor detection, only load the first dark field and first flat field rather than all of them, to minimize file loading time for cor detection.
    voInd=None,  # index of slice to use for cor search (vo)
    voSMin=-150,  # min radius for searching in sinogram (vo)
    voSMax=150,  # max radius for searching in sinogram (vo)
    voSRad=6,  # search radius (vo)
    voStep=0.25,  # search step (vo)
    voRatio=0.5,  # ratio of field-of-view and object size (vo)
    voDrop=20,  # drop lines around vertical center of mask (vo)
    nmInd=None,  # index of slice to use for cor search (nm)
    nmInit=None,  # initial guess for center (nm)
    nmTol=0.5,  # desired sub-pixel accuracy (nm)
    nmMask=True,  # if True, limits analysis to circular region (nm)
    nmRatio=1.0,  # ratio of radius of circular mask to edge of reconstructed image (nm)
    nmSinoOrder=False,  # if True, analyzes in sinogram space. If False, analyzes in radiograph space
    use360to180=False,  # use 360 to 180 conversion
    castTo8bit=False,  # convert data to 8bit before writing
    cast8bit_min=-10,  # min value if converting to 8bit
    cast8bit_max=30,  # max value if converting to 8bit
    useNormalize_nf=False,  # normalize based on background intensity (nf)
    chunk_proj=100,  # chunk size in projection direction
    chunk_sino=100,  # chunk size in sinogram direction
    npad=None,  # amount to pad data before reconstruction
    projused=None,
    # should be slicing in projection dimension (start,end,step) Be sure to add one to the end as stop in python means the last value is omitted
    sinoused=None,
    # should be sliceing in sinogram dimension (start,end,step). If first value is negative, it takes the number of slices from the second value in the middle of the stack.
    correcttilt=0,  # tilt dataset
    tiltcenter_slice=None,  # tilt center (x direction)
    tiltcenter_det=None,  # tilt center (y direction)
    angle_offset=0,
    # this is the angle offset from our default (270) so that tomopy yields output in the same orientation as previous software (Octopus)
    anglelist=None,
    # if not set, will assume evenly spaced angles which will be calculated by the angular range and number of angles found in the file. if set to -1, will read individual angles from each image. alternatively, a list of angles can be passed.
    doBeamHardening=False,
    # turn on beam hardening correction, based on "Correction for beam hardening in computed tomography", Gabor Herman, 1979 Phys. Med. Biol. 24 81
    BeamHardeningCoefficients=None,  # 6 values, tomo = a0 + a1*tomo + a2*tomo^2 + a3*tomo^3 + a4*tomo^4 + a5*tomo^5
    projIgnoreList=None,
    # projections to be ignored in the reconstruction (for simplicity in the code, they will not be removed and will be processed as all other projections but will be set to zero absorption right before reconstruction.
    bfexposureratio=1,  # ratio of exposure time of bf to exposure time of sample
    dorecon=True, #do the tomographic reconstruction
    writenormalized=False,
    writereconstruction=True,
    dominuslog=True,
    slsnumangles=1000,
    slspxsize=0.00081,
    verbose_printing=False,
    recon_algorithm='gridrec',  # choose from gridrec, fbp, and others in tomopy
    dolensdistortion=False,
    lensdistortioncenter=(1280,1080),
    lensdistortionfactors = (1.00015076, 1.9289e-06, -2.4325e-08, 1.00439e-11, -3.99352e-15),
    minimum_transmission = 0.01,
    *args, **kwargs
    ):


    outputFilename = os.path.splitext(filename)[0] if outputFilename is None else outputFilename
    # outputPath = inputPath + 'rec' + os.path.splitext(filename)[0] + '/' if outputPath is None else outputPath + 'rec' + os.path.splitext(filename)[0] + '/'
    outputPath = os.path.join(inputPath, 'rec' + outputFilename) if outputPath is None else os.path.join(outputPath,'rec' + outputFilename)
    fulloutputPath = outputPath if fulloutputPath is None else fulloutputPath
    tempfilenames = [os.path.join(fulloutputPath,'tmp0.h5'), os.path.join(fulloutputPath, 'tmp1.h5')]

    if verbose_printing:
        print("cleaning up previous temp files", end="")
    for tmpfile in tempfilenames:
        try:
            os.remove(tmpfile)
        except OSError:
            pass
    if verbose_printing:
        print(", reading metadata")

    if filetype == 'als':
        datafile = h5py.File(os.path.join(inputPath,filename), 'r')
        gdata = dict(dxchange.reader._find_dataset_group(datafile).attrs)
        pxsize = float(gdata['pxsize']) / 10  # /10 to convert units from mm to cm
        numslices = int(gdata['nslices'])
        numangles = int(gdata['nangles'])
        angularrange = float(gdata['arange'])
        numrays = int(gdata['nrays'])
        inter_bright = int(gdata['i0cycle'])



        dgroup = dxchange.reader._find_dataset_group(datafile)
        keys = list(gdata.keys())
        if 'num_dark_fields' in keys:
            ndark = int(gdata['num_dark_fields'])
        else:
            ndark = dxchange.reader._count_proj(dgroup, dgroup.name.split('/')[-1] + 'drk_0000.tif', numangles, inter_bright=-1) #for darks, don't want to divide out inter_bright for counting projections
        ind_dark = list(range(0, ndark))
        group_dark = [numangles - 1]

        if 'num_bright_field' in keys:
            nflat = int(gdata['num_bright_field'])
        else:
            nflat = dxchange.reader._count_proj(dgroup, dgroup.name.split('/')[-1] + 'bak_0000.tif', numangles, inter_bright=inter_bright)
        ind_flat = list(range(0, nflat))

        # figure out the angle list (a list of angles, one per projection image)
        dtemp = datafile[list(datafile.keys())[0]]
        fltemp = list(dtemp.keys())
        firstangle = float(dtemp[fltemp[0]].attrs.get('rot_angle', 0))
        if anglelist is None:
            # the offset angle should offset from the angle of the first image, which is usually 0, but in the case of timbir data may not be.
            # we add the 270 to be inte same orientation as previous software used at bl832
            angle_offset = 270 + angle_offset - firstangle
            anglelist = tomopy.angles(numangles, angle_offset, angle_offset - angularrange)
        elif anglelist == -1:
            anglelist = np.zeros(shape=numangles)
            for icount in range(0, numangles):
                anglelist[icount] = np.pi / 180 * (270 + angle_offset - float(dtemp[fltemp[icount]].attrs['rot_angle']))
        if inter_bright > 0:
            group_flat = list(range(0, numangles, inter_bright))
            if group_flat[-1] != numangles - 1:
                group_flat.append(numangles - 1)
        elif inter_bright == 0:
            group_flat = [0, numangles - 1]
        else:
            group_flat = None
    elif filetype == 'dxfile':
        _, _, _, anglelist, meta = dxchange.exchange.read_dx(os.path.join(inputPath, filename))
        anglelist = -anglelist
        numslices = int(meta['dimension_y'][0])
        numrays = int(meta['dimension_x'][0])
        pxsize = meta['pixel_size'][0]
        numangles = int(meta['num_angles'][0])
        angularrange = meta['range'][0]
        inter_bright = int(meta['i0cycle'][0])
        group_flat = [0, numangles - 1]
        nflat =  int(meta['num_flat_fields'][0])
        ind_flat = list(range(0, nflat))
        ndark = int(meta['num_dark_fields'][0])
        ind_dark = list(range(0, ndark))
    elif filetype == 'sls':
        datafile = h5py.File(os.path.join(inputPath, filename), 'r')
        slsdata = datafile["exchange/data"]
        numslices = slsdata.shape[1]
        numrays = slsdata.shape[2]
        pxsize = slspxsize
        numangles = slsnumangles
        _, _, _, anglelist = read_sls(os.path.join(inputPath,filename),  exchange_rank=0, proj=(timepoint*numangles,(timepoint+1)*numangles,1), sino=(0,1,1)) #dtype=None, , )
        angularrange = np.abs(anglelist[-1]-anglelist[0])
        inter_bright = 0
        group_flat = [0, numangles - 1]
        nflat = 1 #this variable is not used for sls data
        ind_flat = list(range(0, nflat))
    else:
        print("Not sure what file type, gotta break.")
        return

    npad = int(np.ceil(numrays * np.sqrt(2)) - numrays) // 2 if npad is None else npad
    if projused is not None and (projused[1] > numangles - 1 or projused[0] < 0):  # allows program to deal with out of range projection values
        if projused[1] > numangles:
            print("End Projection value greater than number of angles. Value has been lowered to the number of angles " + str(numangles))
            projused = (projused[0], numangles, projused[2])
        if projused[0] < 0:
            print("Start Projection value less than zero. Value raised to 0")
            projused = (0, projused[1], projused[2])
    if projused is None:
        projused = (0, numangles, 1)
    else:
        # if projused is different than default, need to change numangles and angularrange; dula attempting to do this with these two lines, we'll see if it works! 11/16/17
        angularrange = (angularrange / (numangles - 1)) * (projused[1] - projused[0])
        #dula updated to use anglelist to find angular rage, 11 june 2020, not sure if it will work??
        angularrange = np.abs(anglelist[projused[1]] - anglelist[projused[0]])
        # want angular range to stay constant if we keep the end values consistent
        numangles = len(range(projused[0], projused[1], projused[2]))

    ind_tomo = list(range(0, numangles))
    floc_independent = dxchange.reader._map_loc(ind_tomo, group_flat)

    # figure out how user can pass to do central x number of slices, or set of slices dispersed throughout (without knowing a priori the value of numslices)
    if sinoused is None:
        sinoused = (0, numslices, 1)
    elif sinoused[0] < 0:
        sinoused = (int(np.floor(numslices / 2.0) - np.ceil(sinoused[1] / 2.0)), int(np.floor(numslices / 2.0) + np.floor(sinoused[1] / 2.0)), 1)

    if verbose_printing:
        print('There are ' + str(numslices) + ' sinograms, ' + str(numrays) + ' rays, and ' + str(numangles) + ' projections, with an angular range of ' +str(angularrange) + '.')
        print('Looking at sinograms ' + str(sinoused[0]) + ' through ' + str(sinoused[1]-1) + ' (inclusive) in steps of ' + str(sinoused[2]))

    BeamHardeningCoefficients = (0, 1, 0, 0, 0, .1) if BeamHardeningCoefficients is None else BeamHardeningCoefficients

    if cor is None:
        if verbose_printing:
            print("Detecting center of rotation", end="")

        if angularrange > 300:
            lastcor = int(np.floor(numangles / 2) - 1)
        else:
            lastcor = numangles - 1
        # I don't want to see the warnings about the reader using a deprecated variable in dxchange
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if (filetype == 'als'):
                if corLoadMinimalBakDrk:
                    ind_dark = 0
                    ind_flat = 0
                tomo, flat, dark, floc = dxchange.read_als_832h5(os.path.join(inputPath, filename), ind_tomo=(0, lastcor),ind_dark=ind_dark,ind_flat=ind_flat)
            elif filetype == 'dxfile':
                # if corLoadMinimalBakDrk:
                #     ind_dark = 0
                #     ind_flat = 0
                # tomo, flat, dark, coranglelist, _ = dxchange.exchange.read_dx(os.path.join(inputPath, filename), proj=(0,numangles-1),ind_dark=ind_dark,ind_flat=ind_flat)
                tomo, flat, dark, coranglelist, _ = dxchange.exchange.read_dx(os.path.join(inputPath, filename), proj=(0,lastcor,lastcor-1))
            elif (filetype == 'sls'):
                tomo, flat, dark, coranglelist = read_sls(os.path.join(inputPath,filename), exchange_rank=0, proj=(
                    timepoint * numangles, (timepoint + 1) * numangles, numangles - 1))  # dtype=None, , )
            else:
                return
            if bffilename is not None and (filetype == 'als'):
                tomobf, flatbf, darkbf, flocbf = dxchange.read_als_832h5(os.path.join(inputPath, bffilename))
                flat = tomobf
        tomo = tomo.astype(np.float32)
        if useNormalize_nf and (filetype == 'als'):
            tomopy.normalize_nf(tomo, flat, dark, floc, out=tomo)
            if bfexposureratio != 1:
                tomo = tomo * bfexposureratio
        else:
            tomopy.normalize(tomo, flat, dark, out=tomo)
            if bfexposureratio != 1:
                tomo = tomo * bfexposureratio

        if corFunction == 'vo':

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if (filetype == 'als'):
                    tomovo, flat, dark, floc = dxchange.read_als_832h5(os.path.join(inputPath, filename), sino=(sinoused[0],sinoused[0]+1,1))
                elif (filetype == 'sls'):
                    tomovo, flat, dark, coranglelist = read_sls(os.path.join(inputPath, filename), exchange_rank=0, sino=(sinoused[0],sinoused[0]+1,1), proj=(timepoint*numangles+projused[0],timepoint*numangles+projused[1],projused[2]))  # dtype=None, , )
                else:
                    return
                if bffilename is not None and (filetype == 'als'):
                    tomobf, flatbf, darkbf, flocbf = dxchange.read_als_832h5(os.path.join(inputPath, bffilename), sino=(sinoused[0],sinoused[0]+1,1))
                    flat = tomobf
            tomovo = tomovo.astype(np.float32)

            if useNormalize_nf and (filetype == 'als'):
                tomopy.normalize_nf(tomovo, flat, dark, floc, out=tomovo)
                if bfexposureratio != 1:
                    tomovo = tomovo * bfexposureratio
            else:
                tomopy.normalize(tomovo, flat, dark, out=tomovo)
                if bfexposureratio != 1:
                    tomovo = tomovo * bfexposureratio

            cor = tomopy.find_center_vo(tomovo, ind=voInd, smin=voSMin, smax=voSMax, srad=voSRad, step=voStep,
                                            ratio=voRatio, drop=voDrop)


        elif corFunction == 'nm':
            cor = tomopy.find_center(tomo, tomopy.angles(numangles, angle_offset, angle_offset - angularrange),
                                     ind=nmInd, init=nmInit, tol=nmTol, mask=nmMask, ratio=nmRatio,
                                     sinogram_order=nmSinoOrder)
        elif corFunction == 'pc':
            if angularrange > 300:
                lastcor = int(np.floor(numangles / 2) - 1)
            else:
                lastcor = numangles - 1
            # I don't want to see the warnings about the reader using a deprecated variable in dxchange
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if (filetype == 'als'):
                    tomo, flat, dark, floc = dxchange.read_als_832h5(os.path.join(inputPath, filename), ind_tomo=(0, lastcor))
                elif (filetype == 'dxfile'):
                    tomo, flat, dark, coranglelist,_ = dxchange.read_dx(os.path.join(inputPath, filename), exchange_rank=0, proj=(
                        0, lastcor, lastcor-1))  # dtype=None, , )
                elif (filetype == 'sls'):
                    tomo, flat, dark, coranglelist = read_sls(os.path.join(inputPath, filename), exchange_rank=0, proj=(
                        timepoint * numangles, (timepoint + 1) * numangles, numangles - 1))  # dtype=None, , )
                else:
                    return
                if bffilename is not None and (filetype == 'als'):
                    tomobf, flatbf, darkbf, flocbf = dxchange.read_als_832h5(os.path.join(inputPath, bffilename))
                    flat = tomobf
            tomo = tomo.astype(np.float32)
            if useNormalize_nf and (filetype == 'als'):
                tomopy.normalize_nf(tomo, flat, dark, floc, out=tomo)
                if bfexposureratio != 1:
                    tomo = tomo * bfexposureratio
            else:
                tomopy.normalize(tomo, flat, dark, out=tomo)
                if bfexposureratio != 1:
                    tomo = tomo * bfexposureratio
            cor = tomopy.find_center_pc(tomo[0], tomo[-1], tol=0.25)
        elif corFunction == 'skip': #use this to get back the tomo variable without running processing
            cor = numrays/2
        else:
            raise ValueError("\'corFunction\' must be one of: [ pc, vo, nm ].")
        if verbose_printing:
            print(", {}".format(cor))
    else:
        tomo = 0
        if verbose_printing:
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
    if dominuslog:
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
    if dolensdistortion:
        function_list.append('lensdistortion')
    if use360to180:
        function_list.append('do_360_to_180')
    if doPhaseRetrieval:
        function_list.append('phase_retrieval')
    if dorecon:
        function_list.append('recon_mask')
    if doPolarRing:
        function_list.append('polar_ring')
    if doPolarRing2:
        function_list.append('polar_ring2')
    if castTo8bit:
        function_list.append('castTo8bit')
    if writereconstruction:
        function_list.append('write_reconstruction')
    if writenormalized:
        function_list.append('write_normalized')

    recon_dict = {
        "inputPath": inputPath, #input file path
        "filename": filename, #input file name
        "filetype": filetype,
        "timepoint": timepoint,
        "fulloutputPath": fulloutputPath,
        "outputFilename": outputFilename,
        "bffilename": bffilename, #if there is a separate file with the bright fields
        "doOutliers1D": doOutliers1D,  # outlier removal in 1d (along sinogram columns)
        "outlier_diff1D": outlier_diff1D,  # difference between good data and outlier data (outlier removal)
        "outlier_size1D": outlier_size1D,  # radius around each pixel to look for outliers (outlier removal)
        "doOutliers2D": doOutliers2D,  # outlier removal, standard 2d on each projection
        "outlier_diff2D": outlier_diff2D,  # difference between good data and outlier data (outlier removal)
        "outlier_size2D": outlier_size2D,  # radius around each pixel to look for outliers (outlier removal)
        "doFWringremoval": doFWringremoval,  # Fourier-wavelet ring removal
        "doTIringremoval": doTIringremoval,  # Titarenko ring removal
        "doSFringremoval": doSFringremoval,  # Smoothing filter ring removal
        "ringSigma": ringSigma,  # damping parameter in Fourier space (Fourier-wavelet ring removal)
        "ringLevel": ringLevel,  # number of wavelet transform levels (Fourier-wavelet ring removal)
        "ringWavelet": ringWavelet,  # type of wavelet filter (Fourier-wavelet ring removal)
        "ringNBlock": ringNBlock,  # used in Titarenko ring removal (doTIringremoval)
        "ringAlpha": ringAlpha,  # used in Titarenko ring removal (doTIringremoval)
        "ringSize": ringSize,  # used in smoothing filter ring removal (doSFringremoval)
        "doPhaseRetrieval": doPhaseRetrieval,  # phase retrieval
        "alphaReg": alphaReg,  # smaller = smoother (used for phase retrieval)
        "propagation_dist": propagation_dist,  # sample-to-scintillator distance (phase retrieval)
        "kev": kev,  # energy level (phase retrieval)
        "butterworth_cutoff": butterworth_cutoff,  # 0.1 would be very smooth, 0.4 would be very grainy (reconstruction)
        "butterworth_order": butterworth_order,  # for reconstruction
        "doTranslationCorrection": doTranslationCorrection,  # correct for linear drift during scan
        "xshift": xshift,  # undesired dx transation correction (from 0 degree to 180 degree proj)
        "yshift": yshift,  # undesired dy transation correction (from 0 degree to 180 degree proj)
        "doPolarRing": doPolarRing,  # ring removal
        "Rarc": Rarc,  # min angle needed to be considered ring artifact (ring removal)
        "Rmaxwidth": Rmaxwidth,  # max width of rings to be filtered (ring removal)
        "Rtmax": Rtmax,  # max portion of image to filter (ring removal)
        "Rthr": Rthr,  # max value of offset due to ring artifact (ring removal)
        "Rtmin": Rtmin,  # min value of image to filter (ring removal)
        "doPolarRing2": doPolarRing2,  # ring removal
        "Rarc2": Rarc2,  # min angle needed to be considered ring artifact (ring removal)
        "Rmaxwidth2": Rmaxwidth2,  # max width of rings to be filtered (ring removal)
        "Rtmax2": Rtmax2,  # max portion of image to filter (ring removal)
        "Rthr2": Rthr2,  # max value of offset due to ring artifact (ring removal)
        "Rtmin2": Rtmin2,  # min value of image to filter (ring removal)
        "cor": cor,  # center of rotation (float). If not used then cor will be detected automatically
        "corFunction": corFunction,  # center of rotation function to use - can be 'pc', 'vo', or 'nm'
        "voInd": voInd,  # index of slice to use for cor search (vo)
        "voSMin": voSMin,  # min radius for searching in sinogram (vo)
        "voSMax": voSMax,  # max radius for searching in sinogram (vo)
        "voSRad": voSRad,  # search radius (vo)
        "voStep": voStep,  # search step (vo)
        "voRatio": voRatio,  # ratio of field-of-view and object size (vo)
        "voDrop": voDrop,  # drop lines around vertical center of mask (vo)
        "nmInd": nmInd,  # index of slice to use for cor search (nm)
        "nmInit": nmInit,  # initial guess for center (nm)
        "nmTol": nmTol,  # desired sub-pixel accuracy (nm)
        "nmMask": nmMask,  # if True, limits analysis to circular region (nm)
        "nmRatio": nmRatio,  # ratio of radius of circular mask to edge of reconstructed image (nm)
        "nmSinoOrder": nmSinoOrder,  # if True, analyzes in sinogram space. If False, analyzes in radiograph space
        "use360to180": use360to180,  # use 360 to 180 conversion
        "castTo8bit": castTo8bit,  # convert data to 8bit before writing
        "cast8bit_min": cast8bit_min,  # min value if converting to 8bit
        "cast8bit_max": cast8bit_max,  # max value if converting to 8bit
        "useNormalize_nf": useNormalize_nf,  # normalize based on background intensity (nf)
        "chunk_proj": chunk_proj,  # chunk size in projection direction
        "chunk_sino": chunk_sino,  # chunk size in sinogram direction
        "npad": npad,  # amount to pad data before reconstruction
        "projused": projused, # should be slicing in projection dimension (start,end,step) Be sure to add one to the end as stop in python means the last value is omitted
        "sinoused": sinoused, # should be sliceing in sinogram dimension (start,end,step). If first value is negative, it takes the number of slices from the second value in the middle of the stack.
        "correcttilt": correcttilt,  # tilt dataset
        "tiltcenter_slice": tiltcenter_slice,  # tilt center (x direction)
        "tiltcenter_det": tiltcenter_det,  # tilt center (y direction)
        "angle_offset": angle_offset, # this is the angle offset from our default (270) so that tomopy yields output in the same orientation as previous software (Octopus)
        "anglelist": anglelist, # if not set, will assume evenly spaced angles which will be calculated by the angular range and number of angles found in the file. if set to -1, will read individual angles from each image. alternatively, a list of angles can be passed.
        "doBeamHardening": doBeamHardening, # turn on beam hardening correction, based on "Correction for beam hardening in computed tomography", Gabor Herman, 1979 Phys. Med. Biol. 24 81
        "BeamHardeningCoefficients": BeamHardeningCoefficients,  # 6 values, tomo = a0 + a1*tomo + a2*tomo^2 + a3*tomo^3 + a4*tomo^4 + a5*tomo^5
        "projIgnoreList": projIgnoreList, # projections to be ignored in the reconstruction (for simplicity in the code, they will not be removed and will be processed as all other projections but will be set to zero absorption right before reconstruction.
        "bfexposureratio": bfexposureratio,  # ratio of exposure time of bf to exposure time of sample
        "pxsize": pxsize,
        "numslices": numslices,
        "numangles": numangles,
        "angularrange": angularrange,
        "numrays": numrays,
        "npad": npad,
        "projused": projused,
        "inter_bright": inter_bright,
        "nflat": nflat,
        "ind_flat": ind_flat,
        "ndark": nflat,
        "ind_dark": ind_flat,
        "group_flat": group_flat,
        "ind_tomo": ind_tomo,
        "floc_independent": floc_independent,
        "sinoused": sinoused,
        "BeamHardeningCoefficients": BeamHardeningCoefficients,
        "function_list": function_list,
        "dorecon": dorecon,
        "writenormalized": writenormalized,
        "writereconstruction": writereconstruction,
        "dominuslog": dominuslog,
        "verbose_printing": verbose_printing,
        "recon_algorithm": recon_algorithm,
        "dolensdistortion": dolensdistortion,
        "lensdistortioncenter": lensdistortioncenter,
        "lensdistortionfactors": lensdistortionfactors,
        "minimum_transmission": minimum_transmission,
    }

    #return second variable tomo, (first and last normalized image), to use it for manual COR checking
    return recon_dict, tomo



# to profile memory, uncomment the following line
# and then run program from command line as
# python -m memory_profiler tomopy832.py
# (you have to have memory_profiler installed)
# @profile
def recon(
    filename,
    filetype = 'als',
    timepoint = 0,
    bffilename = None,
    inputPath = './', #input path, location of the data set to reconstruct
    outputFilename = None, #file name for output tif files (a number and .tiff will be added). default is based on input filename
    fulloutputPath = None, # definte the full output path, no automatic sub-folder will be created
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
    doPolarRing2 = False, # ring removal
    Rarc2=30, # min angle needed to be considered ring artifact (ring removal)
    Rmaxwidth2=100, # max width of rings to be filtered (ring removal)
    Rtmax2=3000.0, # max portion of image to filter (ring removal)
    Rthr2=3000.0, # max value of offset due to ring artifact (ring removal)
    Rtmin2=-3000.0, # min value of image to filter (ring removal)
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
    castTo8bit = False, # convert data to 8bit before writing
    cast8bit_min=-10, # min value if converting to 8bit
    cast8bit_max=30, # max value if converting to 8bit
    useNormalize_nf = False, # normalize based on background intensity (nf)
    chunk_proj = 100, # chunk size in projection direction
    chunk_sino = 100, # chunk size in sinogram direction
    npad = None, # amount to pad data before reconstruction
    projused = None, # should be slicing in projection dimension (start,end,step) Be sure to add one to the end as stop in python means the last value is omitted
    sinoused = None, # should be sliceing in sinogram dimension (start,end,step). If first value is negative, it takes the number of slices from the second value in the middle of the stack.
    correcttilt = 0, # tilt dataset
    tiltcenter_slice = None, # tilt center (x direction)
    tiltcenter_det = None, # tilt center (y direction)
    angle_offset = 0, # this is the angle offset from our default (270) so that tomopy yields output in the same orientation as previous software (Octopus)
    anglelist = None, # if not set, will assume evenly spaced angles which will be calculated by the angular range and number of angles found in the file. if set to -1, will read individual angles from each image. alternatively, a list of angles can be passed.
    doBeamHardening = False, # turn on beam hardening correction, based on "Correction for beam hardening in computed tomography", Gabor Herman, 1979 Phys. Med. Biol. 24 81
    BeamHardeningCoefficients = (0, 1, 0, 0, 0, .1), # 6 values, tomo = a0 + a1*tomo + a2*tomo^2 + a3*tomo^3 + a4*tomo^4 + a5*tomo^5
    projIgnoreList = None, # projections to be ignored in the reconstruction (for simplicity in the code, they will not be removed and will be processed as all other projections but will be set to zero absorption right before reconstruction.
    bfexposureratio = 1, #ratio of exposure time of bf to exposure time of sample
    pxsize = 1,
    numslices= 100,
    numangles= 3,
    angularrange= 180,
    numrays= 2560,
    inter_bright= 0,
    nflat= 15,
    ind_flat=1,
    group_flat= None,
    ndrk=10,
    ind_dark=1,
    ind_tomo= [0,1,2],
    floc_independent= 1,
    function_list= ['normalize','minus_log','recon_mask','write_output'],
    dorecon=True,
    writenormalized=False,
    writereconstruction=True,
    dominuslog=True,
    verbose_printing=False,
    recon_algorithm='gridrec', #choose from gridrec, fbp, and others in tomopy
    dolensdistortion=False,
    lensdistortioncenter = (1280,1080),
    lensdistortionfactors = (1.00015076, 1.9289e-06, -2.4325e-08, 1.00439e-11, -3.99352e-15),
    minimum_transmission = 0.01,
    *args, **kwargs
    ):

    start_time = time.time()
    if verbose_printing:
        print("Start {} at:".format(filename)+time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime()))

    filenametowrite = os.path.join(fulloutputPath,outputFilename)
    if verbose_printing:
        print("Time point: {}".format(timepoint))

    tempfilenames = [os.path.join(fulloutputPath,'tmp0.h5'),os.path.join(fulloutputPath,'tmp1.h5')]
    if verbose_printing:
        print("cleaning up previous temp files") #, end="")
    for tmpfile in tempfilenames:
        try:
            os.remove(tmpfile)
        except OSError:
            pass

    numprojused = len(range(projused[0], projused[1],projused[2]))  # number of total projections. We add 1 to include the last projection
    numsinoused = len(range(sinoused[0], sinoused[1],sinoused[2]))  # number of total sinograms. We add 1 to include the last projection
    num_proj_per_chunk = np.minimum(chunk_proj,numprojused)  # sets the chunk size to either all of the projections used or the chunk size
    numprojchunks = (numprojused - 1) // num_proj_per_chunk + 1  # adding 1 fixes the case of the number of projections not being a factor of the chunk size. Subtracting 1 fixes the edge case where the number of projections is a multiple of the chunk size
    num_sino_per_chunk = np.minimum(chunk_sino, numsinoused)  # same as num_proj_per_chunk
    numsinochunks = (numsinoused - 1) // num_sino_per_chunk + 1  # adding 1 fixes the case of the number of sinograms not being a factor of the chunk size. Subtracting 1 fixes the edge case where the number of sinograms is a multiple of the chunk size

    # Figure out first direction to slice
    for func in function_list:
        if slice_dir[func] != 'both':
            axis = slice_dir[func]
            break
    else:
        axis = 'sino'

    done = False
    curfunc = 0
    curtemp = 0

    if not dorecon:
        rec = 0

    while True: # Loop over reading data in certain chunking direction
        if axis=='proj':
            niter = numprojchunks
        else:
            niter = numsinochunks
        for y in range(niter): # Loop over chunks
            if verbose_printing:
                print("{} chunk {} of {}".format(axis, y+1, niter))
            # The standard case. Unless the combinations below are in our function list, we read darks and flats normally, and on next chunck proceed to "else."
            if curfunc == 0 and not (('normalize_nf' in function_list and 'remove_outlier2d' in function_list) or ('remove_outlier1d' in function_list and 'remove_outlier2d' in function_list)):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if axis=='proj':
                        if (filetype=='als'):
                            tomo, flat, dark, floc = dxchange.read_als_832h5(os.path.join(inputPath,filename),ind_tomo=range(y*projused[2]*num_proj_per_chunk+projused[0], np.minimum((y + 1)*projused[2]*num_proj_per_chunk+projused[0],projused[1]),projused[2]),sino=(sinoused[0],sinoused[1],sinoused[2]))
                            if bffilename is not None:
                                tomobf, _, _, _ = dxchange.read_als_832h5(os.path.join(inputPath,bffilename),sino=(sinoused[0],sinoused[1],sinoused[2])) #I don't think we need this for separate bf: ind_tomo=range(y*projused[2]*num_proj_per_chunk+projused[0], np.minimum((y + 1)*projused[2]*num_proj_per_chunk+projused[0],projused[1]),projused[2]),
                                flat = tomobf
                        elif (filetype == 'dxfile'):
                            tomo, flat, dark, _, _= dxchange.read_dx(os.path.join(inputPath, filename), exchange_rank=0,
                                                               proj=( y * projused[2] * num_proj_per_chunk + projused[0],
                                                                     + np.minimum((y + 1) * projused[2] * num_proj_per_chunk + projused[0], projused[1]), projused[2]),
                                                               sino=sinoused)  # dtype=None, , )
                        elif (filetype=='sls'):
                            tomo, flat, dark, _ = read_sls(os.path.join(inputPath,filename),  exchange_rank=0, proj=(timepoint*numangles+y*projused[2]*num_proj_per_chunk+projused[0],timepoint*numangles+np.minimum((y + 1)*projused[2]*num_proj_per_chunk+projused[0],projused[1]),projused[2]), sino=sinoused) #dtype=None, , )
                        else:
                            break
                    else:
                        if (filetype == 'als'):
                            tomo, flat, dark, floc = dxchange.read_als_832h5(os.path.join(inputPath,filename),ind_tomo=range(projused[0],projused[1],projused[2]),sino=(y*sinoused[2]*num_sino_per_chunk+sinoused[0],np.minimum((y + 1)*sinoused[2]*num_sino_per_chunk+sinoused[0],sinoused[1]),sinoused[2]))
                            if bffilename is not None:
                                tomobf, _, _, _ = dxchange.read_als_832h5(os.path.join(inputPath, bffilename),sino=(y*sinoused[2]*num_sino_per_chunk+sinoused[0],np.minimum((y + 1)*sinoused[2]*num_sino_per_chunk+sinoused[0],sinoused[1]),sinoused[2])) # I don't think we need this for separate bf: ind_tomo=range(projused[0],projused[1],projused[2]),
                                flat = tomobf
                        elif (filetype == 'dxfile'):
                                tomo, flat, dark, _, _ = dxchange.read_dx(os.path.join(inputPath, filename), exchange_rank=0,
                                                               proj=( projused[0],
                                                                      projused[1], projused[2]),
                                                               sino=(y * sinoused[2] * num_sino_per_chunk + sinoused[0],
                                                                     np.minimum(
                                                                         (y + 1) * sinoused[2] * num_sino_per_chunk +
                                                                         sinoused[0], sinoused[1]),
                                                                     sinoused[2]))  # dtype=None, , )
                        elif (filetype=='sls'):
                            tomo, flat, dark, _ = read_sls(os.path.join(inputPath,filename),  exchange_rank=0, proj=(timepoint*numangles+projused[0],timepoint*numangles+projused[1],projused[2]), sino=(y*sinoused[2]*num_sino_per_chunk+sinoused[0],np.minimum((y + 1)*sinoused[2]*num_sino_per_chunk+sinoused[0],sinoused[1]),sinoused[2])) #dtype=None, , )
                        else:
                            break
            # Handles the initial reading of scans. Flats and darks are not read in, because the chunking direction will swap before we normalize. We read in darks when we normalize.
            elif curfunc == 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if axis=='proj':
                        if (filetype == 'als'):
                            tomo = read_als_832h5_tomo_only(os.path.join(inputPath,filename),ind_tomo=range(y*projused[2]*num_proj_per_chunk+projused[0], np.minimum((y + 1)*projused[2]*num_proj_per_chunk+projused[0],projused[1]),projused[2]),sino=(sinoused[0],sinoused[1], sinoused[2]))
                        elif (filetype == 'dxfile'):
                            tomo, _, _, _, _ = read_sls(os.path.join(inputPath, filename), exchange_rank=0, proj=(
                            y * projused[2] * num_proj_per_chunk + projused[0],
                            np.minimum((y + 1) * projused[2] * num_proj_per_chunk + projused[0],
                                                               projused[1]), projused[2]),
                                                     sino=sinoused)  # dtype=None, , )
                        elif (filetype=='sls'):
                            tomo, _, _, _ = read_sls(os.path.join(inputPath,filename),  exchange_rank=0, proj=(timepoint*numangles+y*projused[2]*num_proj_per_chunk+projused[0],timepoint*numangles+np.minimum((y + 1)*projused[2]*num_proj_per_chunk+projused[0],projused[1]),projused[2]), sino=sinoused) #dtype=None, , )
                        else:
                            break
                    else:
                        if (filetype == 'als'):
                            tomo = read_als_832h5_tomo_only(os.path.join(inputPath,filename),ind_tomo=range(projused[0],projused[1],projused[2]),sino=(y*sinoused[2]*num_sino_per_chunk+sinoused[0],np.minimum((y + 1)*sinoused[2]*num_sino_per_chunk+sinoused[0],sinoused[1]),sinoused[2]))
                        elif (filetype == 'dxfile'):
                            tomo, _, _, _, _ = dxchange.read_dx(os.path.join(inputPath, filename), exchange_rank=0, proj=(
                             projused[0], projused[1], projused[2]),
                                                     sino=(y * sinoused[2] * num_sino_per_chunk + sinoused[0],
                                                           np.minimum(
                                                               (y + 1) * sinoused[2] * num_sino_per_chunk + sinoused[0],
                                                               sinoused[1]), sinoused[2]))  # dtype=None, , )
                        elif (filetype=='sls'):
                            tomo, _, _, _ = read_sls(os.path.join(inputPath,filename),  exchange_rank=0, proj=(timepoint*numangles+projused[0],timepoint*numangles+projused[1],projused[2]), sino=(y*sinoused[2]*num_sino_per_chunk+sinoused[0],np.minimum((y + 1)*sinoused[2]*num_sino_per_chunk+sinoused[0],sinoused[1]),sinoused[2])) #dtype=None, , )
                        else:
                            break
            # Handles the reading of darks and flats, once we know the chunking direction will not change before normalizing.
            elif ('remove_outlier2d' == function_list[curfunc] and 'normalize' in function_list) or 'normalize_nf' == function_list[curfunc]:
                if axis == 'proj':
                    start, end = y * num_proj_per_chunk, np.minimum((y + 1) * num_proj_per_chunk,numprojused)
                    tomo = dxchange.reader.read_hdf5(tempfilenames[curtemp],'/tmp/tmp',slc=((start,end,1),(0,numslices,1),(0,numrays,1))) #read in intermediate file
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if (filetype == 'als'):
                            flat, dark, floc = read_als_832h5_non_tomo(os.path.join(inputPath,filename),ind_tomo=range(y*projused[2]*num_proj_per_chunk+projused[0], np.minimum((y + 1)*projused[2]*num_proj_per_chunk+projused[0],projused[1]),projused[2]),sino=(sinoused[0],sinoused[1], sinoused[2]))
                            if bffilename is not None:
                                tomobf, _, _, _ = dxchange.read_als_832h5(os.path.join(inputPath,bffilename),sino=(sinoused[0],sinoused[1], sinoused[2])) #I don't think we need this since it is full tomo in separate file: ind_tomo=range(y*projused[2]*num_proj_per_chunk+projused[0], np.minimum((y + 1)*projused[2]*num_proj_per_chunk+projused[0],projused[1]),projused[2])
                                flat = tomobf
                        elif (filetype == 'dxfile'):
                                _, flat, dark, _, _ = dxchange.read_dx(os.path.join(inputPath, filename), exchange_rank=0, proj=(
                                y * projused[2] * num_proj_per_chunk + projused[0],
                                 np.minimum(
                                    (y + 1) * projused[2] * num_proj_per_chunk + projused[0], projused[1]),
                                projused[2]), sino=sinoused)  # dtype=None, , )
                        elif (filetype=='sls'):
                            _, flat, dark, _ = read_sls(os.path.join(inputPath,filename),  exchange_rank=0, proj=(timepoint*numangles+y*projused[2]*num_proj_per_chunk+projused[0],timepoint*numangles+np.minimum((y + 1)*projused[2]*num_proj_per_chunk+projused[0],projused[1]),projused[2]), sino=sinoused) #dtype=None, , )
                        else:
                            break
                else:
                    start, end = y * num_sino_per_chunk, np.minimum((y + 1) * num_sino_per_chunk,numsinoused)
                    tomo = dxchange.reader.read_hdf5(tempfilenames[curtemp],'/tmp/tmp',slc=((0,numangles,1),(start,end,1),(0,numrays,1)))
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if (filetype == 'als'):
                            flat, dark, floc = read_als_832h5_non_tomo(os.path.join(inputPath,filename),ind_tomo=range(projused[0],projused[1],projused[2]),sino=(y*sinoused[2]*num_sino_per_chunk+sinoused[0],np.minimum((y + 1)*sinoused[2]*num_sino_per_chunk+sinoused[0],sinoused[1]),sinoused[2]))
                        elif (filetype == 'dxfile'):
                            _, flat, dark, _, _ = read_sls(os.path.join(inputPath, filename), exchange_rank=0, proj=(
                             projused[0],  projused[1], projused[2]),
                                                        sino=(y * sinoused[2] * num_sino_per_chunk + sinoused[0],
                                                              np.minimum(
                                                                  (y + 1) * sinoused[2] * num_sino_per_chunk + sinoused[
                                                                      0], sinoused[1]), sinoused[2]))  # dtype=None, , )
                        elif (filetype=='sls'):
                            _, flat, dark, _ = read_sls(os.path.join(inputPath,filename),  exchange_rank=0, proj=(timepoint*numangles+projused[0],timepoint*numangles+projused[1],projused[2]), sino=(y*sinoused[2]*num_sino_per_chunk+sinoused[0],np.minimum((y + 1)*sinoused[2]*num_sino_per_chunk+sinoused[0],sinoused[1]),sinoused[2])) #dtype=None, , )
                        else:
                            break
            # Anything after darks and flats have been read or the case in which remove_outlier2d is the current/2nd function and the previous case fails.
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
                if verbose_printing:
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
                    if bfexposureratio != 1:
                        if verbose_printing:
                            print("correcting bfexposureratio")
                        tomo = tomo * bfexposureratio
                elif func_name == 'normalize':
                    tomo = tomo.astype(np.float32,copy=False)
                    tomopy.normalize(tomo, flat, dark, out=tomo)
                    if bfexposureratio != 1:
                        tomo = tomo * bfexposureratio
                        if verbose_printing:
                            print("correcting bfexposureratio")
                elif func_name == 'minus_log':
                    mx = np.float32(minimum_transmission) #setting min %transmission to 1% helps avoid streaking from very high absorbing areas
                    ne.evaluate('where(tomo>mx, tomo, mx)', out=tomo)
                    tomopy.minus_log(tomo, out=tomo)
                elif func_name == 'beam_hardening':
                    loc_dict = {'a{}'.format(i):np.float32(val) for i,val in enumerate(BeamHardeningCoefficients)}
                    loc_dict['tomo'] = tomo
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

                    # add padding of 10 pixels, to be unpadded right after tilt correction.
                    # This makes the tilted image not have zeros at certain edges,
                    # which matters in cases where sample is bigger than the field of view.
                    # For the small amounts we are generally tilting the images, 10 pixels is sufficient.
                    #  tomo = tomopy.pad(tomo, 2, npad=10, mode='edge')
                    #  center_det = center_det + 10

                    cntr = (center_det, new_center)
                    for b in range(tomo.shape[0]):
                        tomo[b] = st.rotate(tomo[b], correcttilt, center=cntr, preserve_range=True, order=1, mode='edge', clip=True) # center=None means image is rotated around its center; order=1 is default, order of spline interpolation
#					tomo = tomo[:, :, 10:-10]
                elif func_name == 'lensdistortion':
                    print(lensdistortioncenter[0])
                    print(lensdistortioncenter[1])
                    print(lensdistortionfactors[0])
                    print(lensdistortionfactors[1])
                    print(lensdistortionfactors[2])
                    print(lensdistortionfactors[3])
                    print(lensdistortionfactors[4])
                    print(type(lensdistortionfactors[0]))
                    print(type(lensdistortionfactors))
                    tomo = tomopy.prep.alignment.distortion_correction_proj(tomo, lensdistortioncenter[0], lensdistortioncenter[1], lensdistortionfactors, ncore=None,nchunk=None)
                elif func_name == 'do_360_to_180':

                    # Keep values around for processing the next chunk in the list
                    keepvalues = [angularrange, numangles, projused, num_proj_per_chunk, numprojchunks, numprojused, numrays, anglelist]

                    # why -.5 on one and not on the other?
                    if tomo.shape[0]%2>0:
                        tomo = sino_360_to_180(tomo[0:-1,:,:], overlap=int(np.round((tomo.shape[2]-cor-.5))*2), rotation='right')
                        angularrange = angularrange/2 - angularrange/(tomo.shape[0]-1)
                    else:
                        tomo = sino_360_to_180(tomo[:,:,:], overlap=int(np.round((tomo.shape[2]-cor))*2), rotation='right')
                        angularrange = angularrange/2
                    numangles = int(numangles/2)
                    projused = (0,numangles-1,1)
                    numprojused = len(range(projused[0],projused[1],projused[2]))
                    num_proj_per_chunk = np.minimum(chunk_proj,numprojused)
                    numprojchunks = (numprojused-1)//num_proj_per_chunk+1
                    numrays = tomo.shape[2]

                    anglelist = anglelist[:numangles]

                elif func_name == 'phase_retrieval':
                    tomo = tomopy.retrieve_phase(tomo, pixel_size=pxsize, dist=propagation_dist, energy=kev, alpha=alphaReg, pad=True)
                
                elif func_name == 'translation_correction':
                    tomo = linear_translation_correction(tomo,dx=xshift,dy=yshift,interpolation=False)
                    
                elif func_name == 'recon_mask':
                    tomo = tomopy.pad(tomo, 2, npad=npad, mode='edge')

                    if projIgnoreList is not None:
                        for badproj in projIgnoreList:
                            tomo[badproj] = 0
                    rec = tomopy.recon(tomo, anglelist, center=cor+npad, algorithm=recon_algorithm, filter_name='butterworth', filter_par=[butterworth_cutoff, butterworth_order])
                    rec = rec[:, npad:-npad, npad:-npad]
                    rec /= pxsize  # convert reconstructed voxel values from 1/pixel to 1/cm
                    rec = tomopy.circ_mask(rec, 0)
                    tomo = tomo[:, :, npad:-npad]
                elif func_name == 'polar_ring':
                    rec = np.ascontiguousarray(rec, dtype=np.float32)
                    rec = tomopy.remove_ring(rec, theta_min=Rarc, rwidth=Rmaxwidth, thresh_max=Rtmax, thresh=Rthr, thresh_min=Rtmin,out=rec)
                elif func_name == 'polar_ring2':
                    rec = np.ascontiguousarray(rec, dtype=np.float32)
                    rec = tomopy.remove_ring(rec, theta_min=Rarc2, rwidth=Rmaxwidth2, thresh_max=Rtmax2, thresh=Rthr2, thresh_min=Rtmin2,out=rec)
                elif func_name == 'castTo8bit':
                    rec = convert8bit(rec, cast8bit_min, cast8bit_max)
                elif func_name == 'write_reconstruction':
                    if dorecon:
                        if sinoused[2] == 1:
                            dxchange.write_tiff_stack(rec, fname=filenametowrite, start=y*num_sino_per_chunk + sinoused[0])
                        else:
                            num = y*sinoused[2]*num_sino_per_chunk+sinoused[0]
                            for sinotowrite in rec:    #fixes issue where dxchange only writes for step sizes of 1
                                dxchange.writer.write_tiff(sinotowrite, fname=filenametowrite + '_' + '{0:0={1}d}'.format(num, 5))
                                num += sinoused[2]
                    else:
                        if verbose_printing:
                            print('Reconstruction was not done because dorecon was set to False.')
                elif func_name == 'write_normalized':
                    if projused[2] == 1:
                        dxchange.write_tiff_stack(tomo, fname=filenametowrite+'_norm', start=y * num_proj_per_chunk + projused[0])
                    else:
                        num = y * projused[2] * num_proj_per_chunk + projused[0]
                        for projtowrite in tomo:  # fixes issue where dxchange only writes for step sizes of 1
                            dxchange.writer.write_tiff(projtowrite,fname=filenametowrite + '_' + '{0:0={1}d}_norm'.format(num, 5))
                            num += projused[2]
                if verbose_printing:
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
    if verbose_printing:
        print("cleaning up temp files")
    for tmpfile in tempfilenames:
        try:
            os.remove(tmpfile)
        except OSError:
            pass
    if verbose_printing:
        print("End Time: "+time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime()))
        print('It took {:.3f} s to process {}'.format(time.time()-start_time,os.path.join(inputPath,filename)))
    return rec, tomo

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

###############################################################################################
# New Readers, so we don't have to read in darks and flats until they're needed
###############################################################################################
# Tomo
###############################################################################################

def read_als_832h5_tomo_only(fname, ind_tomo=None, ind_flat=None, ind_dark=None,
                   proj=None, sino=None):
    """
    Read ALS 8.3.2 hdf5 file with stacked datasets.

    Parameters
    ----------
    See docs for read_als_832h5
    """

    with dxchange.reader.find_dataset_group(fname) as dgroup:
        dname = dgroup.name.split('/')[-1]

        tomo_name = dname + '_0000_0000.tif'

        # Read metadata from dataset group attributes
        keys = list(dgroup.attrs.keys())
        if 'nangles' in keys:
            nproj = int(dgroup.attrs['nangles'])

        # Create arrays of indices to read projections
        if ind_tomo is None:
            ind_tomo = list(range(0, nproj))
        if proj is not None:
            ind_tomo = ind_tomo[slice(*proj)]

        tomo = dxchange.reader.read_hdf5_stack(
            dgroup, tomo_name, ind_tomo, slc=(None, sino))

    return tomo


#####################################################################################
# Non tomo
#####################################################################################

def read_als_832h5_non_tomo(fname, ind_tomo=None, ind_flat=None, ind_dark=None,
                   proj=None, sino=None):
    """
    Read ALS 8.3.2 hdf5 file with stacked datasets.

    Parameters
    ----------
    See docs for read_als_832h5
    """

    with dxchange.reader.find_dataset_group(fname) as dgroup:
        dname = dgroup.name.split('/')[-1]

        flat_name = dname + 'bak_0000.tif'
        dark_name = dname + 'drk_0000.tif'

        # Read metadata from dataset group attributes
        keys = list(dgroup.attrs.keys())
        if 'nangles' in keys:
            nproj = int(dgroup.attrs['nangles'])
        if 'i0cycle' in keys:
            inter_bright = int(dgroup.attrs['i0cycle'])
        if 'num_bright_field' in keys:
            nflat = int(dgroup.attrs['num_bright_field'])
        else:
            nflat = dxchange.reader._count_proj(dgroup, flat_name, nproj,
                                         inter_bright=inter_bright)
        if 'num_dark_fields' in keys:
            ndark = int(dgroup.attrs['num_dark_fields'])
        else:
            ndark = dxchange.reader._count_proj(dgroup, dark_name, nproj)

        # Create arrays of indices to read projections, flats and darks
        if ind_tomo is None:
            ind_tomo = list(range(0, nproj))
        if proj is not None:
            ind_tomo = ind_tomo[slice(*proj)]
        ind_dark = list(range(0, ndark))
        group_dark = [nproj - 1]
        ind_flat = list(range(0, nflat))

        if inter_bright > 0:
            group_flat = list(range(0, nproj, inter_bright))
            if group_flat[-1] != nproj - 1:
                group_flat.append(nproj - 1)
        elif inter_bright == 0:
            group_flat = [0, nproj - 1]
        else:
            group_flat = None

        flat = dxchange.reader.read_hdf5_stack(
            dgroup, flat_name, ind_flat, slc=(None, sino), out_ind=group_flat)

        dark = dxchange.reader.read_hdf5_stack(
            dgroup, dark_name, ind_dark, slc=(None, sino), out_ind=group_dark)

    return flat, dark, dxchange.reader._map_loc(ind_tomo, group_flat)

######################################################################################################


def read_sls(fname, exchange_rank=0, proj=None, sino=None, dtype=None):
    """
    Read sls time resolved data format.

    Parameters
    ----------
    fname : str
        Path to hdf5 file.

    exchange_rank : int, optional
        exchange_rank is added to "exchange" to point tomopy to the data
        to reconstruct. if rank is not set then the data are raw from the
        detector and are located under exchange = "exchange/...", to process
        data that are the result of some intemedite processing step then
        exchange_rank = 1, 2, ... will direct tomopy to process
        "exchange1/...",

    proj : {sequence, int}, optional
        Specify projections to read. (start, end, step)

    sino : {sequence, int}, optional
        Specify sinograms to read. (start, end, step)

    dtype : numpy datatype, optional
        Convert data to this datatype on read if specified.

    ind_tomo : list of int, optional
        Indices of the projection files to read.

    Returns
    -------
    ndarray
        3D tomographic data.

    ndarray
        3D flat field data.

    ndarray
        3D dark field data.

    ndarray
        1D theta in radian.
    """
    if exchange_rank > 0:
        exchange_base = 'exchange{:d}'.format(int(exchange_rank))
    else:
        exchange_base = "exchange"

    tomo_grp = '/'.join([exchange_base, 'data'])
    flat_grp = '/'.join([exchange_base, 'data_white'])
    dark_grp = '/'.join([exchange_base, 'data_dark'])
    theta_grp = '/'.join([exchange_base, 'theta'])

    tomo = dxchange.read_hdf5(fname, tomo_grp, slc=(proj, sino), dtype=dtype)
    flat = dxchange.read_hdf5(fname, flat_grp, slc=(None, sino), dtype=dtype)
    dark = dxchange.read_hdf5(fname, dark_grp, slc=(None, sino), dtype=dtype)
    theta = dxchange.read_hdf5(fname, theta_grp)

    if (theta is None):
        theta_grp = '/'.join([exchange_base, 'theta_aborted'])
        theta = dxchange.read_hdf5(fname, theta_grp)
    if (theta is None):
        if verbose_printing:
            print('could not find thetas, generating them based on 180 degree rotation')
        theta_size = dxchange.read_dx_dims(fname, 'data')[0]
        logger.warn('Generating "%s" [0-180] deg angles for missing "exchange/theta" dataset' % (str(theta_size)))
        theta = np.linspace(0., 180., theta_size)

    theta = theta * np.pi / 180.

    if proj is not None:
        theta = theta[proj[0]:proj[1]:proj[2]]

    return tomo, flat, dark, theta

#Converts spreadsheet.xlsx file with headers into dictionaries
# def read_spreadsheet(filepath):
#     workbook=xlrd.open_workbook(filepath)
#     worksheet = workbook.sheet_by_index(0)
#
#     # imports first row and converts to a list of header strings
#     headerList = []
#     for col_index in range(worksheet.ncols):
#         headerList.append(str(worksheet.cell_value(0,col_index)))
#
#     dataList = []
#     # For each row, create a dictionary and like header name to data
#     # converts each row to following format rowDictionary1 ={'header1':colvalue1,'header2':colvalue2,... }
#     # compiles rowDictinaries into a list: dataList = [rowDictionary1, rowDictionary2,...]
#     for row_index in range(1,worksheet.nrows):
#         rowDictionary = {}
#         for col_index in range(worksheet.ncols):
#             cellValue = worksheet.cell_value(row_index,col_index)
#
#             if type(cellValue)==unicode:
#                 cellValue = str(cellValue)
#
#             # if cell contains string that looks like a tuple, convert to tuple
#             if '(' in str(cellValue):
#                 cellValue = literal_eval(cellValue)
#
#             # if cell contains string or int that looks like 'True', convert to boolean True
#             if str(cellValue).lower() =='true' or (type(cellValue)==int and cellValue==1):
#                 cellValue = True
#
#             # if cell contains string or int that looks like 'False', convert to boolean False
#             if str(cellValue).lower() =='false' or (type(cellValue)==int and cellValue==0):
#                 cellValue = False
#
#             if cellValue != '': # create dictionary element if cell value is not empty
#                 rowDictionary[headerList[col_index]] = cellValue
#         dataList.append(rowDictionary)
#
#     return(dataList)


# D.Y.Parkinson's interpreter for text input files
def main():
    parametersfile = 'input832.txt' if (len(sys.argv)<2) else sys.argv[1]

    if parametersfile.split('.')[-1] == 'txt':
        with open(parametersfile,'r') as theinputfile:
            theinput = theinputfile.read()
            inputlist = theinput.splitlines()
            for reconcounter in range(0,len(inputlist)):
                inputlisttabsplit = inputlist[reconcounter].split()
                if inputlisttabsplit:
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
                else:
                    print("Ending at blank line in input.")
                    break
                print("Read user input:")
                print(functioninput)
                recon_dictionary, _ = recon_setup(**functioninput)
#                recon(**functioninput)
                recon(**recon_dictionary)

# H.S.Barnard Spreadsheet interpreter
#     if parametersfile.split('.')[-1]=='xlsx':
#         functioninput = read_spreadsheet(parametersfile)
#         for i in range(len(functioninput)):
#             recon(**functioninput[i])

if __name__ == '__main__':
    main()
        
