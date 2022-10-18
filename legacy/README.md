# python-alsmicroct

This repo contains python code for reconstructing and processing x-ray micro tomography (microCT) datasets generated at the advanced light source (ALS).

## Overview:
[Beamline 8.3.2][BL832] is located at the Advanced Light Source [(ALS)][ALS] at Lawrence Berkeley National Laboratory [(LBNL)][LBNL]. 8.3.2 is a synchrotron x-ray endstation that is devoted to x-ray microtomography. This repository provides a toolkit for reconstruction and processing data produced by the x-ray tomography performed at 8.3.2.

Tomography analysis requires proessing of large (many GB) datasets that contain 32-bit image stacks. Standard raw datasets contain X-ray radiographs/images/projections) of a sample taken at many angles as the sample is rotated w.r.t. the imaging system. A tomographic reconstruction algorithm is used to convert the multiple views into a 3D dataset/imagestack. These python tools utilize the [TomoPy][TomoPy] libraries, deveoloped at Argon National Laboratory, to perform reconstructions. Additionally, postprocessing tools using numpy, scipy, and matplotlib libaries are included.

[LBNL]:http://www.lbl.gov/
[ALS]:https://www-als.lbl.gov/
[BL832]:http://microct.lbl.gov/
[TomoPy]:https://tomopy.readthedocs.io/en/latest/index.html

## About TomoPy:
[TomoPy][TomoPy] is an open-source Python package for tomographic data processing and image reconstruction. Anaconda is the prefered python distribution for using TomoPy it can be installed directly from TomoPy's [conda channel][TomoPyConda].

[TomoPyConda]:https://anaconda.org/dgursoy/tomopy

## Repo File Structure

`alsmicroct/` contains functions and classes used for managing data pipeline and data analysis

`applications/` contains programs that utilize functions from alsmicroct and tomopy.

`filepath/` contains input files that provide lists of paths the datasets that are to be reconstructed or processed.

`batch/` contains .slurm scripts that are used to submit jobs to the [NERSC] supercomputer.

`tests/` contains test cases for functions in alsmicroct.

[NERSC]: http://www.nersc.gov/


## SPOT Suite API Commands

SPOT Suite is an online platform for storage and datamanagement of ALS data on NERSC. In this library commands commands and data requests can be sent to the SPOT API using functions contained in a class called `SpotSession`. This class utilizes python's `requests` library to communicate with the API. This class is available in `microct_toolbox.data_management` toolkit.

#### `SpotSession(username='default')`

Creating an instance of the `SpotSession()` class establishes a session with the SPOT API. The user will be prompted for username and password.

`username` will default to your spot username, however, you can enter a different username if you are not the owner of the dataset that you will request. You will permission to access data for which are not the owner.

```python
import sys
sys.path.append("[local path]/python-TomographyTools")
import microct_toolbox.data_management as dm

s = dm.SpotSession()

r = s.search("my_search_term") # returns list with one JSON object for each search result
```

##### `SpotSession()` built in functions:

**Check authentication:** 
`check_authentication()` returns True if authentication is active

**Authenticate if connection is lost:**
`authentication()` prompts user for username and password for current session and reconnects.

**Search:** find filenames containing `query` string.
```python
search(query,                    # input search string
	limitnum = 10,               # number of results to show
	skipnum = 0,                 # number of results to skip
	sortterm = "fs.stage_date",  # database field on which to sort
	sorttype = "desc")           # sorttype: desc or asc
```

**Derived Datasets:** 
`derived_datasets(self,dataset)` Finds derived datasets (norm, sino, gridrec, imgrec) from raw dataset. Returns `json` type object

**Request metadata:**
`attributes(self,dataset,username='default)` Requests metadata for dataset.

**Stage dataset from tape storage:**
`stage(self,dataset,username='default')` Stage dataset from tape to disk if required (data is stored long term on tape drives and must be transferred to disk for use).   


**Download raw dataset from SPOT:**

```python
download(dataset,			    # Name of dataset
	username='default',		    # username of dataset owner, defaults to spot login username
	downloadPath='default',		# download destination, defaults to pwd
	downloadName='default')		# download name filename, defaults to name of dataset
```


  
  
## Reconstruction

The reconstruction module serves a wrapper to interface with the the tomopy libraries. The primary function in the module is `recon()` with provides access to a wide range of tomopy's functionality through arguemnts passed into `recon([arguments])`. The `recon()` function is commonly used as follows:

```python
import sys
sys.path.append("[local path]/python-TomographyTools")
from microct_toolbox.reconstruction import recon

recon([dataset],cor=[Center of Rotation])
```

Arguments passed in to `recon` function are listed below:


```python
recon(
	filename,
	inputPath = './',
	outputPath = None,
	outputFilename = None,
	doOutliers1D = False,       # outlier removal in 1d (along sinogram columns)
	outlier_diff1D = 750,       # difference between good data and outlier data (outlier removal)
	outlier_size1D = 3,         # radius around each pixel to look for outliers (outlier removal)
	doOutliers2D = False,       # outlier removal, standard 2d on each projection
	outlier_diff2D = 750,       # difference between good data and outlier data (outlier removal)
	outlier_size2D = 3,         # radius around each pixel to look for outliers (outlier removal)
	doFWringremoval = True,     # Fourier-wavelet ring removal
	doTIringremoval = False,    # Titarenko ring removal
	doSFringremoval = False,    # Smoothing filter ring removal
	ringSigma = 3,              # damping parameter in Fourier space (Fourier-wavelet ring removal)
	ringLevel = 8,              # number of wavelet transform levels (Fourier-wavelet ring removal)
	ringWavelet = 'db5',        # type of wavelet filter (Fourier-wavelet ring removal)
	ringNBlock = 0,             # used in Titarenko ring removal (doTIringremoval)
	ringAlpha = 1.5,            # used in Titarenko ring removal (doTIringremoval)
	ringSize = 5,               # used in smoothing filter ring removal (doSFringremoval)
	doPhaseRetrieval = False,   # phase retrieval
	alphaReg = 0.0002,		    # smaller = smoother (used for phase retrieval)
	propagation_dist = 75,      # sample-to-scintillator distance (phase retrieval)
	kev = 24,                   # energy level (phase retrieval)
	butterworth_cutoff = 0.25, 	# 0.1 would be very smooth, 0.4 would be very grainy (reconstruction)
	butterworth_order = 2, 		# for reconstruction
	doPolarRing = False,        # ring removal
	Rarc = 30,                  # min angle needed to be considered ring artifact (ring removal)
	Rmaxwidth = 100,            # max width of rings to be filtered (ring removal)
	Rtmax = 3000.0,             # max portion of image to filter (ring removal)
	Rthr = 3000.0,              # max value of offset due to ring artifact (ring removal)
	Rtmin = -3000.0,            # min value of image to filter (ring removal)
	cor = None,                 # center of rotation (float). If not used then cor will be detected automatically
	corFunction = 'pc',         # center of rotation function to use - can be 'pc', 'vo', or 'nm'
	voInd = None,               # index of slice to use for cor search (vo)
	voSMin = -40,               # min radius for searching in sinogram (vo)
	voSMax = 40,                # max radius for searching in sinogram (vo)
	voSRad = 10,                # search radius (vo)
	voStep = 0.5,               # search step (vo)
	voRatio = 2.0,              # ratio of field-of-view and object size (vo)
	voDrop = 20,                # drop lines around vertical center of mask (vo)
	nmInd = None,               # index of slice to use for cor search (nm)
	nmInit = None,              # initial guess for center (nm)
	nmTol = 0.5,                # desired sub-pixel accuracy (nm)
	nmMask = True,              # if True, limits analysis to circular region (nm)
	nmRatio = 1.0,              # ratio of radius of circular mask to edge of reconstructed image (nm)
	nmSinoOrder = False,        # if True, analyzes in sinogram space. If False, analyzes in radiograph space
	use360to180 = False,        # use 360 to 180 conversion
	doBilateralFilter = False,  # if True, bilateral filter applied to image just before write step # NOTE: image will be converted to 8bit if it is not already
	bilateral_srad = 3,         # spatial radius for bilateral filter (image will be converted to 8bit if not already)
	bilateral_rrad = 30,        # range radius for bilateral filter (image will be converted to 8bit if not already)
	castTo8bit = False,         # convert data to 8bit before writing
	cast8bit_min=-10,           # min value if converting to 8bit
	cast8bit_max=30,            # max value if converting to 8bit
	useNormalize_nf = False,    # normalize based on background intensity (nf)
	chunk_proj = 100,           # chunk size in projection direction
	chunk_sino = 100,           # chunk size in sinogram direction
	npad = None,                # amount to pad data before reconstruction
	projused = None,            #should be slicing in projection dimension (start,end,step)
	sinoused = None,            #should be sliceing in sinogram dimension (start,end,step). If first value is negative, it takes the number of slices from the second value in the middle of the stack.
	correcttilt = 0,            #tilt dataset
	tiltcenter_slice = None,    # tilt center (x direction)
	tiltcenter_det = None,      # tilt center (y direction)
	angle_offset = 0,           #this is the angle offset from our default (270) so that tomopy yields output in the same orientation as previous software (Octopus)
	anglelist = None,           #if not set, will assume evenly spaced angles which will be calculated by the angular range and number of angles found in the file. if set to -1, will read individual angles from each image. alternatively, a list of angles can be passed.
	doBeamHardening = False,     #turn on beam hardening correction, based on "Correction for beam hardening in computed tomography", Gabor Herman, 1979 Phys. Med. Biol. 24 81
	BeamHardeningCoefficients = None, #6 values, tomo = a0 + a1*tomo + a2*tomo^2 + a3*tomo^3 + a4*tomo^4 + a5*tomo^5
	projIgnoreList = None,      #projections to be ignored in the reconstruction (for simplicity in the code, they will not be removed and will be processed as all other projections but will be set to zero absorption right before reconstruction.
	):
```

## Image Processing

The `image_processing` module contains functions for manipulating image files or reconstructed data. Basic functions like downsampling from 32 bit to 8 bit, scaling, cropping, etc. are included.

**Convert Data to 8-bit**

`convert_DirectoryTo8Bit()` Imports files from a directory, linearly rescales pixels between specified min and max values, converts of 8-bit pixels (0-255), then saves files in a specified output directory. 

```python
convert_DirectoryTo8Bit(inputpath='./', 	# path to input directory
	data_min = -10.0, 						# minimum pixel value in 32 bit image
    data_max = 10.0, 						# maximum pixel value in 32 bit image
    basepath = None,                        # prepends input path
    outputpath = None,                      # path to output directory
    filename=None)							# base name for each image file
```

If outputpath is not specified, output path is set to inputpath appended with "_8bit". If outputpath does not exist, one will be created. If filename is not specified, filename is set to the original filename appended with "_8bit".

`convert_ArrayTo8bit(inputarray,data_min,data_max)` Takes a numpy array 2D or 3D numpy array, rescales between specified min and max pixel values, then converts to array of 8-bit integers (0-255).


**Crop Data**

`crop_Directory()` loads directory and crops images to specified x,y,z ranges.

```python
crop_Directory(inputpath='./',                # takes path to input directory
    xRange=(0,None),                # min and max crop range in x, default is entire x range
    yRange=(0,None),                # min and max crop range in y, default is entire y range
    zRange=(0,None),                # min and max crop range in z, default is entire file list
    outputpath=None);                # output path: default is inputpath + "_cropped"
```

`crop_Array()` takes a 2D or 3D numpy array and crops to specified x,y,z ranges.

```python
crop_Array(inputarray,                # takes 2D or 3D numpy array input
    xRange=(0,None),                # min and max crop range in x, default is entire x range
    yRange=(0,None),                # min and max crop range in x, default is entire x range
    zRange=(0,None))                # min and max crop range in z, default is entire file list
```

**Load Data**

`get_fileList(inputpath='./', extensions=("tif","tiff"))` Finds all files in a directory `inputpath` having a extension specified by `extensions`. Default extensions are "tif" and "tiff".

`load_TiffStack(filepath='./',imagerange='all')` 

Loads files in the directory `filepath`. `imagerange` gives the range of images to upload. The default value is 'all' however smaller ranges can be specified with a tuple such that `imagerange=(firstImage,lastImage)`. For large 32 bit images stacks this function may run into memory limitations.


