# functions for post processing datasets
import glob
import numpy as np
import os
from skimage import io
import skimage.external.tifffile as skTiff
import numexpr as ne # routines for the fast evaluation of array expressions elementwise by using a vector-based virtual machine
# =============================================================================

# -----------------------------------------------------------------------------
# generates list of all files in a directory with the desired extension
def get_fileList(inputpath='./', extensions=("tif","tiff")):
    inputpath = inputpath.rstrip('/')
    fileList=[]

    # iterate over file extensions using glob to find files in directory
    for iExt in range(len(extensions)):
        searchterm = inputpath+'/*.'+extensions[iExt].lstrip('.')
        files = glob.glob(searchterm)
        fileList.extend(files)
    return(fileList)

# -----------------------------------------------------------------------------
# loads images in directory into a numpy array
def load_DataStack(filepath='./',imagerange='all'):
    #Imports Tomography Dataset
    fileList = get_fileList(filepath)

    if imagerange=="all" or imagerange=="All" or imagerange=="ALL":
        imageMin = 0
        imageMax = len(fileList)
    else:
        imageMin = imagerange[0]
        imageMax = imagerange[1]

    image = io.imread(fileList[0])
    rows,cols = image.shape

    dataset = np.zeros((imageMax-imageMin,rows, cols))

    for iImage in range(imageMin,imageMax):
        dataset[iImage-imageMin,:,:] = io.imread(fileList[iImage])
        print(iImage)
    return dataset    

# -----------------------------------------------------------------------------

def convert_DirectoryTo8Bit(inputpath='./', data_min=-10.0, data_max=10.0, basepath=None, outputpath=None,filename=None):

    if type(basepath)==str:
        inputpath = basepath.rstrip('/') +'/'+inputpath.lstrip('/')

    fileList= get_fileList(inputpath)

    # Strip file extension, etc from first filename in list
    if filename == None:
        filename = fileList[0].split('/')[-1]
        filename = filename.rstrip('.tif')
        filename = filename.rstrip('0')
        filename = filename.rstrip('_')
        filename = filename +"_8bit"
        filename = filename.replace('.h5', '')

    if outputpath == None:
        outputpath = inputpath.rstrip('/') + "_8bit/"

    # create output directory if it does not exist
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    for iImage in range(len(fileList)):
        image32 = io.imread(fileList[iImage])
        image8 = convert_ArrayTo8bit(image32,data_min,data_max)
        outputfilepath = outputpath.rstrip('/')+'/'+filename + '_' + '{:04d}'.format(iImage) + '.tiff'
        io.imsave(outputfilepath,image8)
        print('complete: '+ outputfilepath.split('/')[-1])
        #print(iImage)

    print("conversion complete")

# -----------------------------------------------------------------------------

def convert_ArrayTo8bit(inputarray,data_min,data_max):
    rec = inputarray.astype(np.float32,copy=False)
    df = np.float32(data_max-data_min)
    mn = np.float32(data_min)
    scl = ne.evaluate('0.5+255*(rec-mn)/df',truediv=True)
    ne.evaluate('where(scl<0,0,scl)',out=scl)
    ne.evaluate('where(scl>255,255,scl)',out=scl)
    return scl.astype(np.uint8)

# -----------------------------------------------------------------------------

def crop_Directory(inputpath='./',xRange=(0,None),yRange=(0,None),zRange=(0,None),outputpath=None):

    fileList= get_fileList(inputpath)

    image = io.imread(fileList[0])

    # if no input give set range to maximum
    if xRange[1] == None:
        xRange[1] = len(image[:,0])
    if yRange[1] == None:
        yRange[1] = len(image[0,:])
    if zRange[1] == None:
        zRange[1] = len(fileList)

    # Strip file extension, etc from first filename in list
    if filename == None:
        filename = fileList[0].split('/')[-1]
        filename = filename.rstrip('.tif')
        filename = filename.rstrip('0')
        filename = filename.rstrip('_')
        filename = filename +"_cropped"
        filename = filename.replace('.h5','')

    # Autogenerate output path
    if outputpath == None:
        outputpath = inputpath.rstrip('/') + "_croppped/"

    # create output directory if it does not exist
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    # crop and save images
    for iImage in range(zRange[0],zRange[1]):
        image = io.imread(fileList[iImage])
        image_cropped = image[xRange[0]:xRange[1],yRange[0]:yRange[1]]
        outputfilepath = outputpath.rstrip('/')+'/'+filename + '_' + '{:04d}'.format(iImage) + '.tiff'
        io.imsave(outputfilepath,image_cropped)
        print(outputfilepath)
        #print(iImage)
# -----------------------------------------------------------------------------

def crop_Array(inputarray,xRange=(0,None),yRange=(0,None),zRange=(0,None)):

    # if no input give set range to maximum
    if xRange[1] == None:
        xRange[1] = len(image[:,0])
    if yRange[1] == None:
        yRange[1] = len(image[0,:])
    if inputarray.ndim==3 and zRange[1] == None:
        zRange[1] = len(image[0,0,:])

    if inputarray.ndim == 2:
        outputarray = image[xRange[0]:xRange[1],yRange[0]:yRange[1]]
    if inputarray.ndim == 3:
        outputarray = image[xRange[0]:xRange[1],yRange[0]:yRange[1],zRange[0]:zRange[1]]

# -----------------------------------------------------------------------------

def save_TiffStack(dataset,filename="image",outputPath="./"):
    filename = filename.rstrip(".tiff")
    filename = filename.rstrip(".tif")
    outputFile = outputPath + filename + ".tiff"
    print("saving ... : " + outputFile)
    skTiff.imsave(outputFile,dataset)
    print("save complete: " + outputFile)


# -----------------------------------------------------------------------------

def convert_DirectoryToTiffStack(filepath='./'):
    pass


# -----------------------------------------------------------------------------
def convert_ArrayToDirectory(dataset,filename="image",outputPath="./"):
    pass

