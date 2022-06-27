import sys
import os
import multiprocessing as mp
os.environ['NUMEXPR_MAX_THREADS'] = str(mp.cpu_count()) # to avoid numexpr warning
import numexpr
import numpy as np
from skimage import transform, filters, io
import tomopy
import dxchange
import pandas as pd
import base64
import pickle
from pathlib import Path

import util
import ALS_recon_helper as als


use_gpu = als.check_for_gpu()


def main():
    string = sys.argv[:][-1] 
    d = pickle.loads(base64.b64decode(string.encode('utf-8')))
    settings = d["settings"]
    preprocess_settings = d["preprocess"]
    tomo, angles = als.read_data(settings["path"],
                             preprocess_settings=preprocess_settings,
                             proj=settings["angles_ind"],
                             sino=settings["slices_ind"],
                             downsample_factor=settings["downsample_factor"])
 
    recon = als.astra_fbp_recon(tomo, angles, **settings)
    fname = settings["name"]
    dxchange.write_tiff_stack(recon, fname = fname)    
    
    
if __name__ == '__main__':
    main()