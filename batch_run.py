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

def make_settings_dict(n, basepath, settings):
    '''
    Inputs:
    name: string, filename
    settings: one dictionary of base settings
    
    Returns: 
    d: dictionary with updated path and filename
    '''
    d = settings.copy()
    d["path"] = basepath/n
    d["name"] = n
    return d


def view_dictionaries(l):
    '''
    Input: 
    l: list of prepared dictionaries
    Returns 
    df: a dataframe of dictionaries for easy viewing.
    '''
    df = pd.DataFrame(l)
    return df.T

def dictionary_prep(dictionary):
    '''
    Input: 
    dictionary: single dictionary of settings
    Returns 
    st: encoded dictionary
    '''
    pik = pickle.dumps(dictionary, protocol=pickle.HIGHEST_PROTOCOL)
    st = base64.b64encode(pik).decode('utf-8')
    return st


def make_dict_with_settings_and_preprocess(settings, preprocess_settings):
    '''
    Input: 
    dictionary: single dictionary of settings
    Returns 
    st: encoded dictionary
    '''
    both = {}
    both["settings"] = copy.deepcopy(settings)
    both["preprocess"] = copy.deepcopy(preprocess_settings)
    return both

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
    fname = "reconstruction_" + settings["name"]
    dxchange.write_tiff_stack(recon, fname = fname)    
    
    
if __name__ == '__main__':
    main()