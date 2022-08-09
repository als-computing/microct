import sys
import os
import multiprocessing as mp
os.environ['NUMEXPR_MAX_THREADS'] = str(mp.cpu_count()) # to avoid numexpr warning
import numexpr
import numpy as np
import dxchange
# import pandas as pd
import base64
import pickle
import time

import ALS_recon_functions as als
import ALS_recon_helper as helper


use_gpu = als.check_for_gpu()

# def make_settings_dict(n, basepath, settings):
#     '''
#     Inputs:
#     name: string, filename
#     settings: one dictionary of base settings
    
#     Returns: 
#     d: dictionary with updated path and filename
#     '''
#     d = settings.copy()
#     d["path"] = basepath/n
#     d["name"] = n
#     return d


# def view_dictionaries(l):
#     '''
#     Input: 
#     l: list of prepared dictionaries
#     Returns 
#     df: a dataframe of dictionaries for easy viewing.
#     '''
#     df = pd.DataFrame(l)
#     return df.T

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


# def make_dict_with_settings_and_preprocess(settings, preprocess_settings):
#     '''
#     Input: 
#     dictionary: single dictionary of settings
#     Returns 
#     st: encoded dictionary
#     '''
#     both = {}
#     both["settings"] = copy.deepcopy(settings)
#     both["preprocess"] = copy.deepcopy(preprocess_settings)
#     return both

def batch_astra_recon(settings):
#     d = pickle.loads(base64.b64decode(string.encode('utf-8')))
#     settings = d["settings"]
#     preprocess_settings = d["preprocess"]
#     tomo, angles = als.read_data(settings["path"],
#                              preprocess_settings=preprocess_settings,
#                              proj=settings["angles_ind"],
#                              sino=settings["slices_ind"],
#                              downsample_factor=settings["downsample_factor"])
 
#     recon = als.astra_fbp_recon(tomo, angles, **settings)
#     fname = "reconstruction_" + settings["name"]
#     dxchange.write_tiff_stack(recon, fname = fname)    

    nchunk = 50 
    '''
    nchunk is balance between available cpus and memory (larger value can be more parallelized but uses more memory)
    50 was empirically chosen on Perlmutter exclusive node, though 100 was more or less the same
    ''' 
    save_dir = os.path.join(settings["data"]["output_path"],settings["data"]["name"])
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    save_name = os.path.join(save_dir,"img")
    for i in range(np.ceil((settings["data"]['stop_slice']-settings["data"]['start_slice'])/nchunk).astype(int)):
        start_iter = settings["data"]['start_slice']+i*nchunk
        stop_iter = np.minimum(start_iter+nchunk,settings["data"]['stop_slice'])
        print(f"Starting recon of slices {start_iter}-{stop_iter}...",end=' ')
        tic = time.time()

        recon = helper.default_reconstruction(path=settings["data"]["data_path"],
                               angles_ind=settings["data"]['angles_ind'],
                               slices_ind=slice(start_iter,stop_iter,1),
                               proj_downsample=settings["data"]["proj_downsample"],
                               COR=settings["recon"]["COR"],
                               fc=settings["recon"]["fc"],
                               preprocessing_args=settings["preprocess"],
                               postprocessing_args=settings["postprocess"],
                               use_gpu=settings["recon"]["use_gpu"])

        print(f"Finished: took {time.time()-tic} sec. Saving files...")
        dxchange.write_tiff_stack(recon, fname=save_name, start=start_iter)
    print("Done")
    
def mpi4py_svmbir_recon(settings):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()

    for i in range((settings["data"]['stop_slice']-settings["data"]['start_slice'])):
        if i % size == rank:
            print(f"Starting slice {i} on {name}, core {rank} of {size}")
            tic = time.time()
            tomo, angles = als.read_data(settings["data"]["data_path"],
                                         proj=settings["data"]["angles_ind"],
                                         sino=slice(i,i+1,1),
                                         downsample_factor=settings["data"]["proj_downsample"],
                                         args=settings["preprocess"])
            svmbir_recon = als.svmbir_recon(tomo,angles,**svmbir_settings)
            print(f"Finished slice {i}, took {time.time()-tic} sec")


def main():
    string = sys.argv[:][-1] 
    settings = pickle.loads(base64.b64decode(string.encode('utf-8')))
    if 'svmbir_settings' in settings:
        mpi4py_svmbir_recon(settings)
    else:
        batch_astra_recon(settings)
    
if __name__ == '__main__':
    main()