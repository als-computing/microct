"""
ALS_batch_recon.py
Functions that used to either set up reconstruction batch jobs, or used within the batch jobs  
"""

import sys
import os
import multiprocessing as mp
os.environ['NUMEXPR_MAX_THREADS'] = str(mp.cpu_count()) # to avoid numexpr warning
import numexpr
import numpy as np
import dxchange
import base64
import pickle
import time
import datetime
import re
from pathlib import Path

import ALS_recon_functions as als
import ALS_recon_helper as helper

MAX_JOB_SECONDS = 300*60 # 5 hour 00 min

def get_batch_template(algorithm="astra"):
    """ Gets path to appropriate batch scrpit template, depending on whether using Astra or SVMBIR, on Cori or Perlmutter """
    
    s = os.popen("echo $NERSC_HOST")
    out = s.read()
    if algorithm == "svmbir":
        if 'cori' in out:
            return os.path.join('slurm_scripts','svmbir_template_job-cori.txt')
        elif 'perlmutter' in out:
            return os.path.join('slurm_scripts','svmbir_template_job-perlmutter.txt')
        else:
            sys.exit('not on cori or perlmutter for svmbir job -- throwing error')
    if 'cori' in out:
        return os.path.join('slurm_scripts','astra_template_job-cori.txt')
    elif 'perlmutter' in out:
        return os.path.join('slurm_scripts','astra_template_job-perlmutter.txt')
    else:
        sys.exit('not on cori or perlmutter  for astra job -- throwing error')

def create_batch_script(settings):
    """ Completes batch script from template by adding reconstruction settings """
    
    with open (get_batch_template(), "r") as t:
        template = t.read()

    s = os.popen("echo $USER")
    username = s.read()[:-1]
    user_template = template.replace('<username>',username)

    s = os.popen("echo $NERSC_HOST")
    out = s.read()

    # calculate job time by number of slices (on either perlmutter or cori)
    sec_per_100_slices = 45 if 'perlmutter' in out else 90 # may need to adjust a little
    num_slices = settings["data"]["stop_slice"] - settings["data"]["start_slice"]
    total_seconds = int(np.minimum(np.ceil(num_slices/100)*sec_per_100_slices,MAX_JOB_SECONDS))
    seconds = total_seconds % 60
    minutes = (total_seconds // 60) % 60
    hours = (total_seconds // 60) // 60
    user_template = user_template.replace("--time=00:15:00",f"--time={hours:02d}:{minutes:02d}:{seconds:02d}")
        
    configs_dir = Path(os.path.join(settings["data"]["output_path"],"configs/"))
    if not configs_dir.exists():
        os.mkdir(configs_dir)
       
    config_script_name = os.path.join(configs_dir,"config_"+settings["data"]["name"]+".sh")    
    enc = dictionary_prep(settings)
    with open(config_script_name, 'w') as f:
        script = user_template
        script += "\n"
        script += f"shifter python {os.getcwd()}/backend/ALS_batch_recon.py"
        script += " '" + enc + "'"
        f.write(script)
        f.close()
    
    return configs_dir, config_script_name

def create_svmbir_batch_script(settings):
    """ Completes svmbir script from template by adding reconstruction settings """
    with open (get_batch_template(algorithm="svmbir"), "r") as t:
        template = t.read()

    # number of nodes and jobs
    N = int(re.search('#SBATCH -N ([0-9]+)',template)[1])
    n = int(re.search('#SBATCH -n ([0-9]+)',template)[1])

    s = os.popen("echo $USER")
    username = s.read()[:-1]
    user_template = template.replace('<username>',username)        
        
    s = os.popen("echo $NERSC_HOST")
    out = s.read()

    # calculate job time by number of slices (on either perlmutter or cori)
    sec_per_slice = 20*60 # Found 20 min was about right for 8 slices. Can increase if jobs aren't finishing 
    num_slices = settings["data"]["stop_slice"] - settings["data"]["start_slice"]
    total_seconds = int(np.minimum(np.ceil(num_slices/n)*sec_per_slice,MAX_JOB_SECONDS))
    seconds = total_seconds % 60
    minutes = (total_seconds // 60) % 60
    hours = (total_seconds // 60) // 60
    user_template = user_template.replace("--time=00:15:00",f"--time={hours:02d}:{minutes:02d}:{seconds:02d}")

    configs_dir = Path(os.path.join(settings["data"]["output_path"],"configs/"))
    if not configs_dir.exists():
        os.mkdir(configs_dir)

    config_script_name = os.path.join(configs_dir,"svmbir-config_"+settings["data"]["name"]+".sh")    
    enc = dictionary_prep(settings)
    with open(config_script_name, 'w') as f:
        script = user_template
        script += "\n"
        script += f"srun -N {N} -n {n} shifter python {os.getcwd()}/backend/ALS_batch_recon.py"
        script += " '" + enc + "'"
        f.write(script)
        
    return configs_dir, config_script_name


def dictionary_prep(dictionary):
    ''' Encodes reconstruction parameter dictionary into string 
    Input: 
    dictionary: single dictionary of settings
    Returns 
    st: encoded dictionary
    '''
    pik = pickle.dumps(dictionary, protocol=pickle.HIGHEST_PROTOCOL)
    st = base64.b64encode(pik).decode('utf-8')
    return st


def batch_astra_recon(settings): 
    """ Perform Astra reconstruction using encoded settings string """

    print(f"Starting ALS batch Astra recon...")
    
    use_gpu = als.check_for_gpu()
    nchunk = 50 
    '''
    nchunk is balance between available cpus and memory (larger value can be more parallelized but uses more memory)
    50 was empirically chosen on Perlmutter exclusive node, though 100 was more or less the same
    ''' 
    save_dir = os.path.join(settings["data"]["output_path"],settings["data"]["name"])
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    save_name = os.path.join(save_dir,settings["data"]["name"])
    
    # if COR is None, use cross-correlation finder
    if settings["recon"]["COR"] is None:
        settings["recon"]["COR"] = als.auto_find_cor(settings["data"]["data_path"])

    tic0 = time.time()
    for i in range(np.ceil((settings["data"]['stop_slice']-settings["data"]['start_slice'])/nchunk).astype(int)):
        start_iter = settings["data"]['start_slice']+i*nchunk
        stop_iter = np.minimum(start_iter+nchunk,settings["data"]['stop_slice']+1)
        print(f"Starting recon of slices {start_iter}-{stop_iter}...",end=' ')
        tic = time.time()

        recon, _ = helper.reconstruct(path=settings["data"]["data_path"],
                                      angles_ind=settings["data"]['angles_ind'],
                                      slices_ind=slice(start_iter,stop_iter,1),
                                      COR=settings["recon"]["COR"],
                                      method=settings["recon"]["method"],
                                      proj_downsample=settings["data"]["proj_downsample"],
                                      fc=settings["recon"]["fc"],
                                      preprocessing_settings=settings["preprocess"],
                                      postprocessing_settings=settings["postprocess"],
                                      use_gpu=use_gpu)

        print(f"Finished: took {time.time()-tic} sec. Saving files...")
        dxchange.write_tiff_stack(recon, fname=save_name, start=start_iter)
    print(f"Done, took {time.time()-tic0} sec")
    
def mpi4py_svmbir_recon(settings):
    """ Perform SVMBIR reconstruction using encoded settings string. Parallelize over slices using mpi4py """

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    
    save_dir = os.path.join(settings["data"]["output_path"],settings["data"]["name"]+"-svmbir")
    if rank == 0: # to avoid multiple tasks doing this at the same time
        if not os.path.exists(save_dir): os.makedirs(save_dir)
                        
    # if COR is None, use cross-correlation finder
    if settings["svmbir_settings"]["COR"] is None:
        settings["svmbir_settings"]["COR"] = als.auto_find_cor(settings["data"]["data_path"])    

    SLICES_PER_CHUNK = 8 # hardcoded parameter -- didn't see improvement at 16, but didn't test much        
    NUM_CHUNKS = int(np.ceil((settings["data"]['stop_slice']-settings["data"]['start_slice'])/SLICES_PER_CHUNK))
    
    print(f"SLICES_PER_CHUNK: {SLICES_PER_CHUNK},    NUM_CHUNKS: {NUM_CHUNKS}")
    
    for i in range(NUM_CHUNKS):
        if i % size == rank:
            start_slice = settings["data"]['start_slice'] + i*SLICES_PER_CHUNK
            end_slice = np.minimum((i+1)*SLICES_PER_CHUNK,settings["data"]['stop_slice']+1)
            print(f"Starting SVMBIR recon of slices {start_slice} to {end_slice-1} on {name}, core {rank} of {size}")
            save_name = os.path.join(save_dir,settings["data"]["name"])
            tic = time.time()
            
            tomo, angles = als.read_data(settings["data"]["data_path"],
                                         proj=settings["data"]["angles_ind"],
                                         sino=slice(start_slice,end_slice),
                                         downsample_factor=settings["data"]["proj_downsample"],
                                         preprocess_settings=settings["preprocess"],
                                         postprocess_settings=settings["postprocess"])
            
            svmbir_recon = als.svmbir_recon(tomo,angles,**settings["svmbir_settings"])
            svmbir_recon = als.mask_recon(svmbir_recon)
            print(f"Finished slice {start_slice} to {end_slice} on {name}, core {rank} of {size}, took {time.time()-tic} sec")
            dxchange.write_tiff_stack(svmbir_recon, fname=save_name, start=start_slice)

def main():
    string = sys.argv[:][-1] 
    settings = pickle.loads(base64.b64decode(string.encode('utf-8')))
    if settings["recon"]["method"] == "svmbir":
        mpi4py_svmbir_recon(settings)
    else:
        batch_astra_recon(settings)
       

    
if __name__ == '__main__':
    main()