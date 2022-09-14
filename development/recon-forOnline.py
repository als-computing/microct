#!/usr/bin/env python
# coding: utf-8
import os
import dxchange as dx
import tomopy
import numpy as np
import sys
import json
import pickle
import base64

###
# *Having issues with passing the sino in a good way. Can't pass tuple
# and passing it in some other form and then making it into a tuple was giving me issues.
# *Saving the recon(s)?
# *This initial version kind of assumes there is just a single recon.
###

def load_data(settings):
    filepath = settings['filepath']
    prj, flat, dark, ang = dx.read_aps_32id(str(filepath), sino = (0,1))
    return prj, ang

def run_recon(prj, ang, recon_settings):
    recon = tomopy.recon(tomo=prj, theta=ang, **recon_settings)
    return recon       

def save_recon(recon, settings):
    filepath = settings['filepath']
    savepath = settings['savepath']
    name = settings["name"]
    filename = os.path.join(savepath, "reconstruction_settings.json")
    with open(filename, 'w') as fp:
        json.dump(settings, fp)
    ###
    filename = os.path.join(savepath, name+"out.plk")
    with open(filename, 'wb') as handle:
        pickle.dump(recon, handle, protocol=pickle.HIGHEST_PROTOCOL)
    ###
    
def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    string = sys.argv[:][-1] 
    #settings = pickle.loads(base64.b64decode(string.encode('utf-8')))
    settings = json.loads(string)
    print(settings)

    ## break into two dicts for the loading and the recon
    data_settings = {'filepath':settings['filepath'], 'savepath':settings['savepath'], 'name':settings['name']}
    #print(data_settings)
    key_list = ['filepath', 'savepath', 'name']
    for key in key_list:
        settings.pop(key, None)

    recon_settings = settings
    #debug
    print(recon_settings)
    
    prj, ang = load_data(data_settings)
    print("Loaded data", flush = True)
    recon = run_recon(prj, ang, recon_settings)
    print("Recon done",flush = True)
    save_recon(recon, data_settings)
    print("Recon saved",flush = True)


if __name__ == '__main__':
    main()
    
