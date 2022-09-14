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
import reconstructionGPU2 as reconGPU


    
def main():
    string = sys.argv[:][-1] 
    settings = pickle.loads(base64.b64decode(string.encode('utf-8')))

    
    recon = reconGPU.recon(**settings)
    print("Finished")


if __name__ == '__main__':
    main()
    
