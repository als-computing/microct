from __future__ import print_function
import numpy as np
import os
os.environ["NUMEXPR_MAX_THREADS"] = '999'
import numbers
import dxchange
import pickle
import json
import base64
import requests
from authlib.integrations.requests_client import OAuth2Session
from authlib.oauth2.rfc7523 import PrivateKeyJWT
from reconstructionGPU import recon_setup, recon

def string_prep(dictionary, jwt): 
    '''
    Inputs: 
    dictionary: 
    '''
    pik = pickle.dumps(dictionary, protocol=pickle.HIGHEST_PROTOCOL)
    st = base64.b64encode(pik).decode('utf-8')
    return st

def get_jwt(filename):
    token_url = "https://oidc.nersc.gov/c2id/token"
    
    with open(filename) as f: 
        keys = json.load(f)
    client_id = keys["client"]
    private_key = keys["private"]
    public_key = keys["public"]
    
    client = OAuth2Session(
        client_id, 
        private_key, 
        PrivateKeyJWT(token_url),
        grant_type="client_credentials",
        token_endpoint=token_url
    )

    client.register_client_auth_method(PrivateKeyJWT(token_url))
    client.fetch_token()
    resp = client.fetch_token(token_url, grant_type="client_credentials")
    jwt = resp["access_token"]
    print(jwt)
    return jwt
    
    
def submit(dictionary, jwt):
    st = string_prep(dictionary, jwt)
    body = {"data": st, "jwt": jwt}
    endpoint = f"http://test.lgupta.development.svc.spin.nersc.org/inputs/"
    r = requests.post(endpoint, json = body)
    print(r)
    return r.json()


def get_task_info(task_id, jwt):
    ## This doesn't work yet :(
    body = {"task_id": str(task_id), "jwt": jwt}
    endpoint = f"http://test.lgupta.development.svc.spin.nersc.org/tasks/"
    r = requests.get(endpoint, json = body)
    return r.json()
