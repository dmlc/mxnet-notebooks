import numpy as np
import os
from mult import read_mult


def downloadData():
    data_url = 'https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/cdl'
    for filename in ('mult.dat', 'cf-train-1-users.dat', 'cf-test-1-users.dat', 'raw-data.csv'):
        if not os.path.exists(filename):
            os.system("wget %s/%s" % (data_url, filename))



def get_mult():
    if not os.path.exists('mult.dat'):
        downloadData()
    X = read_mult('mult.dat',8000).astype(np.float32)
    return X

def get_dummy_mult():
    X = np.random.rand(100,100)
    X[X<0.9] = 0
    return X

def read_user(f_in='cf-train-1-users.dat',num_u=5551,num_v=16980):
    fp = open(f_in)
    R = np.mat(np.zeros((num_u,num_v)))
    for i,line in enumerate(fp):
        segs = line.strip().split(' ')[1:]
        for seg in segs:
            R[i,int(seg)] = 1
    return R

def read_dummy_user():
    R = np.mat(np.random.rand(100,100))
    R[R<0.9] = 0
    R[R>0.8] = 1
    return R

