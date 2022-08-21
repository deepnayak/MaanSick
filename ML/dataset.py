from unicodedata import name
import sklearn
import random
from sklearn.model_selection import train_test_split
from load_data import *
from split import split
import numpy as np
import pickle

# def shuffle(data):
#     random.shuffle(data)

def fa_only(data):
    fa_data = []
    for i in data:
        fa_data.append([i[0][0],i[1]])
    return fa_data

def flatten(data):
    flattened_data = []
    for i in data:
        fa = np.array(i[0][0]).flatten()
        md = np.array(i[0][1]).flatten()
        rd = np.array(i[0][2]).flatten()
        ad = np.array(i[0][3]).flatten()
        flattened_data.append([[fa,md,rd,ad],i[1]])
    return flattened_data

def pd_only(data):
    pd_data = []
    for i in data:
        pd_data.append([i[0],i[1][0]])
    return pd_data

def depression_only(data):
    dep_thresh = 4
    depression_data = []
    for i in data:
        if i[1][1]>=dep_thresh:
            depression_data.append([i[0],1])
        else:
            depression_data.append([i[0],0])
    return depression_data

if __name__ == '__main__':
    # data = calculate_dtic()
    with open('dataDump_zoom_default.txt', 'rb') as dumpFile:
        data = pickle.load(dumpFile)
    flat_data = flatten(data)
    fa_data = fa_only(flat_data)
    pd_out_data = depression_only(fa_data)
    pd_out_data = np.array(pd_out_data, dtype=object)
    print(pd_out_data.shape)
    np.save("dep_out_data_zoom.npy",pd_out_data)
    split("dep_out_data_zoom.npy")