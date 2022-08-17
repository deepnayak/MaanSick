from unicodedata import name
import sklearn
import random
from sklearn.model_selection import train_test_split
from load_data import *
import numpy as np

# def shuffle(data):
#     random.shuffle(data)

def split(data):
    random.shuffle(data)
    x = [i[0] for i in data]
    y = [i[1] for i in data]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)
    print(len(x_train),len(x_test),len(x_val))

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
        if i[1][0]>dep_thresh:
            depression_data.append([i[0],1])
        else:
            depression_data.append([i[0],0])
    return depression_data

if __name__ == '__main__':
    data = calculate_dtic()
    flat_data = flatten(data)
    fa_data = fa_only(flat_data)
    pd_out_data = pd_only(fa_data)
    split(pd_out_data)
    np.save("pd_out_data.npy",np.array(pd_out_data, dtype=object))