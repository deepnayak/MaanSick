import numpy as np 
import cv2
import os
import pickle
import pandas as pd
from ReliefF import ReliefF
import scipy.interpolate as interp
from scipy.ndimage import zoom

from .dtiprocess import dti_process


class DepPredict:
    def __init__(self):
        self.filename = './model/svm_model_dep_zoom.sav'
        self.model = pickle.load(open(self.filename, 'rb'))
        self.ref_model_size = 235415
        self.minimumDimension = 100000
    
    def interpolate1D(self, data):
        print(data.shape)
        arr_interp = interp.interp1d(np.arange(data.size),data)
        if data.size>self.ref_model_size:
            final_arr = arr_interp(np.linspace(0,data.size-1,self.ref_model_size))
        else:
            final_arr = arr_interp(np.linspace(0,data.size-1,self.ref_model_size))
        return final_arr

    def zoom_input(self, data):
        minimumDimension = min(min(data.shape), self.minimumDimension)
        return zoom(data, (50/data.shape[0], 50/data.shape[1], 50/data.shape[2]))

    def processData(self,data):
        data = self.zoom_input(data)
        data = data.flatten()
        df = pd.DataFrame([data])
        return df
    
    def prediction(self,nii_file, bval, bvec):
        print("Enetered the class prediction function", nii_file, bval, bvec)
        fa, md, rd, ad = dti_process("./model/" + nii_file, "./model/" + bval, "./model/" + bvec)
        # fa, md, rd, ad = dti_process("./model/p07677_bmatrix_1000.nii.gz", "./model/p07677_bval_1000", "./model/p07677_grad_1000")

        newProcessedData = self.processData(np.array(fa))
        return self.model.predict(newProcessedData)