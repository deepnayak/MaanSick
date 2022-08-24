import numpy as np 
import cv2
import os
import pickle
import pandas as pd
from ReliefF import ReliefF
import scipy.interpolate as interp
from scipy.ndimage import zoom

import numpy as np
import matplotlib.pyplot as plt
import dipy.reconst.dki as dki
import dipy.reconst.dti as dti
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.segment.mask import median_otsu
from scipy.misc import face
from scipy.ndimage.filters import gaussian_filter
import nibabel as nib
from scipy.ndimage import zoom
import tensorflow as tf
from PIL import Image, ImageOps
import nibabel as nib
import matplotlib.pyplot as plt
import threading

from .dtiprocess import dti_process

# Importing model and initializing visualization model
model = tf.keras.models.load_model("./model/cnnBrainDepressionBest")
successive_outputs = [layer.output for layer in model.layers]
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)

class CNN:
    def depression_quadrant(self, fileName):
        q_outs = [0,0,0,0]
        files = [f"./static/tempNii/{fileName}_7_{x}.nii.gz" for x in range(4)]
        for nii_file in files:
            data, affine = load_nifti(nii_file)
            quads = data[:25,:25], data[:25,25:], data[25:,25:], data[25:,:25]
            qsums = [quad.sum() for quad in quads]
            q_outs[qsums.index(max(qsums))] +=1

        quad_dict = {
            0: {"quad": "A","suggestions": ["Logical", "Analytical", "Fact Based", "Quantitative"],"diseases": [],"suggestions": [],"lobes": []},
            1: {"quad": "B","functions": ["Sequential", "Organized", "Detailed", "Planned"],"diseases": [],"suggestions": [],"lobes": []},
            2: {"quad": "C","suggestions": ["Holistic", "Intuitive", "Integrating", "Synthesising"],"diseases": [], "suggestions": [], "lobes": []},
            3: {"quad": "D","suggestions": ["Interpersonal", "Feeling Based", "Kinesthetic", "Emotional"],"diseases": [],"suggestions": [],"lobes": []}
        }

        dep_quad = q_outs.index(max(q_outs))
        output = quad_dict[dep_quad]

        return output

    def processNiiFile(self, parameterList, niiFile):
        fa, md, rd, ad = [zoom(x, (50/x.shape[0], 50/x.shape[1], 50/x.shape[2])) for x in parameterList]

        inputMat = np.moveaxis(np.array([fa]), 0, -1)

        successive_feature_maps = visualization_model.predict(np.array([inputMat]))

        for i in [0, 2, 4, 7]:
            for patient in successive_feature_maps[i]:
                newImages = np.moveaxis(patient, -1, 0)
                data, affine = load_nifti(niiFile)
                for x in range(4):
                    if i == 7:
                        fa_img = nib.Nifti1Image(newImages[x].astype(np.float32), affine)
                        nib.save(fa_img, f'./static/tempNii/{niiFile}_{i}_{x}.nii.gz')
                    if x == 3:
                        fa_img = nib.Nifti1Image(newImages[x].astype(np.float32), affine)
                        nib.save(fa_img, f'./static/tempNii/{niiFile}_{i}_{x}.nii.gz')
                        twoDimage = nib.load(f"./static/tempNii/{niiFile}_{i}_{x}.nii.gz").get_fdata()
                        # twoDimage = np.array(fa_img.slicer[0:1])
                        twoDimage = zoom(twoDimage[:, 30, :], (20, 20))
                        # twoDimage = np.moveaxis(twoDimage, -1, 0)
                        print(twoDimage.shape)
                        
                        rescaled = (255.0 / twoDimage.max() * (twoDimage - twoDimage.min())).astype(np.uint8)

                        im = Image.fromarray(rescaled)
                        im.save(f"./static/images/{niiFile}_{i}_{x}.png")
                        
                        # plt.imshow(twoDimage[0])
                        # plt.show()
                break



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
        t1 = threading.Thread(target=CNN.processNiiFile, args=([fa, md, rd, ad], nii_file))
        t1.start()
        return self.model.predict(newProcessedData)