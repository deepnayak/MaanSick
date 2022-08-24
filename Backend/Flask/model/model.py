import numpy as np 
import cv2
import os
import pickle
import pandas as pd
from ReliefF import ReliefF
import scipy.interpolate as interp
from scipy.ndimage import zoom

import numpy as np
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

from .dtiprocess import dti_process

# Importing model and initializing visualization model
model = tf.keras.models.load_model("cnnBrainDepressionBest")
successive_outputs = [layer.output for layer in model.layers]
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)

class CNN:
    
    def dti_process(nii_file,bval_file,bvec_file):
        data, affine = load_nifti(nii_file)
        bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
        gtab = gradient_table(bvals, bvecs)

        # First of all, we mask and crop the data. This is a quick way to avoid calculating Tensors on the background of the image. This is done using DIPY’s mask module.
        maskdata, mask = median_otsu(data, vol_idx=range(10, 50), median_radius=3, numpass=1, autocrop=True, dilate=2)

        # Now that we have prepared the datasets we can go forward with the voxel reconstruction. First, we instantiate the Tensor model in the following way.
        tenmodel = dti.TensorModel(gtab)
        # print(gtab)

        # Fitting the data is very simple. We just need to call the fit method of the TensorModel in the following way:
        tenfit = tenmodel.fit(maskdata, mask=mask)

        # qf is the tensor that contains the diffusion matrix for each voxel in the 3D space
        qf = tenfit.quadratic_form
        # print(qf.shape)

        # Eigen Values and Vectors
        eigvals, eigvecs = dti.decompose_tensor(qf)

        # Diffusion metrics
        fa = tenfit.fa
        md = tenfit.md
        rd = tenfit.rd
        ad = tenfit.ad
        # print(fa.shape,md.shape,rd.shape,ad.shape)
        fa_img = nib.Nifti1Image(fa.astype(np.float32), affine)
        nib.save(fa_img, 'tensor_fa.nii.gz')
        return fa,md,rd,ad

    def processNiiFile(niiFile, bvecFile, bvalFile):
        fa, md, rd, ad = [zoom(x, (50/x.shape[0], 50/x.shape[1], 50/x.shape[2])) for x in dti_process(niiFile, bvecFile, bvalFile)]

        inputMat = np.moveaxis(np.array([fa]), 0, -1)

        successive_feature_maps = visualization_model.predict(np.array([inputMat]))

        for i in [0, 2, 4, 7]:
            for patient in successive_feature_maps[i]:
                newImages = np.moveaxis(patient, -1, 0)
                data, affine = load_nifti(niiFile)
                for x in range(4):
                    if x == 3:
                        fa_img = nib.Nifti1Image(newImages[x].astype(np.float32), affine)
                        nib.save(fa_img, f'tensor_fa50_{i}_{x}.nii.gz')
                        twoDimage = nib.load(f"tensor_fa50_{i}_{x}.nii.gz").get_fdata()
                        # twoDimage = np.array(fa_img.slicer[0:1])
                        twoDimage = zoom(twoDimage[:, 30, :], (20, 20))
                        # twoDimage = np.moveaxis(twoDimage, -1, 0)
                        print(twoDimage.shape)
                        
                        rescaled = (255.0 / twoDimage.max() * (twoDimage - twoDimage.min())).astype(np.uint8)

                        im = Image.fromarray(rescaled)
                        im.save(f"tensor_fa50_{i}_{x}.png")
                        
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
        return self.model.predict(newProcessedData)