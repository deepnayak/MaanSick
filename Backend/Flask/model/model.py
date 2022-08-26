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
model = tf.keras.models.load_model("./model/cnnBrainDepressionBest/")
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
            0: {"quad": "A","functions": ["Logical", "Analytical", "Fact Based", "Quantitative"],"diseases": ["Black and white thinking"],"suggestions": ["Try to separate what you do from who you are: When we equate our performance on a single metric with our overall worth, we’re going to become vulnerable to black and white thinking.", "Try listing options: If black and white thinking has you locked into only two outcomes or possibilities, as an exercise, write down as many other options as you can imagine. If you’re having trouble getting started, try coming up with three alternatives at first.", "Practice reality reminders: When you feel paralyzed by black and white thinking, say or write small factual statements, like There are several ways I can solve this problem, I’ll make a better decision if I take time to get more information, and Both of us may be partially right.", "Find out what other people think: Black and white thinking can keep you from seeing things from someone else’s perspective. When you’re in conflict with someone, calmly ask clarifying questions so you can come to a clear understanding of their viewpoint."],"lobes": ["Frontal Lobe", "Temporal Lobe"]},
            
            1: {"quad": "B","functions": ["Sequential", "Organized", "Detailed", "Planned"],"diseases": ["OCD","Anxiety Disorder"],"suggestions": ["Stay active. Participate in activities that you enjoy and that make you feel good about yourself.", "Enjoy social interaction and caring relationships, which can lessen your worries.", "Avoid alcohol or drug use. Alcohol and drug use can cause or worsen anxiety","Cognitive-behavioral therapy is a type of psychotherapy", "Medications: Drugs called serotonin reuptake inhibitors (SRIs), selective SRIs (SSRIs) and tricyclic antidepressants may help", "Exposure and response prevention (EX/RP)  therapy"],"lobes": ["Frontal Lobe"]},


            2: {"quad": "C","functions": ["Interpersonal", "Feeling Based", "Kinesthetic", "Emotional"],"diseases": ["Emotional lability","Eating disorders"],"suggestions": ["Take frequent breaks from social situations to calm yourself.", "Look for a local support group or online community to meet other people dealing with the condition that caused your emotional lability.", "Practice slow breathing techniques and focus on your breath during episodes.", "Figure out what triggers your episodes, such as stress or fatigue. ", "Distract yourself from rising emotions with a change of activity or position.","Avoid alcohol: May be harmful and aggravate certain conditions.", "Reduce caffeine intake: Reduces risk of aggravating certain conditions.", "Physical exercise: Aerobic activity for 20–30 minutes 5 days a week improves cardiovascular health. ", "Quitting smoking", "Relaxation techniques: Deep breathing, meditation, yoga, rhythmic exercise and other activities that reduce symptoms of stress", "Stress management: Pursuing an enjoyable activity or verbalising frustration to reduce stress and improve mental health.", "Healthy diet: A diet that provides essential nutrients and adequate calories, while avoiding excess sugar, carbohydrates and fatty foods."],"lobes": ["Temporal Lobe"]},

            3: {"quad": "D","functions": ["Holistic", "Intuitive", "Integrating", "Synthesising"],"diseases": ["attention deficit hyperactivity disorder (ADHD)","Bipolar Disorder"], "suggestions": ["Staying organized: set appointment reminders, highlight important days on the calendar, mark deadlines, and keep lists and other information handy. Set aside time each day to update your lists and schedules. Don’t let the task become a chore in itself; think of it like a routine task such as brushing your teeth, and do it daily so it becomes an established habit.", "Staying focused: Reduce distractions, jot down ideas as they come to you, meet deadlines"], "lobes": ["Prefrontal cortex (Frontal Lobe)","Temporal Lobe"]}
        }

        dep_quad = q_outs.index(max(q_outs))
        output = quad_dict[dep_quad]

        return output

    def processNiiFile(self, parameterList, niiFile):
        print("HiHiHaha ", niiFile)
        fa, md, rd, ad = [zoom(x, (50/x.shape[0], 50/x.shape[1], 50/x.shape[2])) for x in parameterList]

        inputMat = np.moveaxis(np.array([fa]), 0, -1)

        successive_feature_maps = visualization_model.predict(np.array([inputMat]))

        for i in [0, 2, 4, 7]:
            for patient in successive_feature_maps[i]:
                newImages = np.moveaxis(patient, -1, 0)
                data, affine = load_nifti("./model/"+niiFile)
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
        self.filename = './model/svm_model_dep_zoom_cross_val.sav'
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
        t1 = threading.Thread(target=CNN.processNiiFile, args=(CNN(),[fa, md, rd, ad], nii_file))
        t1.start()
        return self.model.predict_proba(newProcessedData)