import nibabel as nib
import pickle
from dipy.io.image import load_nifti
import numpy as np
with open('dataDump_zoom_default.txt', 'rb') as dumpFile:
        data = pickle.load(dumpFile)

fa = data[0][0][0]
data, affine = load_nifti("park4\\NITRC_PD_DATA\\p06316\\p06316_bmatrix_1000.nii.gz")
fa_img = nib.Nifti1Image(fa.astype(np.float32), affine)
nib.save(fa_img, 'tensor_fa50.nii.gz')