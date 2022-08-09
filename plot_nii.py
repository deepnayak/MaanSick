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

# To open from folder
# fraw, fbval, fbvec, t1_fname = get_fnames('cfin_multib')
# data, affine = load_nifti(fraw)
# bvals, bvecs = read_bvals_bvecs(fbval, fbvec)

nii_file = "run1\\sub-01 dwi sub-01_run-1_dwi.nii.gz"
bval_file = "run1\\sub-01 dwi sub-01_run-1_dwi.bval"
bvec_file = "run1\\sub-01 dwi sub-01_run-1_dwi.bvec"

# Display/Plot nii files
data, affine, img = load_nifti(nii_file, return_img=True)
print(data.shape)
print(img.header.get_zooms()[:3])

axial_middle = data.shape[2] // 2
plt.figure('Showing the datasets')
plt.subplot(1, 2, 1).set_axis_off()
plt.imshow(data[:, :, axial_middle].T, cmap='gray', origin='lower')
plt.subplot(1, 2, 2).set_axis_off()
plt.imshow(data[:, :, axial_middle].T, cmap='gray', origin='lower')
plt.show()

