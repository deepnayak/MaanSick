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
# nii_file = "p06316_bmatrix_1000.nii.gz"
# bval_file = "p06316_bval_1000"
# bvec_file = "p06316_grad_1000"

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
    # fa_img = nib.Nifti1Image(fa.astype(np.float32), affine)
    # nib.save(fa_img, 'tensor_fa.nii.gz')
    return fa,md,rd,ad

# for i in range(len(eigvals)):
#     for j in range(len(eigvals[i])):
#         for k in range(len(eigvals[i][j])):
#             egv = eigvals[i][j][k]
#             if np.any(egv):
#                 print(fa[i][j][k],md[i][j][k],rd[i][j][k],ad[i][j][k])
# dti_process("park4\\NITRC_PD_DATA\\p06316\\p06316_bmatrix_1000.nii.gz","park4\\NITRC_PD_DATA\\p06316\\p06316_bval_1000","park4\\NITRC_PD_DATA\\p06316\\p06316_grad_1000")

