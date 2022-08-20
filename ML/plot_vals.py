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

nii_file = "run1\\sub-01 dwi sub-01_run-1_dwi.nii.gz"
bval_file = "run1\\sub-01 dwi sub-01_run-1_dwi.bval"
bvec_file = "run1\\sub-01 dwi sub-01_run-1_dwi.bvec"

data, affine = load_nifti(nii_file)
bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
gtab = gradient_table(bvals, bvecs)

maskdata, mask = median_otsu(data, vol_idx=[0, 1], median_radius=4, numpass=2,autocrop=False, dilate=1)

fwhm = 1.25
gauss_std = fwhm / np.sqrt(8 * np.log(2))  # converting fwhm to Gaussian std
data_smooth = np.zeros(data.shape)
for v in range(data.shape[-1]):
    data_smooth[..., v] = gaussian_filter(data[..., v], sigma=gauss_std)
                  
tenmodel = dti.TensorModel(gtab)
tenfit = tenmodel.fit(data_smooth, mask=mask)
qf = tenfit.quadratic_form
print(qf.shape)
eigvals, eigvecs = dti.decompose_tensor(qf)
fa = tenfit.fa
md = tenfit.md
rd = tenfit.rd
ad = tenfit.ad


axial_slice = 9

fig1, ax = plt.subplots(1, 4, figsize=(12, 6),
                        subplot_kw={'xticks': [], 'yticks': []})

fig1.subplots_adjust(hspace=0.3, wspace=0.05)

ax.flat[0].imshow(fa[:, :, axial_slice].T, cmap='gray',
                  vmin=0, vmax=0.7, origin='lower')
ax.flat[0].set_title('FA (DTI)')
ax.flat[1].imshow(md[:, :, axial_slice].T, cmap='gray',
                  vmin=0, vmax=2.0e-3, origin='lower')
ax.flat[1].set_title('MD (DTI)')
ax.flat[2].imshow(ad[:, :, axial_slice].T, cmap='gray',
                  vmin=0, vmax=2.0e-3, origin='lower')
ax.flat[2].set_title('AD (DTI)')
ax.flat[3].imshow(rd[:, :, axial_slice].T, cmap='gray',
                  vmin=0, vmax=2.0e-3, origin='lower')
ax.flat[3].set_title('RD (DTI)')

plt.show()
fig1.savefig('Diffusion_tensor_measures_from_DTI.png')