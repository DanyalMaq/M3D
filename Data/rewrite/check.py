import numpy as np
import nibabel as nib

path = "/home/dxm060/M3D/Data/patient_data/p010012_01/seg.nii.gz"

data = nib.load(path).get_fdata()
print(data)
print(np.unique(data))