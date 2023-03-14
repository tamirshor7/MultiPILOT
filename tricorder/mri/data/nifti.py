import numpy as np
import nibabel as nib

def FileRead(file_path, return_data=True, return_meta=False):
    img = nib.load(file_path)
    if return_data and not return_meta:
        return img.get_fdata()
    elif not return_data and return_meta:
        return img.header
    else:
        return img.get_fdata(), img.header

def FileSave(data, file_path, useV2=True):
    if useV2:
        nib.save(nib.Nifti2Image(data, np.eye(4)) , file_path)
    else:
        nib.save(nib.Nifti1Image(data, np.eye(4)) , file_path)
