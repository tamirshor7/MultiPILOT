import SimpleITK as sitk
import pydicom
import numpy as np
from .sitk import FileSave as stikSave
import os

def ReadSeries(folder_path, returnIDs=False):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(folder_path)
    imgs = []
    sIDs = []
    for sid in series_ids:
        try:
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(folder_path, sid)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            imgs.append(sitk.GetArrayFromImage(image))
            sIDs.append(sid)
        except:
            pass
    imgs = np.array(imgs)
    if not returnIDs:
        return imgs
    else:
        return imgs, sIDs

def ReadDICOMDIR(dicomdir_path, returnIDs=False):
    ds = pydicom.dcmread(dicomdir_path)
    imgs = []
    sIDs = []

    # Iterate through the PATIENT records
    for patient in ds.patient_records:
        pID = f"{patient.PatientID}_{patient.PatientName}"
        # Find all the STUDY records for the patient
        studies = [
            ii for ii in patient.children if ii.DirectoryRecordType == "STUDY"
        ]
        for study in studies:
            sID = f"{study.StudyID}"
            # Find all the SERIES records in the study
            all_series = [
                ii for ii in study.children if ii.DirectoryRecordType == "SERIES"
            ]
            for series in all_series:
                # Find all the IMAGE records in the series
                images = [
                    ii for ii in series.children
                    if ii.DirectoryRecordType == "IMAGE"
                ]

                # Get the absolute file path to each instance                
                elems = [ii["ReferencedFileID"] for ii in images] # Each IMAGE contains a relative file path to the root directory
                paths = [[ee.value] if ee.VM == 1 else ee.value for ee in elems] # Make sure the relative file path is always a list of str
                paths = [os.path.join(os.path.dirname(dicomdir_path), os.sep.join(p)) for p in paths]
                
                try:
                    reader = sitk.ImageSeriesReader()
                    reader.SetFileNames(paths)
                    image = reader.Execute()
                    imgs.append(sitk.GetArrayFromImage(image))
                    sIDs.append(f"{pID}_{sID}_{series.SeriesInstanceUID}")
                except:
                    pass

    imgs = np.array(imgs)
    if not returnIDs:
        return imgs
    else:
        return imgs, sIDs

def FileRead(file_path, return_data=True, return_ds=False):
    ds = pydicom.dcmread(file_path)
    if return_data and not return_ds:
        return ds.pixel_array
    elif not return_data and return_ds:
        return ds
    else:
        return ds.pixel_array, ds

def toNIFTI(dicom_path, nifti_path, isSeries=True):
    if isSeries:
        imgs, IDs = ReadSeries(dicom_path, returnIDs=True)
    else:
        imgs, IDs = ReadDICOMDIR(dicom_path, returnIDs=True)
    for i, (im, id) in enumerate(zip(imgs, IDs)):
        stikSave(im, f"{nifti_path}/{id}.nii.gz")