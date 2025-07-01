import os
import numpy as np
import json
import nibabel as nib
from PIL import Image
import concurrent.futures
from tqdm import tqdm
from collections import Counter
import unicodedata
import monai.transforms as mtf
from multiprocessing import Pool
import shutil
from unidecode import unidecode

def extract_and_save_labels(mask_path, output_folder):
    """
    Splits a multi-label segmentation mask into individual binary masks and saves them.
    """
    os.makedirs(output_folder, exist_ok=True)

    mask_nii = nib.load(mask_path)
    mask_data = mask_nii.get_fdata().astype(np.uint8).transpose(2, 1, 0)
    unique_labels = np.unique(mask_data)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background

    for label in unique_labels:
        binary = (mask_data == label).astype(np.uint8)
        label_nii = nib.Nifti1Image(binary.transpose(2, 1, 0), affine=mask_nii.affine)
        nib.save(label_nii, os.path.join(output_folder, f"{int(label)}.nii.gz"))

def process_patient_folder(patient_dir):
    """
    Processes a single patient directory to extract label masks.
    """
    mask_path = os.path.join(patient_dir, "mask.nii.gz")
    if os.path.exists(mask_path):
        segmentations_dir = os.path.join(patient_dir, "segmentations")
        extract_and_save_labels(mask_path, segmentations_dir)
        return f"✅ Processed: {os.path.basename(patient_dir)}"
    else:
        return f"⛔ Skipped: {os.path.basename(patient_dir)} (missing mask.nii.gz)"

def process_all_patients_parallel(root_dir):
    """
    Runs segmentation extraction for all patient folders in parallel.
    """
    patient_dirs = [
        os.path.join(root_dir, name)
        for name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, name))
    ]

    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_patient_folder, pd) for pd in patient_dirs]
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            results.append(f.result())

    for r in sorted(results):
        print(r)

if __name__ == "__main__":
    # Change these to point to your actual folders
    process_all_patients_parallel("../training/")
    process_all_patients_parallel("../testing/")
