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
from utils import crop_image_around_lesion

transforms = mtf.Compose([
    # mtf.SpatialCropd(keys=["image", "con"], roi_start=[100, 0, 0], roi_end=[350, 200, 200]),
    mtf.CropForegroundd(keys=["pet", "ct", "seg"], source_key="pet"),
    mtf.Resized(keys=["pet", "ct", "seg"], spatial_size=[32,256,256],
                mode=['trilinear', 'trilinear', 'nearest'])
])

def extract_and_save_labels(mask_path, output_folder):
    """
    Splits a multi-label segmentation mask into individual binary masks and saves them.

    Returns:
    Saves the individual masks in a folder in the same dir
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

def separate_segmentations(patient_dir, output_dir):
    """
    Processes a single patient directory to extract label masks.
    """
    seg_path = os.path.join(patient_dir, "seg.nii.gz")
    print(seg_path)
    if os.path.exists(seg_path):
        segmentations_dir = os.path.join(output_dir, "segmentations")
        extract_and_save_labels(seg_path, segmentations_dir)
        print(f"✅ Processed: {os.path.basename(patient_dir)}")
    else:
        print(f"⛔ Skipped: {os.path.basename(patient_dir)} (missing mask.nii.gz)")
    
def process_item(item, root_dir, output_dir):
    """
    Args:
        item (str): Name of the patient subdirectory to process.
        root_dir (str): Root directory containing all patient subdirectories.
        output_dir (str): Directory where processed outputs will be saved.
    """
    # Got a patient folder
    item_path = os.path.join(root_dir, item) 
    # Output directory
    output_item_dir = os.path.join(output_dir, item)
    os.makedirs(output_item_dir, exist_ok=True)
    # separate_segmentations(item_path, item_path)

    # Exit if not a dir
    if not os.path.isdir(item_path):
        print("Error with", item_path, "Not a dir for some reason")
        return

    required_files = {
        "pet": os.path.join(item_path, "pet.nii.gz"),
        "ct": os.path.join(item_path, "ct.nii.gz")
    }

    required_folders = {
        "seg": os.path.join(item_path, "segmentations")
    }

    if not all(os.path.exists(path) for path in required_files.values()):
        print(f"Skipping {item} — missing one or more required files.")
        return

    image_data = {}
    for key in ["pet", "ct"]:
        nii = nib.load(required_files[key])
        data = nii.get_fdata().transpose(2, 1, 0)[np.newaxis, ...]  # Shape: (1, z, y, x)
        image_data[key] = data

    # Load and transpose all images in one loop
    # print(os.listdir(required_folders['seg']))
    for seg in os.listdir(required_folders['seg']):
        # print(seg)
        number = int(seg.split('.')[0])
        seg_path = os.path.join(required_folders['seg'], seg)
        nii = nib.load(seg_path)
        data = nii.get_fdata().transpose(2, 1, 0)[np.newaxis, ...]  # Shape: (1, z, y, x)
        seg_data = data

        # Crop CT and PET/mask around lesion
        ct_cropped, _ = crop_image_around_lesion(image_data["ct"], seg_data, 80, -300, 400)
        pet_cropped, mask_cropped = crop_image_around_lesion(image_data["pet"], seg_data, 80, 0, 12)

        transformed = transforms({
            "pet": pet_cropped,
            "ct": ct_cropped,
            "seg": mask_cropped
        })

        save_dir = os.path.join(output_item_dir, str(number))
        os.makedirs(save_dir, exist_ok=True)

        # Save processed outputs
        np.save(os.path.join(save_dir, "pet.npy"), transformed["pet"])
        np.save(os.path.join(save_dir, "ct.npy"), transformed["ct"])
        np.save(os.path.join(save_dir, f"{number}.npy"), transformed["seg"])

    # Figure out text part
    # shutil.copy(required_files["text"], os.path.join(output_item_dir, "text.txt"))

    print(f"✅ Processed and saved: {item}")

def separate_segmentations_parallel(root_dir):
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
        futures = [executor.submit(separate_segmentations, pd) for pd in patient_dirs]
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            results.append(f.result())

    for r in sorted(results):
        print(r)

def create_dataset_parallel(root_dir, output_dir):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit each directory processing task
        futures = [executor.submit(process_item, item, root_dir, output_dir) 
                   for item in os.listdir(root_dir)]
        # Wait for all futures to complete
        for future in concurrent.futures.as_completed(futures):
            pass  # You can add error handling or other tasks here if needed

if __name__ == "__main__":
    # Change these to point to your actual folders
    # separate_segmentations_parallel("../patient_data/")
    # separate_segmentations_parallel("../patient_data/")

    create_dataset_parallel("../patient_data/", "../data/patient_data/")
    print("Transformation complete training.")

    # create_dataset_parallel("../training/", "../data/training_npy_ct/")
    # print("Transformation complete training.")

    # create_dataset_parallel("../testing/", "../data/testing_npy_ct/")
    # print("Transformation complete testing.")
