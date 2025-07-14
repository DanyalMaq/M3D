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
from utils import crop_image_around_lesion, focal_crop_around_mask

def image_info(name, img):
    # pass
    print("-----------------------")
    print(f"{name} shape:", img.shape)
    print(f"{name} unique values:", np.unique(img))
    print(f"{name} sum", np.sum(img))
    print("-----------------------")

transforms = mtf.Compose([
    # mtf.SpatialCropd(keys=["image", "mask"], roi_start=[100, 0, 0], roi_end=[350, 200, 200]),
    mtf.CropForegroundd(keys=["pet", "mask", "ct"], source_key="pet"),
    mtf.Resized(keys=["pet", "mask", "ct"], spatial_size=[32,256,256],
                mode=['trilinear', 'nearest', 'trilinear'])
])

transforms_focal = mtf.Compose([
    # mtf.SpatialCropd(keys=["image", "mask"], roi_start=[100, 0, 0], roi_end=[350, 200, 200]),
    mtf.CropForegroundd(keys=["pet_focal", "mask_focal", "ct_focal"], source_key="pet_focal"),
    mtf.Resized(keys=["pet_focal", "mask_focal", "ct_focal"], spatial_size=[32,256,256],
                mode=['trilinear', 'nearest', 'trilinear'])
])

def process_item(item, root_dir, output_dir):
    item_path = os.path.join(root_dir, item)
    if not os.path.isdir(item_path):
        print("Error with", item_path)
        return

    required_files = {
        "pet": os.path.join(item_path, "pet.nii.gz"),
        "mask": os.path.join(item_path, "mask.nii.gz"),
        "ct": os.path.join(item_path, "ct.nii.gz"),
        # "text": os.path.join(item_path, "text.txt"),
    }

    if not all(os.path.exists(path) for path in required_files.values()):
        print(f"Skipping {item} — missing one or more required files.")
        return

    # Output directory
    output_item_dir = os.path.join(output_dir, item)
    # os.makedirs(output_item_dir, exist_ok=True)

    # Load and transpose all images in one loop
    image_data = {}
    for key in ["pet", "mask", "ct"]:
        nii = nib.load(required_files[key])
        data = nii.get_fdata().transpose(2, 1, 0)[np.newaxis, ...]  # Shape: (1, z, y, x)
        image_data[key] = data

    # Crop CT and PET/mask around lesion
    ct_cropped, _ = crop_image_around_lesion(image_data["ct"], image_data["mask"], 80, -300, 400)
    pet_cropped, mask_cropped = crop_image_around_lesion(image_data["pet"], image_data["mask"], 80, 0, 12)
    pet_focal, mask_focal, ct_focal = focal_crop_around_mask(pet_cropped, mask_cropped, ct_cropped)

    # Apply MONAI transforms
    transformed = transforms({
        "pet": pet_cropped,
        "mask": mask_cropped,
        "ct": ct_cropped
    })
    transformed_focal = transforms_focal({
        "pet_focal": pet_focal,
        "mask_focal": mask_focal,
        "ct_focal": ct_focal
    })

    # Save processed outputs
    np.save(os.path.join(output_item_dir, "pet.npy"), transformed["pet"])
    np.save(os.path.join(output_item_dir, "ct.npy"), transformed["ct"])
    np.save(os.path.join(output_item_dir, "mask.npy"), transformed["mask"])
    np.save(os.path.join(output_item_dir, "pet_focal.npy"), transformed_focal["pet_focal"])
    np.save(os.path.join(output_item_dir, "mask_focal.npy"), transformed_focal["mask_focal"])
    np.save(os.path.join(output_item_dir, "ct_focal.npy"), transformed_focal["ct_focal"])

    # Copy text file
    # shutil.copy(required_files["text"], os.path.join(output_item_dir, "text.txt"))

    print(f"✅ Processed and saved: {item}")


def create_dataset_parallel(root_dir, output_dir):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit each directory processing task
        futures = [executor.submit(process_item, item, root_dir, output_dir) 
                   for item in os.listdir(root_dir)]
        # Wait for all futures to complete
        for future in concurrent.futures.as_completed(futures):
            pass  # You can add error handling or other tasks here if needed

create_dataset_parallel("../training/", "../data/training_npy_ct/")
print("Transformation complete training.")

create_dataset_parallel("../testing/", "../data/testing_npy_ct/")
print("Transformation complete testing.")

# for item in os.listdir("../training/"):
#     process_item(item, "../training/", "../data/training_npy/")
#     print("Transformation complete training.")

# create_dataset("../testing/", "../data/testing_npy/")
# print("Transformation complete training.")
