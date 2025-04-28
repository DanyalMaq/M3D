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

def image_info(name, img):
    # pass
    print("-----------------------")
    print(f"{name} shape:", img.shape)
    print(f"{name} unique values:", np.unique(img))
    print(f"{name} sum", np.sum(img))
    print("-----------------------")

transforms = mtf.Compose([
    # mtf.SpatialCropd(keys=["image", "con"], roi_start=[100, 0, 0], roi_end=[350, 200, 200]),
    mtf.CropForegroundd(keys=["image", "con"], source_key="image"),
    mtf.Resized(keys=["image", "con"], spatial_size=[112,256,352],
                mode=['trilinear', 'nearest'])
])

def crop_pet_around_lesion(pet_image: np.ndarray, mask_image: np.ndarray, margin: int = 60):
    """
    Crops the PET scan and segmentation mask around the lesion along the z-axis.

    Parameters:
    - pet_image (np.ndarray): PET scan of shape (1, 350, 200, 200)
    - mask_image (np.ndarray): Segmentation mask of shape (1, 350, 200, 200)
    - margin (int): Number of slices to include above and below the lesion

    Returns:
    - cropped_pet (np.ndarray): Cropped PET scan
    - cropped_mask (np.ndarray): Cropped segmentation mask
    """
    assert pet_image.shape == mask_image.shape, "PET and mask must have the same shape"
    assert pet_image.ndim == 4, "Expected input shape (1, z, y, x)"

    # Remove batch dimension temporarily for easier handling
    pet = pet_image[0]
    mask = mask_image[0]
    clip_val = 12
    clipped = np.clip(pet, 0, clip_val)
    # Step 2: Normalize to [0, 1]
    pet = clipped / clip_val

    # Find lesion indices along z-axis
    lesion_slices = np.any(mask, axis=(1, 2))  # shape: (350,)
    lesion_indices = np.where(lesion_slices)[0]

    if lesion_indices.size == 0:
        raise ValueError("No lesion found in the segmentation mask.")

    lesion_center = int(np.mean(lesion_indices))
    start_slice = max(0, lesion_center - margin)
    end_slice = min(pet.shape[0], lesion_center + margin + 1)

    # Crop
    cropped_pet = pet[start_slice:end_slice, :, :]
    cropped_mask = mask[start_slice:end_slice, :, :]

    # Add batch dimension back
    cropped_pet = cropped_pet[np.newaxis, ...]
    cropped_mask = cropped_mask[np.newaxis, ...]

    return cropped_pet, cropped_mask

def process_item(item, root_dir, output_dir):
    item_path = os.path.join(root_dir, item)
    if os.path.isdir(item_path):
        pet_file = os.path.join(item_path, "pet.nii.gz")
        mask_file = os.path.join(item_path, "mask.nii.gz")
        if os.path.exists(pet_file) and os.path.exists(mask_file):
            output_item_dir = os.path.join(output_dir, item)
            os.makedirs(output_item_dir, exist_ok=True)
            dir = os.path.join(output_item_dir, "image.npy")
            pet_image = nib.load(pet_file).get_fdata().transpose(2, 1, 0)[np.newaxis, ...]
            mask_image = nib.load(mask_file).get_fdata().transpose(2, 1, 0)[np.newaxis, ...]
            pet_image, mask_image = crop_pet_around_lesion(pet_image, mask_image, 60)

            pair = {
                "image": pet_image,
                "con": mask_image,
            }

            items = transforms(pair)
            image = items['image']
            contour = items['con']

            np.save(os.path.join(output_item_dir, "image.npy"), image)
            np.save(os.path.join(output_item_dir, "contour.npy"), contour)

            shutil.copyfile(item_path+"/text.txt", output_item_dir+"/text.txt")
            
            print(f"Transformed and saved: {item} to {dir}")
            
            # affine = np.eye(4)
            # data = pet_image[0]
            # nifti_data = data.transpose(1, 2, 0)
            # nifti_img = nib.Nifti1Image(nifti_data, affine)
            # nib.save(nifti_img, os.path.join(output_item_dir, "image.nii.gz"))
            # data = mask_image[0]
            # nifti_data = data.transpose(1, 2, 0)
            # nifti_img = nib.Nifti1Image(nifti_data, affine)
            # nib.save(nifti_img, os.path.join(output_item_dir, "mask.nii.gz"))

            # Uncomment if you want to transform and save a combined text file
            # json_file_path = item_path+"/text.json"
            # txt_file_path = output_item_dir+"/text.txt"
            # with open(json_file_path, 'r') as json_file:
            #     data = json.load(json_file)
            # combined_text = '\n'.join(data.values())
            # with open(txt_file_path, 'w') as txt_file:
            #     txt_file.write(combined_text)
        else:
            print(f"Missing pet.nii.gz or mask.nii.gz in: {item}")

def create_dataset_parallel(root_dir, output_dir):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit each directory processing task
        futures = [executor.submit(process_item, item, root_dir, output_dir) 
                   for item in os.listdir(root_dir)]
        # Wait for all futures to complete
        for future in concurrent.futures.as_completed(futures):
            pass  # You can add error handling or other tasks here if needed

create_dataset_parallel("../training/", "../data/training_npy/")
print("Transformation complete training.")

create_dataset_parallel("../testing/", "../data/testing_npy/")
print("Transformation complete testing.")

# for item in os.listdir("../training/"):
#     process_item(item, "../training/", "../data/training_npy/")
#     print("Transformation complete training.")

# create_dataset("../testing/", "../data/testing_npy/")
# print("Transformation complete training.")
