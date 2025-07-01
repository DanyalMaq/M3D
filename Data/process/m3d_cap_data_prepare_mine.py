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
    mtf.CropForegroundd(keys=["pet", "ct", "con"], source_key="pet"),
    mtf.Resized(keys=["pet", "ct", "con"], spatial_size=[32,256,256],
                mode=['trilinear', 'trilinear', 'nearest'])
])

def crop_image_around_lesion(
    image: np.ndarray,
    mask_image: np.ndarray,
    margin: int = 60,
    clip_val_min: float = 0,
    clip_val_max: float = 12,
    normalize: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Crops the Medical scan and segmentation mask around the lesion along the z-axis.

    Parameters:
    - image (np.ndarray): Medical scan of shape (1, Z, Y, X)
    - mask_image (np.ndarray): Segmentation mask of shape (1, Z, Y, X)
    - margin (int): Number of slices to include above and below the lesion
    - clip_val_min (float): Minimum PET value for clipping
    - clip_val_max (float): Maximum PET value for clipping
    - normalize (bool): Whether to normalize PET to [0, 1] after clipping

    Returns:
    - cropped_pet (np.ndarray): Cropped and normalized PET scan
    - cropped_mask (np.ndarray): Cropped mask
    """
    assert image.shape == mask_image.shape, "PET and mask must have the same shape"
    assert image.ndim == 4, "Expected shape (1, Z, Y, X)"

    # Remove batch dimension
    image, mask = image[0], mask_image[0]

    # Clip and normalize PET
    if normalize:
        image = np.clip(image, clip_val_min, clip_val_max)
        image = (image - clip_val_min) / (clip_val_max - clip_val_min)

    # Detect lesion bounds
    lesion_slices = np.any(mask, axis=(1, 2))
    if not lesion_slices.any():
        raise ValueError("No lesion found in the segmentation mask.")

    z_indices = np.where(lesion_slices)[0]
    center = (z_indices[0] + z_indices[-1]) // 2  # faster than mean for span
    start = max(0, center - margin)
    end = min(image.shape[0], center + margin + 1)

    # Crop and reintroduce batch dimension
    return (
        image[np.newaxis, start:end],
        mask[np.newaxis, start:end]
    )


def process_item(item, root_dir, output_dir):
    item_path = os.path.join(root_dir, item)
    if not os.path.isdir(item_path):
        print("Error with", item_path)
        return

    required_files = {
        "pet": os.path.join(item_path, "pet.nii.gz"),
        "mask": os.path.join(item_path, "mask.nii.gz"),
        "ct": os.path.join(item_path, "ct.nii.gz"),
        "text": os.path.join(item_path, "text.txt"),
    }

    if not all(os.path.exists(path) for path in required_files.values()):
        print(f"Skipping {item} — missing one or more required files.")
        return

    # Output directory
    output_item_dir = os.path.join(output_dir, item)
    os.makedirs(output_item_dir, exist_ok=True)

    # Load and transpose all images in one loop
    image_data = {}
    for key in ["pet", "mask", "ct"]:
        nii = nib.load(required_files[key])
        data = nii.get_fdata().transpose(2, 1, 0)[np.newaxis, ...]  # Shape: (1, z, y, x)
        image_data[key] = data

    # Crop CT and PET/mask around lesion
    ct_cropped, _ = crop_image_around_lesion(image_data["ct"], image_data["mask"], 80, -300, 400)
    pet_cropped, mask_cropped = crop_image_around_lesion(image_data["pet"], image_data["mask"], 80, 0, 12)

    # Apply MONAI transforms
    transformed = transforms({
        "pet": pet_cropped,
        "ct": ct_cropped,
        "con": mask_cropped
    })

    # Save processed outputs
    np.save(os.path.join(output_item_dir, "pet.npy"), transformed["pet"])
    np.save(os.path.join(output_item_dir, "ct.npy"), transformed["ct"])
    np.save(os.path.join(output_item_dir, "contour.npy"), transformed["con"])

    # Copy text file
    shutil.copy(required_files["text"], os.path.join(output_item_dir, "text.txt"))

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
