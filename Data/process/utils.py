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

def crop_image_around_lesion(
    image: np.ndarray,
    mask_image: np.ndarray,
    margin: int = 60,
    clip_val_min: float = 0,
    clip_val_max: float = 12,
    normalize: bool = True,
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

def image_info(name, img):
    # pass
    print("-----------------------")
    print(f"{name} shape:", img.shape)
    print(f"{name} unique values:", np.unique(img))
    print(f"{name} sum", np.sum(img))
    print("-----------------------")