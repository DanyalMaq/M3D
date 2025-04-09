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
    mtf.SpatialCropd(keys=["image", "con"], roi_start=[100, 0, 0], roi_end=[350, 200, 200]),
    mtf.CropForegroundd(keys=["image", "con"], source_key="image"),
    mtf.Resized(keys=["image", "con"], spatial_size=[32,256,256],
                mode=['trilinear', 'nearest'])
])

def create_dataset(root_dir, output_dir):
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            pet_file = os.path.join(item_path, "pet.nii.gz")
            mask_file = os.path.join(item_path, "mask.nii.gz")
            if os.path.exists(pet_file) and os.path.exists(mask_file):
                output_item_dir = os.path.join(output_dir, item)
                os.makedirs(output_item_dir, exist_ok=True)
                dir = os.path.join(output_item_dir, "image.npy")
                print(f"Transformed and saved: {item} to {dir}")
                pet_image = nib.load(pet_file).get_fdata().transpose(2, 0, 1)[np.newaxis, ...]
                mask_image = nib.load(mask_file).get_fdata().transpose(2, 0, 1)[np.newaxis, ...]

                # print(np.max(pet_image))
                # print(np.max(mask_image))
                # print(np.unique(mask_image, return_counts=True))

                # combined_image = pet_image + (mask_image*255)
                # combined_image = np.clip(combined_image, a_min=None, a_max=255.0)
                mask_image *= 255
                # image_info("pet", pet_image)
                # image_info("mask", mask_image)
                # image_info("combined", combined_image)

                # image = transforms(combined_image)
                # image_info("transformed", image)
                # image_info("pet", pet_image)
                # image_info("mask", mask_image)
                pair = {
                    "image": pet_image,
                    "con": mask_image,
                }

                items = transforms(pair)
                image = items['image']
                contour = items['con']

                # print(np.max(image))
                # print(np.max(seg))

                np.save(os.path.join(output_item_dir, "image.npy"), image)
                np.save(os.path.join(output_item_dir, "contour.npy"), contour)

                # Convert the json into a combined text file
                # json_file_path = item_path+"/text.json"
                # txt_file_path = output_item_dir+"/text.txt"

                # with open(json_file_path, 'r') as json_file:
                #     data = json.load(json_file)

                # # Combine all the text values into one string
                # combined_text = '\n'.join(data.values())  # Joins all the text with a newline between each

                # # Save the combined text into a text file
                # with open(txt_file_path, 'w') as txt_file:
                #     txt_file.write(combined_text)

                shutil.copyfile(item_path+"/text.txt", output_item_dir+"/text.txt")
                
                # dir = os.path.join(output_item_dir, "image.npy")
                # print(f"Transformed and saved: {item} to {dir}")
            else:
                print(f"Missing pet.nii.gz or mask.nii.gz in: {item}")

def process_item(item, root_dir, output_dir):
    item_path = os.path.join(root_dir, item)
    if os.path.isdir(item_path):
        pet_file = os.path.join(item_path, "pet.nii.gz")
        mask_file = os.path.join(item_path, "mask.nii.gz")
        if os.path.exists(pet_file) and os.path.exists(mask_file):
            output_item_dir = os.path.join(output_dir, item)
            os.makedirs(output_item_dir, exist_ok=True)
            dir = os.path.join(output_item_dir, "image.npy")
            print(f"Transformed and saved: {item} to {dir}")
            pet_image = nib.load(pet_file).get_fdata().transpose(2, 0, 1)[np.newaxis, ...]
            mask_image = nib.load(mask_file).get_fdata().transpose(2, 0, 1)[np.newaxis, ...]

            mask_image *= 255

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

# create_dataset("../training/", "../data/training_npy/")
# print("Transformation complete training.")

# create_dataset("../testing/", "../data/testing_npy/")
# print("Transformation complete training.")
