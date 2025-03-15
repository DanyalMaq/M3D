import os
import json

def create_json_from_directory(train, test):
    data = {"train": [], "validation": []}
    
    # Traverse the directory and its subdirectories
    for subdir, _, files in os.walk(train):
        image_npy_file = None
        contour_npy_file = None
        txt_file = None
        
        
        # Identify the .npy and .txt files
        for file in files:
            if file.endswith('.npy') and file.startswith('image'):
                image_npy_file = os.path.join(subdir, file)
            elif file.endswith('.txt'):
                txt_file = os.path.join(subdir, file)
            elif file.endswith('.npy') and file.startswith('contour'):
                contour_npy_file = os.path.join(subdir, file) 
        
        # If both .npy and .txt files exist in the subdir, add to the data
        if image_npy_file and txt_file and contour_npy_file:
            data["train"].append({
                "image": image_npy_file,
                "contour": contour_npy_file,
                "text": txt_file
            })
        
    for subdir, _, files in os.walk(test):
        image_npy_file = None
        contour_npy_file = None
        txt_file = None

        
        # Identify the .npy and .txt files
        for file in files:
            if file.endswith('.npy') and file.startswith('image'):
                image_npy_file = os.path.join(subdir, file)
            elif file.endswith('.txt'):
                txt_file = os.path.join(subdir, file)
            elif file.endswith('.npy') and file.startswith('contour'):
                contour_npy_file = os.path.join(subdir, file) 
        
        # If both .npy and .txt files exist in the subdir, add to the data
        if image_npy_file and txt_file and contour_npy_file:
            data["validation"].append({
                "image": image_npy_file,
                "contour": contour_npy_file,
                "text": txt_file
            })
    
    # Save the data to a JSON file
    with open("output.json", "w") as json_file:
        json.dump(data, json_file, indent=4)

# Replace 'your_directory_path' with the path to the root directory
create_json_from_directory('/home/dxm060/docker/M3D/Data/data/training_npy', '/home/dxm060/docker/M3D/Data/data/testing_npy')
