import os
import json

def create_json_from_directory(train, test):
    data = {"train": [], "validation": []}
    
    # Traverse the directory and its subdirectories
    print(train)
    for subdir, _, files in os.walk(train):
        pet_npy_file = None
        contour_npy_file = None
        txt_file = None
        
        
        # Identify the .npy and .txt files
        for file in files:
            if file.endswith('.npy') and file.startswith('pet'):
                pet_npy_file = os.path.join(subdir, file)
            elif file.endswith('.txt'):
                txt_file = os.path.join(subdir, file)
            elif file.endswith('.npy') and file.startswith('contour'):
                contour_npy_file = os.path.join(subdir, file)
            elif file.endswith('.npy') and file.startswith('ct'):
                ct_npy_file = os.path.join(subdir, file) 
        
        # If both .npy and .txt files exist in the subdir, add to the data
        if pet_npy_file and txt_file and contour_npy_file and ct_npy_file:
            data["train"].append({
                "pet": pet_npy_file,
                "contour": contour_npy_file,
                "ct": ct_npy_file,
                "text": txt_file
            })
        
    for subdir, _, files in os.walk(test):
        pet_npy_file = None
        contour_npy_file = None
        txt_file = None

        
        # Identify the .npy and .txt files
        for file in files:
            if file.endswith('.npy') and file.startswith('pet'):
                pet_npy_file = os.path.join(subdir, file)
            elif file.endswith('.txt'):
                txt_file = os.path.join(subdir, file)
            elif file.endswith('.npy') and file.startswith('contour'):
                contour_npy_file = os.path.join(subdir, file) 
            elif file.endswith('.npy') and file.startswith('ct'):
                ct_npy_file = os.path.join(subdir, file) 
        
        # If both .npy and .txt files exist in the subdir, add to the data
        if pet_npy_file and txt_file and contour_npy_file and ct_npy_file:
            data["validation"].append({
                "pet": pet_npy_file,
                "contour": contour_npy_file,
                "ct": ct_npy_file,
                "text": txt_file
            })
    
    # Save the data to a JSON file
    with open("output.json", "w") as json_file:
        json.dump(data, json_file, indent=4)

# Replace 'your_directory_path' with the path to the root directory
create_json_from_directory('../data/training_npy_ct', '../data/testing_npy_ct')
