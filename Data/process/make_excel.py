import os
import json

def collect_entries_from_dir(root_dir):
    entries = []

    for subdir, _, files in os.walk(root_dir):
        pet_npy_file = None
        ct_npy_file = None
        mask_npy_file = None
        txt_file = None

        for file in files:
            if file.endswith('.npy'):
                if file.startswith('pet'):
                    pet_npy_file = os.path.join(subdir, file)
                elif file.startswith('ct'):
                    ct_npy_file = os.path.join(subdir, file)
                else:
                    mask_npy_file = os.path.join(subdir, file)
            elif file.endswith('.txt'):
                txt_file = os.path.join(subdir, file)

        if pet_npy_file and ct_npy_file and mask_npy_file and txt_file:
            entries.append({
                "pet": pet_npy_file,
                "ct": ct_npy_file,
                "mask": mask_npy_file,
                "text": txt_file
            })

    return entries

def create_json_from_directory(train_dir, output_file="../output.json"):
    data = {
        "train": collect_entries_from_dir(train_dir)
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

# Run the function
create_json_from_directory('../data/testing_npy_ct')