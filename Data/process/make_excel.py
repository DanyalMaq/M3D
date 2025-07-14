import os
import json

def collect_entries_from_dir(root_dir, text_key="struct"):
    """
    Collects file paths for pet, ct, mask, optional *_focal versions, and a text file.

    Parameters:
    - root_dir: str, directory to walk through
    - text_key: str, "text" or "struct" — determines which .txt file to collect

    Returns:
    - list of dictionaries with file paths
    """
    entries = []

    for subdir, _, files in os.walk(root_dir):
        paths = {
            "pet": None,
            "ct": None,
            "mask": None,
            "pet_focal": None,
            "ct_focal": None,
            "mask_focal": None,
            "text": None,
        }

        for file in files:
            full_path = os.path.join(subdir, file)

            if file.endswith(".npy"):
                for key in ["pet", "ct", "mask"]:
                    if file.startswith(f"{key}_focal"):
                        paths[f"{key}_focal"] = full_path
                        break
                    elif file.startswith(key):
                        paths[key] = full_path
                        break

            elif file == f"{text_key}.txt":
                paths["text"] = full_path

        # Require base images and the requested txt file
        if paths["pet"] and paths["ct"] and paths["mask"] and paths["text"]:
            entries.append({k: v for k, v in paths.items() if v is not None})

    return entries

def create_json_from_directories(input_dirs, output_file="../output.json", text_key="text"):
    """
    Creates a JSON from directories containing medical data.

    Parameters:
    - input_dirs: dict of split names to paths (e.g., {"train": "path", "test": "path"})
    - output_file: str, path to output JSON
    - text_key: str, "text" or "struct"
    """
    data = {
        split: collect_entries_from_dir(path, text_key=text_key)
        for split, path in input_dirs.items()
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

# Example usage
create_json_from_directories(
    {
        "train": "../data/training_npy_ct",
        "test": "../data/testing_npy_ct"
    },
    output_file="../output.json",
    text_key="struct"  # Use "text" or "struct" here
)
