import os
import argparse

from totalsegmentator.python_api import totalsegmentator

def segment(args):
    BASE_DIR = '/mym3d/Data/patient_data'
    print(os.listdir(BASE_DIR))

    for root, dirs, files in os.walk(BASE_DIR):
        print(root, dirs, files)
        if 'ct.nii.gz' in files:
            ct_path = os.path.join(root, 'ct.nii.gz')
            seg_path = os.path.join(root, 'seg.nii.gz')
            
            totalsegmentator(ct_path, seg_path, ml=True, fast=True)
            
            try:
                pass
            except Exception as e:
                print(f"Error processing {root}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate formatted radiology information.")
    # parser.add_argument("--model_path", type=str)

    args = parser.parse_args()

    segment(args)

if __name__ == "__main__":
    main()

