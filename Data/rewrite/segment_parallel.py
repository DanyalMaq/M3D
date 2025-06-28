import os
import argparse
from concurrent.futures import ProcessPoolExecutor
from totalsegmentator.python_api import totalsegmentator

def run_segment(ct_path, seg_path, device):
    print(f"Running on {device}: {ct_path}")
    try:
        totalsegmentator(ct_path, seg_path, ml=True, fast=True, device=device)
    except Exception as e:
        print(f"Error processing {ct_path}: {e}")

def run_segment_wrapper(args):
    return run_segment(*args)

def segment():
    BASE_DIR = '/mym3d/Data/training'
    all_cases = []

    # Gather all ct.nii.gz paths
    for root, dirs, files in os.walk(BASE_DIR):
        if 'ct.nii.gz' in files:
            ct_path = os.path.join(root, 'ct.nii.gz')
            seg_path = os.path.join(root, 'seg.nii.gz')
            all_cases.append((ct_path, seg_path))

    # Run in batches of 18
    batches = 18
    for i in range(0, len(all_cases), batches):
        batch = all_cases[i:i+batches]
        tasks = []
        for j, (ct_path, seg_path) in enumerate(batch):
            if j < batches / 3:
                device = "gpu:0"
            elif j < (2 * batches) / 3:
                device = "gpu:1"
            else:
                device = "gpu:2"
            tasks.append((ct_path, seg_path, device))

        with ProcessPoolExecutor(max_workers=18) as executor:
            executor.map(run_segment_wrapper, tasks)

def main():
    parser = argparse.ArgumentParser(description="Parallel TotalSegmentator")
    args = parser.parse_args()
    segment()

if __name__ == "__main__":
    main()
