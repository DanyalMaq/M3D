'''
Download Merlin and test the model on sample data that is downloaded from huggingface
'''
import os
import warnings
import torch
from torch.utils.data import Dataset, DataLoader

from merlin.data import download_sample_data
# from merlin.data import DataLoader
from merlin.data import MyCapDataset, DataCollator
from merlin import Merlin


warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Merlin(device=device)
model = model.to(device)
model.eval()
# model.cuda()

data_dir = os.path.join(os.path.dirname(__file__), "abct_data")
cache_dir = data_dir.replace("abct_data", "abct_data_cache")
ex = "/home/dxm060/M3D/Data/patient_data/p002749_01/pet.nii.gz"
datalist = [
    {
        "image": download_sample_data(data_dir), # function returns local path to nifti file
        "text": "Lower thorax: A small low-attenuating fluid structure is noted in the right cardiophrenic angle in keeping with a tiny pericardial cyst."
        "Liver and biliary tree: Normal. Gallbladder: Normal. Spleen: Normal. Pancreas: Normal. Adrenal glands: Normal. "
        "Kidneys and ureters: Symmetric enhancement and excretion of the bilateral kidneys, with no striated nephrogram to suggest pyelonephritis. "
        "Urothelial enhancement bilaterally, consistent with urinary tract infection. No renal/ureteral calculi. No hydronephrosis. "
        "Gastrointestinal tract: Normal. Normal gas-filled appendix. Peritoneal cavity: No free fluid. "
        "Bladder: Marked urothelial enhancement consistent with cystitis. Uterus and ovaries: Normal. "
        "Vasculature: Patent. Lymph nodes: Normal. Abdominal wall: Normal. "
        "Musculoskeletal: Degenerative change of the spine.",
    },
    {
        "image": download_sample_data(data_dir), # function returns local path to nifti file
        "text": "Lower thorax: A small low-attenuating fluid structure is noted in the right cardiophrenic angle in keeping with a tiny pericardial cyst."
        "Liver and biliary tree: Normal. Gallbladder: Normal. Spleen: Normal. Pancreas: Normal. Adrenal glands: Normal. "
        "Kidneys and ureters: Symmetric enhancement and excretion of the bilateral kidneys, with no striated nephrogram to suggest pyelonephritis. "
        "Urothelial enhancement bilaterally, consistent with urinary tract infection. No renal/ureteral calculi. No hydronephrosis. "
        "Gastrointestinal tract: Normal. Normal gas-filled appendix. Peritoneal cavity: No free fluid. "
        "Bladder: Marked urothelial enhancement consistent with cystitis. Uterus and ovaries: Normal. "
        "Vasculature: Patent. Lymph nodes: Normal. Abdominal wall: Normal. "
        "Musculoskeletal: Degenerative change of the spine.",
    },
]

datalist = [
    {
        "image": "mym3d/Data/patient_data/p002749_01/pet.nii.gz",
        "contour": "mym3d/Data/patient_data/p002749_01/mask.nii.gz"
    },
    {
        "image": "mym3d/Data/patient_data/p002749_01/pet.nii.gz",
        "contour": "mym3d/Data/patient_data/p002749_01/mask.nii.gz"
    }
]

# dataloader = DataLoader(
#     datalist=datalist,
#     cache_dir=cache_dir,
#     batchsize=8,
#     shuffle=True,
#     num_workers=0,
# )

dataset = MyCapDataset(datalist)
data_collator = DataCollator()
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=data_collator)
    
## Get the Image Embeddings
model = Merlin(ImageEmbedding=True)
model = model.to(device)
model.eval()
# model.cuda()

for batch in dataloader:
    images = torch.cat([_.unsqueeze(0) for _ in batch["images"]], dim=0)
    outputs = model(
        batch["image"].to(device), 
        )
    print(f"\n================== Output Shapes ==================")
    print(f"Image embeddings shape (Can be used for downstream tasks): {outputs[0].shape}")
    break
    
