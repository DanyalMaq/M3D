import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

import json
import pandas as pd

import monai.transforms as mtf
from monai.data import load_decathlon_datalist
from monai.data import set_track_meta

from ..utils.utils import mask2box
from .dataset_info import dataset_info
from .prompt_templates import Caption_templates, PosREC_templates, PosREG_templates, Seg_template
from .term_dictionary import term_dict


class MyCapDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        self.data_root = args.data_root
        self.image_tokens = "<im_patch>" * args.proj_out_num

        with open(args.cap_data_path, 'r') as f:
            self.data_list = json.load(f)[mode]

        self.caption_prompts = Seg_template

        set_track_meta(False)
        self.transform = self._build_transform()

    def _build_transform(self):
        base = [
            mtf.ToTensord(keys=self.args.modality_keys, dtype=torch.float)
        ]
        if self.mode == "train":
            aug = [
                mtf.RandScaleIntensityd(keys=self.args.modality_keys, factors=0.1, prob=0.2),
                mtf.RandShiftIntensityd(keys=self.args.modality_keys, offsets=0.1, prob=0.2)
            ]
            return mtf.Compose(aug + base)
        return mtf.Compose(base)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        for attempt in range(10):
            try:
                data = self.data_list[idx]

                # Load image data
                item = {}
                for key in self.args.modality_keys:
                    path = os.path.join(self.data_root, data[key])
                    item[key] = np.load(path)

                item = self.transform(item)

                # Read text
                text_path = os.path.join(self.data_root, data["text"])
                with open(text_path, 'r') as f:
                    answer = f.read().strip()

                prompt = random.choice(self.caption_prompts)
                question = self.image_tokens + prompt

                # Tokenize full prompt + answer
                full_tokens = self.tokenizer(
                    question + ' ' + answer,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                input_ids = full_tokens["input_ids"][0]
                attn_mask = full_tokens["attention_mask"][0]

                # Insert <eos> if needed
                valid_len = attn_mask.sum()
                if valid_len < len(input_ids):
                    input_ids[valid_len] = self.tokenizer.eos_token_id

                # Tokenize question only
                question_tokens = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                question_len = question_tokens["attention_mask"][0].sum()

                # Prepare label
                label = input_ids.clone()
                label[:question_len] = -100
                label[label == self.tokenizer.pad_token_id] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id and valid_len < len(label):
                    label[valid_len] = self.tokenizer.eos_token_id

                ret = {
                    **{k: item[k] for k in self.args.modality_keys},
                    'input_id': input_ids,
                    'label': label,
                    'attention_mask': attn_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "Caption"
                }

                if self.args.seg_enable:
                    ret['seg'] = torch.zeros_like(item[self.args.modality_keys[0]])

                return ret

            except Exception as e:
                print(f"[Warning] Error at idx {idx} (attempt {attempt + 1}): {e}")
                idx = random.randint(0, len(self.data_list) - 1)

        raise RuntimeError(f"Failed to load item after 2 attempts at index {idx}")


class RefSegDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90d(keys=["image", "seg"], prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=0),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensord(keys=["image"], dtype=torch.float),
                    mtf.ToTensord(keys=["seg"], dtype=torch.int),
                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.data_list = pd.read_csv(args.refseg_data_train_path, engine='python')
            self.transform = train_transform
        elif mode == 'validation':
            self.data_list = pd.read_csv(args.refseg_data_test_path, engine='python')
            self.transform = val_transform
        elif mode == 'test':
            self.data_list = pd.read_csv(args.refseg_data_test_path, engine='python')
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list.iloc[idx]
                image_path = os.path.join(self.args.data_root, data["Image"])

                image_array = np.load(image_path)  # 1*32*256*256, normalized

                seg_path = os.path.join(self.args.data_root, data["Mask"])
                seg_array = np.load(seg_path)
                seg_array = (seg_array == data["Mask_ID"]).astype(np.int8)

                item = {
                    "image": image_array,
                    "seg": seg_array,
                }

                it = self.transform(item)

                image = it['image']
                seg = it['seg']  # C*D*H*W

                question = data["Question"]
                question = self.image_tokens + ' ' + question

                answer = data["Answer"]

                self.tokenizer.padding_side = "right"
                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'seg': seg,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "refseg",
                }

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class TextDatasets(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        super(TextDatasets, self).__init__()
        self.ds_list = [
            CapDataset(args, tokenizer, mode),
            VQADataset(args, tokenizer, close_ended=True, mode=mode),
            VQADataset(args, tokenizer, close_ended=False, mode=mode),
        ]
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class UniDatasets(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        super(UniDatasets, self).__init__()
        self.ds_list = [
            MyCapDataset(args, tokenizer, mode),
            # RefSegDataset(args, tokenizer, mode)
        ]
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]



