import torch
from torch.utils.data import Dataset
import monai
from copy import deepcopy
import shutil
import tempfile
from pathlib import Path
from typing import List
from monai.utils import look_up_option
from monai.data.utils import SUPPORTED_PICKLE_MOD
from dataclasses import dataclass, field

from merlin.data.monai_transforms import ImageTransforms

import monai.transforms as mtf
from monai.data import set_track_meta
import random
import numpy as np

class CTPersistentDataset(monai.data.PersistentDataset):
    def __init__(self, data, transform, cache_dir=None):
        super().__init__(data=data, transform=transform, cache_dir=cache_dir)

        print(f"Size of dataset: {self.__len__()}\n")

    def _cachecheck(self, item_transformed):
        hashfile = None
        _item_transformed = deepcopy(item_transformed)
        image_path = item_transformed.get("image")
        image_data = {
            "image": item_transformed.get("image")
        }  # Assuming the image data is under the 'image' key

        if self.cache_dir is not None and image_data is not None:
            data_item_md5 = self.hash_func(image_data).decode(
                "utf-8"
            )  # Hash based on image data
            hashfile = self.cache_dir / f"{data_item_md5}.pt"

        if hashfile is not None and hashfile.is_file():
            cached_image = torch.load(hashfile)
            _item_transformed["image"] = cached_image
            return _item_transformed

        _image_transformed = self._pre_transform(image_data)["image"]
        _item_transformed["image"] = _image_transformed
        if hashfile is None:
            return _item_transformed
        try:
            # NOTE: Writing to a temporary directory and then using a nearly atomic rename operation
            #       to make the cache more robust to manual killing of parent process
            #       which may leave partially written cache files in an incomplete state
            with tempfile.TemporaryDirectory() as tmpdirname:
                temp_hash_file = Path(tmpdirname) / hashfile.name
                torch.save(
                    obj=_image_transformed,
                    f=temp_hash_file,
                    pickle_module=look_up_option(
                        self.pickle_module, SUPPORTED_PICKLE_MOD
                    ),
                    pickle_protocol=self.pickle_protocol,
                )
                if temp_hash_file.is_file() and not hashfile.is_file():
                    # On Unix, if target exists and is a file, it will be replaced silently if the user has permission.
                    # for more details: https://docs.python.org/3/library/shutil.html#shutil.move.
                    try:
                        shutil.move(str(temp_hash_file), hashfile)
                    except FileExistsError:
                        pass
        except PermissionError:  # project-monai/monai issue #3613
            pass
        return _item_transformed

    def _transform(self, index: int):
        pre_random_item = self._cachecheck(self.data[index])
        return self._post_transform(pre_random_item)


class DataLoader(monai.data.DataLoader):
    def __init__(
        self,
        datalist: List[dict],
        cache_dir: str,
        batchsize: int,
        shuffle: bool = True,
        num_workers: int = 0,
    ):
        self.datalist = datalist
        self.cache_dir = cache_dir
        self.batchsize = batchsize
        self.dataset = CTPersistentDataset(
            data=datalist,
            transform=ImageTransforms,
            cache_dir=cache_dir,
        )
        super().__init__(
            self.dataset,
            batch_size=batchsize,
            shuffle=shuffle,
            num_workers=num_workers,
        )


@dataclass
class DataCollator:
    def __init__(self):
        pass
    def __call__(self, batch: list) -> dict:
        images, contours, input_ids, labels, attention_mask = tuple(
            [b[key] for b in batch] for key in ('image', 'contour'))

        images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
        contours = torch.cat([_.unsqueeze(0) for _ in contours], dim=0)

        return_dict = dict(
            images=images,
            contours=contours
        )

        return return_dict

class MyCapDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        self.data_root
        train_transform = mtf.Compose(
            [
                mtf.RandRotate90d(keys=["image", "con"], prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlipd(keys=["image", "con"], prob=0.10, spatial_axis=1),
                mtf.RandFlipd(keys=["image", "con"], prob=0.10, spatial_axis=2),
                mtf.RandFlipd(keys=["image", "con"], prob=0.10, spatial_axis=0),
                mtf.RandScaleIntensityd(keys=["image", "con"], factors=0.1, prob=0.5),
                mtf.RandShiftIntensityd(keys=["image", "con"], offsets=0.1, prob=0.5),

                mtf.ToTensord(keys=["image", "con"], dtype=torch.float),
            ]
        )
        val_transform = mtf.Compose(
                [
                    mtf.ToTensord(keys=["image", "con"], dtype=torch.float),
                ]
            )
        set_track_meta(False)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 5
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                image_path = data["image"]
                contour_path = data["contour"]
                image_abs_path = os.path.join(self.data_root, image_path)
                contour_abs_path = os.path.join(self.data_root, contour_path)

                image = np.load(image_abs_path)  # nomalized 0-1, C,D,H,W
                contour = np.load(contour_abs_path)  # nomalized 0-1, C,D,H,W
                # image = np.load(img_abs_path)[np.newaxis, ...]  # nomalized

                item = {
                    'image': image,
                    'con': contour,
                }

                it = self.transform(item)
                images = [it['image']]
                images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
                print(images.shape)
                contours = [it['con']]
                contours = torch.cat([_.unsqueeze(0) for _ in contours], dim=0)
                print(contours.shape)

                ret = {
                    'image': images,
                    'contour': contours,
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)