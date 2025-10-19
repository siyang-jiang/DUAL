from functools import partial
import json
import os
import random
from PIL import Image
import pickle
from typing import Any, Callable, Optional

import numpy as np
import torch
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
from configs.dataset_specs.cifar_100_c.perturbation_params import PERTURBATION_PARAMS
from src.data_tools.samplers import BeforeCorruptionSampler
from src.data_tools.transform import TransformLoader
from src.data_tools.utils import get_perturbations, get_query_perturbations


import torchvision.transforms.functional as TF


class CIFAR100CMeta(CIFAR100):
    def __init__(
        self,
        root: str,
        split: str,
        image_size: int,
        target_transform: Optional[Callable] = None,
        download: bool = True,  # Siyang Changed
        SIMCLR: Optional[Callable] = False,
        two_stream: Optional[Callable] = False,
        SIMCLR_val: bool = False,
        R_pertubation_level: int = -1,
    ):
        self.two_stream = two_stream
        self.split = split
        self.SIMCLR_val = SIMCLR_val
        transform = TransformLoader(image_size).get_composed_transform(
            aug=SIMCLR
        )  # Wait, what is this??
        self.toTensor = transforms.ToTensor()
        self.transform_test = TransformLoader(image_size).get_composed_transform(
            aug=False
        )

        super(CIFAR10, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        downloaded_list = self.train_list + self.test_list

        self._load_meta()

        # We need to write this import here (and not at the top) to avoid cyclic imports
        from configs.dataset_config import SPECS_ROOT

        with open(SPECS_ROOT / f"{split}.json", "r") as file:
            self.split_specs = json.load(file)

        self.class_to_idx = {
            class_name: self.class_to_idx[class_name]
            for class_name in self.split_specs["class_names"]
        }
        self.id_to_class = {v: k for k, v in self.class_to_idx.items()}

        self.images: Any = []
        self.labels = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                items_to_keep = [
                    item
                    for item in range(len(entry["data"]))
                    if entry["fine_labels"][item] in self.class_to_idx.values()
                ]
                self.images.append([entry["data"][item] for item in items_to_keep])
                self.labels.extend(
                    [entry["fine_labels"][item] for item in items_to_keep]
                )

        self.images = np.vstack(self.images).reshape(-1, 3, 32, 32)
        self.images = self.images.transpose((0, 2, 3, 1))  # convert to HWC

        self.perturbations, self.id_to_domain = get_perturbations(
            self.split_specs["perturbations"], PERTURBATION_PARAMS, image_size
        )
        self.another_query_perturbations = False
        self.duplicate_times = 2
        if self.two_stream and self.split == "test":  # SIMCLR val
            self.SIMCLR = True
            self.duplicate_times = 10
        else:
            self.SIMCLR = False

    def __len__(self):
        return (
            len(self.images) * self.duplicate_times
        )  # * len(self.perturbations) ← this is the original code, it will duplicate the dataset length for each perturbation

    def _torture(self, image, perturbations):
        for p in perturbations:
            image = p(image)
        return image

    def __getitem__(self, item):
        if self.SIMCLR:
            original_data_index = item // self.duplicate_times
        else:
            original_data_index = (
                item - len(self.images) if item >= len(self.images) else item
            )
        perturbation_list = [np.array]
        for types in self.perturbations.keys():
            if random.random() < 0.5:  # Hyperparameter
                perturbation_list.append(random.choice(self.perturbations[types]))
                # TODO: maybe a look-up table can make it faster

        # perturbation_index = item % len(self.perturbations)
        perturbation_index = item // len(self.images)
        img, label = (
            Image.fromarray(self.images[original_data_index]),
            self.labels[original_data_index],
        )
        img_p = self._torture(img, perturbation_list)

        if self.transform is not None:
            # TODO: some perturbations output arrays, some output images. We need to clean that.
            # Fixed, but very sllllloooowwwww.
            if isinstance(img_p, np.ndarray):
                img_p = img_p.astype(np.uint8)
                img_p = Image.fromarray(img_p)
            img_p = self.transform(img_p)
            img1 = self.transform_test(img) if self.SIMCLR_val else self.transform(img)
            img2 = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.two_stream:
            images_batch = [img1, img_p]
            if (
                self.split == "test"
            ):  # img 1 is from transform_test when SIMCLR_val is True
                images_batch[1] = img2

            query_image_p = img_p
            if self.another_query_perturbations:
                for types in self.perturbations.keys():
                    if random.random() < 0.5:
                        perturbation_list.append(
                            random.choice(self.perturbations[types])
                        )
                query_image_p = self._torture(img, perturbation_list)
            images_batch.append(query_image_p)
            for i, img in enumerate(images_batch):
                images_batch[i] = (
                    self.toTensor(img) if type(img) != torch.Tensor else img
                )
            return images_batch, label, perturbation_index
        else:
            return img_p, label, perturbation_index

    def get_sampler(self):
        return partial(BeforeCorruptionSampler, self)


# import torch
# from torch import nn
# from src.data_tools.perturbations import gaussian_noise
# from torchvision import transforms
# class CIFAR100CMetaShift(CIFAR100CMeta):
#     def gaussian_transfrom(self,input):
#         input = gaussian_noise(
#             input, severity_params=5e-4, image_size=32)
#         return torch.tensor(input, dtype=torch.float32)

#     def __init__(self,*args,**kwargs):
#         LEVEL_DICT = {
#             "resize": [2, 4, 8],
#             "blur": [3, 5, 7],
#             "noise": [1e-4, 5e-4, 1e-3],
#             "colorjitter": [0.5, 0.75, 1],
#         }
#         super().__init__(*args,**kwargs)
#         self.shifts = [
#             nn.Sequential(transforms.Resize(32//4),
#                         transforms.Resize(32)),
#             transforms.GaussianBlur(
#                 kernel_size=LEVEL_DICT["blur"][1]),
#             partial(self.gaussian_transfrom),
#             transforms.ColorJitter(brightness=LEVEL_DICT["colorjitter"][1],
#                                 contrast=LEVEL_DICT["colorjitter"][1],
#                                 saturation=LEVEL_DICT["colorjitter"][1])
#         ]

#     def __getitem__(self, item):
#         original_data_index = item // len(self.perturbations)
#         perturbation_index = item % len(self.perturbations)
#         shift_index = torch.randint(0,4,(1,)).squeeze(0)
#         img = Image.fromarray(self.images[original_data_index])

#         img_p = self.perturbations[perturbation_index](img)

#         if self.transform is not None:
#             # TODO: some perturbations output arrays, some output images. We need to clean that.
#             if isinstance(img_p, np.ndarray):
#                 img_p = img_p.astype(np.uint8)
#                 img_p = Image.fromarray(img_p)
#             img_p = self.transform(img_p)

#         images = torch.tensor([])
#         label = torch.tensor([])
#         for i in range(4):
#             shifted_images = self.shifts[i]images
#             img = torch.cat([img,shifted_images.to(device)])
#             label = torch.cat([label,torch.ones([batch[0].shape[0]],device=device)*i])

#         return img_p, label, perturbation_index
