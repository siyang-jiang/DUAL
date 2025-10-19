import os
import pickle
from typing import Callable, Optional
from torchvision.datasets import CIFAR10, CIFAR100
from src.data_tools.sqsdataset import SQSDataset
import numpy as np


class CIFAR100CMeta(SQSDataset, CIFAR100):
    def __init__(
        self,
        root: str,
        split: str,
        image_size: int,
        target_transform: Optional[
            Callable
        ] = None,  # This target means label, not image
        two_stream: Optional[Callable] = False,
        SIMCLR: Optional[Callable] = False,
        SIMCLR_val: bool = False,
        perturbation_twice: Optional[Callable] = False,
        no_pertubated: Optional[Callable] = False,
    ):
        SQSDataset.__init__(
            self,
            split,
            image_size,
            target_transform,
            two_stream,
            SIMCLR,
            SIMCLR_val,
            perturbation_twice,
            no_pertubated,
        )
        CIFAR10.__init__(
            self,
            root,
            transform=self.transform,
            target_transform=target_transform,
            download=True,
        )

        self.download()
        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )
        downloaded_list = self.train_list + self.test_list
        self._load_meta()

        self.class_to_idx = {
            class_name: self.class_to_idx[class_name]
            for class_name in self.split_specs["class_names"]
        }
        self.id_to_class = {v: k for k, v in self.class_to_idx.items()}

        self.images = []
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

        # ! Temp, Baseline training only
        # convert labels into class indices
        if no_pertubated:
            import torch

            label_tensor = torch.tensor(self.labels)
            unique_labels = label_tensor.unique()
            self.label_dict = {label.item(): i for i, label in enumerate(unique_labels)}
            self.labels = [self.label_dict[label] for label in self.labels]
