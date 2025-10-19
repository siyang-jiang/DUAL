import json
from functools import partial
from pathlib import Path

import os
import torch
from PIL import Image
from typing import Callable, Optional

import numpy as np
import pandas as pd
from loguru import logger
from torchvision.datasets import VisionDataset
from torchvision import transforms
from tqdm import tqdm
from src.data_tools.sqsdataset import SQSDataset
from configs.dataset_specs.tiered_imagenet_c.perturbation_params import (
    PERTURBATION_PARAMS,
)
from src.data_tools.samplers import AfterCorruptionSampler, BeforeCorruptionSampler
from src.data_tools.transform import TransformLoader
from src.data_tools.utils import get_perturbations, load_image_as_array


class TieredImageNetC(SQSDataset, VisionDataset):
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
        )
        VisionDataset.__init__(
            self, root, transform=self.transform, target_transform=target_transform
        )
        # We need to write this import here (and not at the top) to avoid cyclic imports
        from configs.dataset_config import SPECS_ROOT

        with open(SPECS_ROOT / f"{split}.json", "r") as file:
            split_specs = json.load(file)

        self.root = Path(root)
        self.class_list = split_specs["class_names"]
        self.id_to_class = dict(enumerate(self.class_list))
        self.class_to_id = {v: k for k, v in self.id_to_class.items()}

        self.perturbations, self.id_to_domain = get_perturbations(
            split_specs["perturbations"], PERTURBATION_PARAMS, image_size
        )
        self.domain_to_id = {v: k for k, v in self.id_to_domain.items()}

        logger.info(f"Retrieving {split} images ...")
        ### Save images and labels to disk
        # self.images, self.labels = self.get_images_and_labels()
        # images_df = pd.DataFrame(
        #     {
        #         "class_id": self.labels,
        #         "img_name": [os.path.basename(x) for x in self.images],
        #     }
        # )
        # self.labels = np.array(self.labels)
        # pbar = tqdm(range(len(images_df)))
        # self.images = np.stack(
        #     [(load_image_as_array(str(self.root / self.id_to_class[row.class_id] / row.img_name), image_size)) for _, (_, row) in zip(pbar, images_df.iterrows())]
        # )

        # torch.save([self.labels, self.images], save_path, pickle_protocol=4)

        ## load images and labels from disk
        self.labels, self.images = torch.load(Path(root) / f"{split}_{image_size}.pt")

    def get_images_and_labels(self):
        image_names = []
        image_labels = []

        for class_id, class_name in enumerate(self.class_list):
            class_images_paths = [
                str(image_path)
                for image_path in (self.root / class_name).glob("*")
                if image_path.is_file()
            ]
            image_names += class_images_paths
            image_labels += len(class_images_paths) * [class_id]

        return image_names, image_labels
