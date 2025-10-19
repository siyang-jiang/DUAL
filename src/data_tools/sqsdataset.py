import json
import random
from functools import partial
from typing import Callable, Optional

from loguru import logger
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms


from configs.dataset_specs.cifar_100_c.perturbation_params import PERTURBATION_PARAMS

from src.data_tools.samplers import BeforeCorruptionSampler
from src.data_tools.transform import TransformLoader
from src.data_tools.utils import get_perturbations


class SQSDataset:
    def __init__(
        self,
        split: str,
        image_size: int,
        target_transform: Optional[
            Callable
        ] = None,  # This target means label, not image
        two_stream: Optional[Callable] = False,
        SIMCLR: Optional[Callable] = False,
        SIMCLR_val: bool = False,
        perturbation_twice: Optional[Callable] = False,
        no_pertubated: Optional[Callable] = False,  # for other baseline only
    ):
        """
        Returns:
            if two_stream is True, return a tuple of ([img1, img_p, img_p2], label, perturbation_index)
                img_p2 == img_p if perturbation_twice is False
            else, return a tuple of (img, label, perturbation_index)
            note that the pertubation_index is useless now
        """
        self.two_stream = two_stream
        self.split = split
        self.SIMCLR_val = SIMCLR_val
        self.target_transform = target_transform

        if no_pertubated:
            logger.warning("No pertubated mode is on!")

        self.no_pertubated = no_pertubated
        self.transform = TransformLoader(image_size).get_composed_transform(aug=SIMCLR)
        self.transform_test = TransformLoader(image_size).get_composed_transform(
            aug=False
        )

        self.toTensor = transforms.ToTensor()
        self.PILtoTensor = transforms.PILToTensor()

        self.images = []
        self.labels = []

        from configs.dataset_config import SPECS_ROOT

        split_file = f"{split}.json"  # Test class and train perturbation
        logger.info(f"Loading json split {split_file}...")
        with open(SPECS_ROOT / split_file, "r") as file:
            self.split_specs = json.load(file)

        self.perturbations, self.id_to_domain = get_perturbations(
            self.split_specs["perturbations"], PERTURBATION_PARAMS, image_size
        )
        self.perturbation_twice = perturbation_twice
        self.duplicate_times = (
            2 if not SIMCLR else 10
        )  # For SIMCLR loader, to avoid the dataset length is too short for an epoch.
        self.SIMCLR = SIMCLR
        from configs.experiment_config import PROPORTION  # avoid circular import
        from configs.experiment_config import MULTI_PERTUBATION

        self.multi_p_type = int(MULTI_PERTUBATION)
        self.p_rate = PROPORTION if MULTI_PERTUBATION == -1 else 1.0

    def __len__(self):
        return (
            len(self.images) * self.duplicate_times
        )  # * len(self.perturbations) ← this is the original code, it will duplicate the dataset length for each perturbation

    def _torture(self, image, perturbations):
        for p in perturbations:
            image = p(image)
        return image

    def _tensorize(self, image_batch):
        return [
            self.toTensor(img) if type(img) != torch.Tensor else img
            for img in image_batch
        ]

    def __getitem__(self, item):
        if self.SIMCLR:
            original_data_index = item // self.duplicate_times
        else:
            original_data_index = (
                item - len(self.images) if item >= len(self.images) else item
            )  # Ugly but worked
        perturbation_list = []
        pertubation_types = (
            random.sample(self.perturbations.keys(), self.multi_p_type)
            if self.multi_p_type > 0
            else self.perturbations.keys()
        )
        for types in pertubation_types:
            if random.random() < self.p_rate:  # Hyperparameter
                perturbation_list.append(random.choice(self.perturbations[types]))

        # perturbation_index = item % len(self.perturbations)
        perturbation_index = item // len(self.images)
        img, label = (
            self.images[original_data_index],
            self.labels[original_data_index],
        )  # [np.array, int]

        #! TEMP
        if self.no_pertubated:
            if self.split != "test":
                return TF.to_tensor(img), label
            else:
                return self._tensorize(self._torture(img, perturbation_list)), label
        img_p = self._torture(img, perturbation_list)  # np.array

        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.two_stream:
            if self.SIMCLR:
                # if isinstance(img_p, np.ndarray):
                #     img_p = img_p.astype(np.uint8)
                #     img_p = Image.fromarray(img_p)
                # img_p = self.transform(TF.to_pil_image(img_p))
                img = TF.to_pil_image(img)
                img1 = (
                    self.transform_test(img) if self.SIMCLR_val else self.transform(img)
                )  # Should the SIMCLR learn from the perturbated image?
                img2 = self.transform(img)

                np_img = np.array(img)  # TODO: Better transform
                images_batch = [
                    img1,
                    img2,
                    img,
                    self._torture(np_img, perturbation_list),
                ]  # [img1, img2, pure_img, pure_img_p]
                return self._tensorize(images_batch), label, perturbation_index
            else:
                images_batch = [img, img_p]
                if self.perturbation_twice:
                    for types in self.perturbations.keys():
                        if random.random() < 0.5:
                            perturbation_list.append(
                                random.choice(self.perturbations[types])
                            )
                    img_p2 = self._torture(img, perturbation_list)
                    images_batch.append(img_p2)  # [img1, img_p, img_p2]
                else:
                    images_batch.append(img_p)  # [img1, img_p, img_p]
                return self._tensorize(images_batch), label, perturbation_index
        else:
            return TF.to_tensor(img_p), label, perturbation_index

    def get_sampler(self):
        return partial(BeforeCorruptionSampler, self)
