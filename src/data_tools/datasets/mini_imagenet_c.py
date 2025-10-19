import os
import torch
from typing import Callable, Optional
import pandas as pd
from torchvision.datasets import VisionDataset

from src.data_tools.sqsdataset import SQSDataset


class MiniImageNetC(SQSDataset, VisionDataset):
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

        self.domain_to_id = {v: k for k, v in self.id_to_domain.items()}

        # Get images and labels
        from configs.dataset_config import SPECS_ROOT

        images_df = pd.read_csv(SPECS_ROOT / f"{split}_images.csv").assign(
            image_paths=lambda df: df.apply(
                lambda row: os.path.join(root, *row), axis=1
            )
        )
        # self.images = np.stack(
        #     [
        #         load_image_as_array(image_path, image_size)
        #         for image_path in tqdm(images_df.image_paths)
        #     ]
        # )
        self.images = torch.load(os.path.join(root, f"mini-{split}-images.pt"))
        self.class_list = images_df.class_name.unique()
        self.id_to_class = dict(enumerate(self.class_list))
        self.class_to_id = {v: k for k, v in self.id_to_class.items()}
        self.labels = list(images_df.class_name.map(self.class_to_id))
        # torch.save(self.images,os.path.join(root,f"mini-{split}-images.pt"),pickle_protocol = 4)
