import os
import h5py
from PIL import Image

import torch
from src.data_tools.datasets.meta_h5_dataset import perturbator
from torchvision import transforms


class MetaValDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, num_episodes=4000, SIMCLR=False):
        super().__init__()
        self.image_size = 128
        self.num_episodes = num_episodes
        self.h5_path = h5_path
        self.h5_file = None
        self.perturbator = perturbator(
            self.image_size, f"configs/dataset_specs/meta_dataset/test.json"
        )
        self.resize = transforms.Resize((self.image_size, self.image_size))
        self.to_tensor = transforms.ToTensor()
        print("MetaValDataset init done")
        print(f"num_episodes: {num_episodes}")

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")

        record = self.h5_file[str(idx)]
        support_images = torch.tensor(record["sx"][()])
        support_labels = torch.tensor(record["sy"][()])
        query_images = torch.tensor(record["x"][()])
        query_labels = torch.tensor(record["y"][()])

        if (
            support_images.shape[2] != self.image_size
            or support_images.shape[3] != self.image_size
        ):
            support_images = (
                self.resize(self.to_tensor(support_images)).numpy().transpose(1, 2, 0)
            )
            query_images = (
                self.resize(self.to_tensor(query_images)).numpy().transpose(1, 2, 0)
            )

        for i, img in enumerate(support_images):
            support_images[i] = self.perturbator.torture(img)

        for i, img in enumerate(query_images):
            query_images[i] = self.perturbator.torture(img)

        return support_images, support_labels, query_images, query_labels, 0, 0, 0


if __name__ == "__main__":
    dset = MetaValDataset("../../tf_records/ilsvrc_2012_v2/val_episodes4000.h5")

    data_loader_val = torch.utils.data.DataLoader(
        dset, batch_size=1, num_workers=4, pin_memory=True, drop_last=False
    )

    sx, sy, x, y = next(iter(data_loader_val))
    print(sx.shape)
    print(x.shape)
