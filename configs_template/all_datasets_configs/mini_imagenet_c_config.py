from pathlib import Path

from src.data_tools.datasets import MiniImageNetC

DATASET = MiniImageNetC
DATA_ROOT = Path("data/mini-pt")  #
IMAGE_SIZE = 224
SPECS_ROOT = Path("configs/dataset_specs/mini_imagenet_c")
CLASSES = {"train": 64, "val": 16, "test": 20}
