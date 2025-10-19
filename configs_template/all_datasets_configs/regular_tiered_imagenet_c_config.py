from pathlib import Path

from src.data_tools.datasets import TieredImageNetC

"""
Config for tieredImageNet-C using raw ILSVRC2015 images.
"""

DATASET = TieredImageNetC
DATA_ROOT = Path("data/tired_imagenet_pt")
IMAGE_SIZE = 84
SPECS_ROOT = Path("configs/dataset_specs/tiered_imagenet_c")
CLASSES = {"train": 351, "val": 97, "test": 160}
