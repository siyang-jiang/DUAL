from pathlib import Path

from src.data_tools.datasets.meta_h5_dataset import (
    FullMetaDatasetH5,
    FullMetaDatasetH5_ERM,
)

ERM_DATASET = FullMetaDatasetH5_ERM
DATASET = FullMetaDatasetH5
IMAGE_SIZE = 128
DATA_ROOT = Path("data/meta_dataset")
SPECS_ROOT = Path("configs/dataset_specs/meta_dataset")
CLASSES = None
